import ast
import inspect
import textwrap
from functools import update_wrapper
from .node import Node


class CalcContextState:
    """Tracks per-context caches and overrides."""

    def __init__(self):
        self.cache = {}
        self.overrides = {}

    def get_override(self, node):
        return self.overrides.get(node, _NO_VALUE)

    def get_cached(self, node):
        if node in self.overrides:
            return self.overrides[node]
        return self.cache.get(node, _NO_VALUE)

    def store_cached(self, node, value):
        if node not in self.overrides:
            self.cache[node] = value

    def set_override(self, node, value):
        self.overrides[node] = value
        # Clear cached calculations so dependents recompute within the context.
        self.cache.clear()


_NO_VALUE = object()
_context_stack = []
_layers_by_name = {}


def _get_current_context():
    if not _context_stack:
        return None
    return _context_stack[-1]


class calc_context:
    """Context manager to scope calculation caches and overrides."""

    def __enter__(self):
        state = CalcContextState()
        _context_stack.append(state)
        self._state = state
        return self

    def __exit__(self, exc_type, exc, tb):
        _context_stack.pop()
        self._state = None


class _CalcLayer:
    """Persistent context manager whose state can be reused across entries."""

    def __init__(self, name=None):
        self.name = name
        self._state = CalcContextState()

    @property
    def state(self):
        return self._state

    def __enter__(self):
        _context_stack.append(self._state)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not _context_stack or _context_stack[-1] is not self._state:
            raise RuntimeError("calc_layer context stack out of sync.")
        _context_stack.pop()


def calc_layer(name=None):
    """Creates or retrieves a persistent calculation layer."""

    if name is not None:
        layer = _layers_by_name.get(name)
        if layer is not None:
            return layer

    layer = _CalcLayer(name=name)

    if name is not None:
        _layers_by_name[name] = layer

    return layer

class Graph:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Graph, cls).__new__(cls)
            cls._instance.nodes = {}
            cls._instance.func_to_wrapper = {}
        return cls._instance

    def register_node(self, node):
        if node.id not in self.nodes:
            self.nodes[node.id] = node
        return self.nodes[node.id]

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def clear(self):
        self.nodes.clear()
        self.func_to_wrapper.clear()

graph = Graph()


class ExecutionTracker:
    """Tracks execution context to infer dynamic dependencies."""

    def __init__(self):
        self.stack = []

    def enter(self, node):
        context = {
            'node': node,
            'executed_children': set(),
        }
        self.stack.append(context)
        return context

    def exit(self):
        return self.stack.pop()

    def record_dependency(self, child_node):
        if not self.stack:
            return

        parent_context = self.stack[-1]
        parent_node = parent_context['node']
        if child_node is parent_node:
            return

        if child_node not in parent_node.children:
            parent_node.add_child(child_node, relation_type='input', conditional=False, dynamic=True)

        parent_context['executed_children'].add(child_node)


execution_tracker = ExecutionTracker()

class CalcNodeTransformer(ast.NodeTransformer):
    def __init__(self, owner_class, available_vars):
        self.owner_class = owner_class
        self.available_vars = available_vars  # Combined globals and closure nonlocals
        self.dependencies_map = {}
        self.dependencies = []
        self.context_stack = ['normal']

    # Context management helpers -------------------------------------------------
    def _push_context(self, context):
        self.context_stack.append(context)

    def _pop_context(self):
        self.context_stack.pop()

    def _in_context(self, context):
        return context in self.context_stack

    # Node visitors --------------------------------------------------------------
    def visit_If(self, node):
        self._push_context('predicate_test')
        node.test = self.visit(node.test)
        self._pop_context()

        self._push_context('predicate_body')
        node.body = [self.visit(stmt) for stmt in node.body]
        self._pop_context()

        if node.orelse:
            self._push_context('predicate_body')
            node.orelse = [self.visit(stmt) for stmt in node.orelse]
            self._pop_context()

        return node

    def visit_IfExp(self, node):
        self._push_context('predicate_test')
        node.test = self.visit(node.test)
        self._pop_context()

        self._push_context('predicate_body')
        node.body = self.visit(node.body)
        self._pop_context()

        self._push_context('predicate_body')
        node.orelse = self.visit(node.orelse)
        self._pop_context()

        return node

    def visit_While(self, node):
        self._push_context('predicate_test')
        node.test = self.visit(node.test)
        self._pop_context()

        self._push_context('predicate_body')
        node.body = [self.visit(stmt) for stmt in node.body]
        self._pop_context()

        if node.orelse:
            self._push_context('predicate_body')
            node.orelse = [self.visit(stmt) for stmt in node.orelse]
            self._pop_context()

        return node

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Call):
            self._push_context('attribute_base')
            node.value = self.visit(node.value)
            self._pop_context()
        else:
            node.value = self.visit(node.value)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)

        callee_wrapper = None
        # Case 1: Standalone function or closure call, e.g., `b()`
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            target = self.available_vars.get(func_name)
            if isinstance(target, calc_node):
                callee_wrapper = target
        # Case 2: Instance method call, e.g., `self.i()`
        elif (self.owner_class and isinstance(node.func, ast.Attribute) and
              isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'):
            method_name = node.func.attr
            target = getattr(self.owner_class, method_name, None)
            if isinstance(target, calc_node):
                callee_wrapper = target

        if not callee_wrapper:
            return node

        dependency_type = 'input'
        if self._in_context('predicate_test') or self._in_context('attribute_base'):
            dependency_type = 'edge'

        conditional = self._in_context('predicate_body')
        pre_evaluate = dependency_type == 'edge' or (dependency_type == 'input' and not conditional)

        entry = self.dependencies_map.get(callee_wrapper)
        if not entry:
            temp_var_name = f"__dep_{callee_wrapper.name}"
            entry = {
                'wrapper': callee_wrapper,
                'temp_var_name': temp_var_name,
                'dependency_type': dependency_type,
                'conditional': conditional,
                'pre_evaluate': pre_evaluate,
                'call_node': node if pre_evaluate else None,
            }
            self.dependencies_map[callee_wrapper] = entry
        else:
            if dependency_type == 'edge':
                entry['dependency_type'] = 'edge'
            entry['conditional'] = entry['conditional'] or conditional
            if pre_evaluate and not entry.get('pre_evaluate'):
                entry['pre_evaluate'] = True
                entry['call_node'] = node

        # If this dependency will be pre-evaluated, replace the call with the temp variable
        if entry.get('pre_evaluate'):
            return ast.Name(id=entry['temp_var_name'], ctx=ast.Load())

        return node

    def finalize(self):
        self.dependencies = list(self.dependencies_map.values())

class calc_node:
    def __init__(self, func):
        self.func = func
        self.owner = None
        self.name = func.__name__
        self.dependencies_info = []
        self.static_dependencies = []
        self.compiled_func = None
        self.analysis_done = False
        graph.func_to_wrapper[func] = self
        update_wrapper(self, self.func)

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name

    def _analyze_and_compile(self):
        if self.analysis_done:
            return

        source = textwrap.dedent(inspect.getsource(self.func))
        tree = ast.parse(source)

        func_def_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == self.func.__name__:
                func_def_node = node
                break

        if not func_def_node:
            raise TypeError(f"Could not find function definition for {self.name} in its source code.")

        # Combine globals and closure nonlocals for dependency resolution
        closure_vars = {**self.func.__globals__, **inspect.getclosurevars(self.func).nonlocals}
        transformer = CalcNodeTransformer(self.owner, closure_vars)
        transformer.visit(func_def_node)
        transformer.finalize()
        self.dependencies_info = transformer.dependencies

        # Maintain compatibility for tests that inspect static_dependencies
        seen_wrappers = set()
        self.static_dependencies = []
        for dep in self.dependencies_info:
            wrapper = dep['wrapper']
            if wrapper and wrapper not in seen_wrappers:
                self.static_dependencies.append(wrapper)
                seen_wrappers.add(wrapper)

        pre_eval_nodes = []
        for dep in self.dependencies_info:
            if dep.get('pre_evaluate') and dep.get('call_node') is not None:
                assign_node = ast.Assign(
                    targets=[ast.Name(id=dep['temp_var_name'], ctx=ast.Store())],
                    value=dep['call_node']
                )
                pre_eval_nodes.append(assign_node)

        func_def_node.body = pre_eval_nodes + func_def_node.body
        # Strip decorator to prevent re-application on exec
        func_def_node.decorator_list = []

        # Compile only the function definition in a new module to avoid side effects
        module_node = ast.Module(body=[func_def_node], type_ignores=[])
        ast.fix_missing_locations(module_node)

        # The namespace for exec needs all the variables the function might access
        exec_namespace = closure_vars.copy()

        exec(compile(module_node, inspect.getsourcefile(self.func), 'exec'), exec_namespace)
        self.compiled_func = exec_namespace[self.name]
        self.analysis_done = True

    def _get_node(self, instance=None):
        node_id = (self.func.__module__, self.name, id(instance)) if instance else (self.func.__module__, self.name)
        node = graph.get_node(node_id)
        if not node:
            self._analyze_and_compile()
            node = Node(self.func, self.name, instance)
            node = graph.register_node(node)
            dependency_map = {}
            for dep in self.dependencies_info:
                wrapper = dep['wrapper']
                if not wrapper:
                    continue
                info = dependency_map.setdefault(wrapper, {
                    'relation_type': dep['dependency_type'],
                    'conditional': dep['conditional'],
                })
                if dep['dependency_type'] == 'edge':
                    info['relation_type'] = 'edge'
                info['conditional'] = info['conditional'] or dep['conditional']

            for dep_wrapper, info in dependency_map.items():
                # For instance methods, get the corresponding node for the same instance
                dep_node = dep_wrapper._get_node(instance)
                node.add_child(dep_node, relation_type=info['relation_type'], conditional=info['conditional'])
        return node

    def _execute_node(self, node, instance, args, kwargs):
        if not self.analysis_done:
            self._analyze_and_compile()

        execution_tracker.enter(node)
        try:
            if instance:
                return self.compiled_func(instance, *args, **kwargs)
            return self.compiled_func(*args, **kwargs)
        finally:
            context = execution_tracker.exit()
            node.update_children_activity(context['executed_children'])

    def execute(self, instance, *args, **kwargs):
        node = self._get_node(instance)
        execution_tracker.record_dependency(node)
        context_state = _get_current_context()

        if context_state:
            override_value = context_state.get_override(node)
            if override_value is not _NO_VALUE:
                return override_value

            cached_value = context_state.get_cached(node)
            if cached_value is not _NO_VALUE:
                return cached_value

            if not node.is_dirty:
                value = node.result
                context_state.store_cached(node, value)
                return value

            result = self._execute_node(node, instance, args, kwargs)
            context_state.store_cached(node, result)
            return result

        if not node.is_dirty:
            return node.result

        result = self._execute_node(node, instance, args, kwargs)
        node.set_value(result)
        return result

    def __call__(self, *args, **kwargs):
        return self.execute(None, *args, **kwargs)

    def invalidate(self, instance=None):
        node = self._get_node(instance)
        if node:
            node.invalidate()

    def set_value(self, value, instance=None):
        node = self._get_node(instance)
        if node:
            node.set_value(value, manual=True)

    def override(self, value, instance=None):
        context_state = _get_current_context()
        if context_state is None:
            raise RuntimeError("override() must be used within a calc_context.")

        node = self._get_node(instance)
        if node:
            context_state.set_override(node, value)

    def describe_dependencies(self, instance=None):
        node = self._get_node(instance)
        return {key: list(value) for key, value in node.child_categories().items()}

    def __get__(self, instance, owner):
        if instance is None:
            return self

        class BoundMethod:
            def __init__(self, decorator, instance):
                self.decorator = decorator
                self.instance = instance
                update_wrapper(self, decorator.func)

            def __call__(self, *args, **kwargs):
                return self.decorator.execute(self.instance, *args, **kwargs)

            def invalidate(self):
                self.decorator.invalidate(self.instance)

            def set_value(self, value):
                self.decorator.set_value(value, self.instance)

            def override(self, value):
                self.decorator.override(value, self.instance)

            def describe_dependencies(self):
                return self.decorator.describe_dependencies(self.instance)

        return BoundMethod(self, instance)
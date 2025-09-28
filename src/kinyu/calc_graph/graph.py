import ast
import inspect
import textwrap
import types
from functools import partial, update_wrapper
from .node import Node

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

class CalcNodeTransformer(ast.NodeTransformer):
    def __init__(self, owner_class, available_vars):
        self.owner_class = owner_class
        self.available_vars = available_vars # Combined globals and closure nonlocals
        self.dependencies = []

    def visit_Call(self, node):
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

        if callee_wrapper:
            temp_var_name = f"__dep_{callee_wrapper.name}"
            # Add dependency only once
            if not any(dep[0] is callee_wrapper for dep in self.dependencies):
                self.dependencies.append((callee_wrapper, node, temp_var_name))
            # Replace the call with a variable name
            return ast.Name(id=temp_var_name, ctx=ast.Load())

        return self.generic_visit(node)

class calc_node:
    def __init__(self, func):
        self.func = func
        self.owner = None
        self.name = func.__name__
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
        self.static_dependencies = [dep[0] for dep in transformer.dependencies]

        pre_eval_nodes = []
        for dep_wrapper, original_call, temp_var_name in transformer.dependencies:
            assign_node = ast.Assign(
                targets=[ast.Name(id=temp_var_name, ctx=ast.Store())],
                value=original_call
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
            for dep_wrapper in self.static_dependencies:
                # For instance methods, get the corresponding node for the same instance
                dep_node = dep_wrapper._get_node(instance)
                node.add_child(dep_node)
        return node

    def execute(self, instance, *args, **kwargs):
        node = self._get_node(instance)
        if not node.is_dirty:
            return node.result

        if not self.analysis_done:
            self._analyze_and_compile()

        if instance:
            result = self.compiled_func(instance, *args, **kwargs)
        else:
            result = self.compiled_func(*args, **kwargs)

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
            node.set_value(value)

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

        return BoundMethod(self, instance)
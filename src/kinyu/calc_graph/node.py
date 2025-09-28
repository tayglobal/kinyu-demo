class Node:
    """
    Represents a node in the calculation graph. It stores the state,
    manages dependencies (children) and dependents (parents), and caches the result.
    """
    def __init__(self, func, name, instance=None):
        self.func = func
        self.name = name
        self.instance = instance
        self.id = self._get_id()
        self._result = None
        self._is_dirty = True
        # Nodes this node depends on (dependencies) stored with metadata
        self.children = {}  # child_node -> {relation_type, conditional, active, dynamic}
        self.parents = set()   # Nodes that depend on this node (dependents)

    def _get_id(self):
        """Generates a unique ID for the node using the decorator's name."""
        if self.instance:
            return (self.func.__module__, self.name, id(self.instance))
        return (self.func.__module__, self.name)

    def add_child(self, child_node, relation_type='input', conditional=False, dynamic=False):
        """Adds a dependency (child) with metadata and establishes the inverse relationship."""
        info = self.children.get(child_node)
        if info is None:
            self.children[child_node] = {
                'relation_type': relation_type,
                'conditional': conditional,
                'active': False,
                'dynamic': dynamic,
            }
            child_node.parents.add(self)
        else:
            # Merge metadata, upgrading to edge if required and preserving conditional flags.
            if relation_type == 'edge':
                info['relation_type'] = 'edge'
            info['conditional'] = info['conditional'] or conditional
            info['dynamic'] = info['dynamic'] or dynamic

    def update_children_activity(self, executed_children):
        """Marks which children executed during the latest evaluation."""
        for child, info in self.children.items():
            info['active'] = child in executed_children

    def child_categories(self):
        """Returns the children grouped by their current role in the graph."""
        categories = {
            'input_nodes': [],
            'edge_nodes': [],
            'predicate_nodes': [],
        }

        for child, info in self.children.items():
            relation_type = info.get('relation_type', 'input')
            is_active = info.get('active', False)
            is_conditional = info.get('conditional', False)
            is_dynamic = info.get('dynamic', False)

            if relation_type == 'edge':
                categories['edge_nodes'].append(child)
            elif is_active:
                categories['input_nodes'].append(child)
            elif is_conditional or is_dynamic:
                categories['predicate_nodes'].append(child)
            else:
                # Default to input if the dependency is unconditional but hasn't executed yet.
                categories['input_nodes'].append(child)

        return categories

    def invalidate(self):
        """Marks the node as dirty and recursively invalidates its parents."""
        if not self._is_dirty:
            self._is_dirty = True
            self._result = None
            for parent in self.parents:
                parent.invalidate()

    def set_value(self, value):
        """Sets the node's result, marks it as clean, and invalidates parents."""
        self._result = value
        self._is_dirty = False
        # Invalidate parents because their dependency has changed.
        for parent in self.parents:
            parent.invalidate()

    @property
    def is_dirty(self):
        """Returns True if the node needs re-computation."""
        return self._is_dirty

    @property
    def result(self):
        """Returns the cached result of the node."""
        return self._result

    def __repr__(self):
        return f"<Node id={self.id} dirty={self.is_dirty}>"
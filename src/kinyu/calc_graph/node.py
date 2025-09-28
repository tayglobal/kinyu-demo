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
        self.children = set()  # Nodes this node depends on (dependencies)
        self.parents = set()   # Nodes that depend on this node (dependents)

    def _get_id(self):
        """Generates a unique ID for the node using the decorator's name."""
        if self.instance:
            return (self.func.__module__, self.name, id(self.instance))
        return (self.func.__module__, self.name)

    def add_child(self, child_node):
        """Adds a dependency (child) and establishes the inverse relationship."""
        self.children.add(child_node)
        child_node.parents.add(self)

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
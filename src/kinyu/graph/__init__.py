import functools
from kinyu_graph import Graph

# Global graph instance
_graph = Graph()

def g_func(func):
    """
    A decorator that turns a Python function into a node in a dependency graph.
    """
    func_name = func.__name__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return _graph.execute(func, func_name, args, kwargs)

    return wrapper

def override_value(func_name, val, *args, **kwargs):
    """Overrides the value of a node in the graph."""
    return _graph.override_value(func_name, val, args, kwargs)

# Expose the main graph methods at the module level
clear_cache = _graph.clear
get_edges = _graph.get_edges
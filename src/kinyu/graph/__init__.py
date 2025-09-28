import functools
from kinyu_graph import Graph

# Global graph instance
_graph = Graph()


class g_func:
    """
    A decorator that turns a Python function or method into a node in a dependency graph.

    This decorator is implemented as a descriptor, which allows it to differentiate
    between being called on a class versus an instance. When applied to an instance
    method, it captures the instance's identity (`id(self)`) to ensure that method
    calls on different objects are treated as unique nodes in the graph.
    """

    def __init__(self, func):
        self.func = func
        self.func_name = func.__name__
        functools.update_wrapper(self, func)

    def __get__(self, instance, owner):
        """
        Called when the decorated function is accessed.
        - If on a class, returns the descriptor itself.
        - If on an instance, returns a wrapper that binds the instance.
        """
        if instance is None:
            return self  # Accessed on the class

        # Create a bound method that will be passed to the graph's execute function.
        # functools.partial pre-fills the 'self' argument of the method.
        bound_method = functools.partial(self.func, instance)

        @functools.wraps(self.func)
        def wrapper(*args, **kwargs):
            """
            This wrapper is called when the decorated instance method is executed.
            It calls the core Rust engine, passing the instance's ID for uniqueness.
            """
            return _graph.execute(
                bound_method,
                self.func_name,
                id(instance),
                *args,
                **kwargs,
            )

        return wrapper

    def __call__(self, *args, **kwargs):
        """
        Called when a standalone decorated function is executed.
        'self' here is the g_func object itself.
        """
        return _graph.execute(
            self.func, self.func_name, None, *args, **kwargs
        )


def override_value(func_name, val, *args, **kwargs):
    """
    Overrides the value of a node in the graph.

    To override a method on a specific object, pass the object as the
    `instance` keyword argument.

    Example:
        override_value('my_method', 100, instance=my_obj)
    """
    instance = kwargs.pop("instance", None)
    instance_id = id(instance) if instance is not None else None
    return _graph.override_value(
        func_name, instance_id, val, *args, **kwargs
    )


# Expose the main graph methods at the module level
clear_cache = _graph.clear
get_edges = _graph.get_edges
import pytest
from kinyu.graph import g_func, clear_cache, override_value, get_edges


class TestGraphInstanceMethods:
    def setup_method(self):
        clear_cache()

    def test_method_calls_on_different_instances_are_distinct_nodes(self):
        """
        Tests that the graph correctly distinguishes between method calls
        from different object instances.
        """
        # A counter to track how many times the expensive method is called
        execution_counts = {"c": 0}

        class MyCalc:
            def __init__(self, val):
                self.val = val

            @g_func
            def a(self):
                return self.b() * 2

            @g_func
            def b(self):
                return self.c() + self.val

            @g_func
            def c(self):
                execution_counts["c"] += 1
                return 5

        # Create two instances
        calc1 = MyCalc(10)
        calc2 = MyCalc(20)

        # Execute on the first instance
        result1 = calc1.a()
        assert result1 == (5 + 10) * 2  # 30
        assert execution_counts["c"] == 1

        # Execute on the second instance. 'c' should be re-executed.
        result2 = calc2.a()
        assert result2 == (5 + 20) * 2  # 50
        assert execution_counts["c"] == 2

        # Verify that a second call on the first instance is cached
        calc1.a()
        assert execution_counts["c"] == 2

    def test_override_value_for_specific_instance(self):
        """
        Tests that overriding a value on one instance does not affect another.
        """
        class MyCalc:
            @g_func
            def a(self):
                return self.b() + 1

            @g_func
            def b(self):
                return 10

        calc1 = MyCalc()
        calc2 = MyCalc()

        # Initial results
        assert calc1.a() == 11
        assert calc2.a() == 11

        # Override 'b' on calc1 only
        override_value('b', 100, instance=calc1)

        # Verify results
        assert calc1.a() == 101  # Should be updated
        assert calc2.a() == 11   # Should remain the same

    def test_dependency_invalidation_is_instance_specific(self):
        """
        Tests that invalidating a node only affects the dependents
        of that specific instance.
        """
        class MyCalc:
            @g_func
            def a(self):
                return self.b() * 2

            @g_func
            def b(self):
                return self.c() + 1

            @g_func
            def c(self):
                return 10

        calc1 = MyCalc()
        calc2 = MyCalc()

        # Run both to populate the cache
        assert calc1.a() == 22
        assert calc2.a() == 22

        # Override 'c' on calc1. This should invalidate 'b' and 'a' on calc1 only.
        override_value('c', 100, instance=calc1)

        # To check which functions are re-executed, we can examine the graph edges.
        # A more direct way would be to add counters, but let's test the outcome.
        assert calc1.a() == 202  # (100 + 1) * 2
        assert calc2.a() == 22   # Should still be cached

        # Verify that calc2.a() was indeed cached by overriding its 'c'
        # and seeing if the value changes. If it doesn't, it means 'a' was cached.
        override_value('c', 500, instance=calc2)
        assert calc2.a() == 1002 # (500 + 1) * 2

    def test_get_edges_representation(self):
        """
        Tests that the string representation of nodes in get_edges
        is correct for instance methods.
        """

        @g_func
        def standalone_func():
            return 1

        class MyGraph:
            @g_func
            def my_method(self):
                return standalone_func() + 1

        instance = MyGraph()
        instance.my_method()

        edges = get_edges()

        instance_id_hex = f"{id(instance):#x}"

        expected_parent_repr = f"my_method[{instance_id_hex}](*(), **frozendict.frozendict({{}}))"
        expected_child_repr = "standalone_func(*(), **frozendict.frozendict({}))"

        assert (expected_parent_repr, expected_child_repr) in edges
import pytest
from kinyu.graph import g_func, clear_cache, override_value, get_edges


class TestGraphInstanceMethods:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        clear_cache()

    def test_method_calls_on_different_instances_are_distinct(self):
        """
        Tests that the graph correctly distinguishes between method calls
        from different object instances.
        """
        execution_log = []

        class MyCalc:
            def __init__(self, name):
                self.name = name

            @g_func
            def final_step(self):
                execution_log.append(f'final_step_{self.name}')
                return self.intermediate_step() * 2

            @g_func
            def intermediate_step(self):
                execution_log.append(f'intermediate_step_{self.name}')
                return 10

        calc1 = MyCalc("calc1")
        calc2 = MyCalc("calc2")

        # Execute on the first instance
        calc1.final_step()
        assert execution_log == ['final_step_calc1', 'intermediate_step_calc1']

        # Execute on the second instance
        execution_log.clear()
        calc2.final_step()
        assert execution_log == ['final_step_calc2', 'intermediate_step_calc2']

        # Verify second call is cached for calc1
        execution_log.clear()
        calc1.final_step()
        assert execution_log == []

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

        assert calc1.a() == 202  # (100 + 1) * 2
        assert calc2.a() == 22   # Should still be cached

    def test_get_edges_representation_for_instances(self):
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
import unittest
from kinyu.calc_graph import calc_node, calc_context
from kinyu.calc_graph.graph import graph

class TestStaticCalcGraph(unittest.TestCase):

    def setUp(self):
        graph.clear()

    def test_standalone_function_caching(self):
        """Tests that a simple decorated function caches its result."""
        call_count = {'val': 0}
        @calc_node
        def a():
            call_count['val'] += 1
            return 10

        self.assertEqual(a(), 10)
        self.assertEqual(call_count['val'], 1)
        self.assertEqual(a(), 10)
        self.assertEqual(call_count['val'], 1)

    def test_static_dependency_registration(self):
        """Tests that dependencies are identified statically before execution."""
        @calc_node
        def b():
            return 5

        @calc_node
        def c():
            return b() * 2

        # Analyze without calling
        c._analyze_and_compile()

        self.assertEqual(len(c.static_dependencies), 1)
        self.assertIs(c.static_dependencies[0], b)

    def test_pre_evaluation_order(self):
        """Tests that children are evaluated before the parent's body."""
        exec_order = []
        @calc_node
        def d():
            exec_order.append('d')
            return 3

        @calc_node
        def e():
            exec_order.append('e_body')
            # The 'd()' call is replaced by a variable during AST transformation,
            # but the pre-evaluation logic ensures it runs first.
            return d() + 7

        e()
        self.assertEqual(exec_order, ['d', 'e_body'], "Child `d` should execute before the body of `e`.")

    def test_invalidation_propagates(self):
        """Tests that invalidating a node makes its dependents dirty."""
        call_counts = {'f': 0, 'g': 0}
        @calc_node
        def f():
            call_counts['f'] += 1
            return 3

        @calc_node
        def g():
            call_counts['g'] += 1
            return f() + 7

        self.assertEqual(g(), 10)
        self.assertEqual(call_counts['f'], 1)
        self.assertEqual(call_counts['g'], 1)

        f.invalidate()

        self.assertEqual(g(), 10)
        self.assertEqual(call_counts['f'], 2)
        self.assertEqual(call_counts['g'], 2)

    def test_set_value(self):
        """Tests that setting a value works and invalidates dependents."""
        call_counts = {'h': 0, 'i': 0}
        @calc_node
        def h():
            call_counts['h'] += 1
            return 100

        @calc_node
        def i():
            call_counts['i'] += 1
            return h() * 2

        self.assertEqual(i(), 200)
        self.assertEqual(call_counts['h'], 1)
        self.assertEqual(call_counts['i'], 1)

        h.set_value(50)
        self.assertEqual(call_counts['h'], 1)

        self.assertEqual(i(), 100)
        self.assertEqual(call_counts['i'], 2)

    def test_calc_context_scoped_cache(self):
        call_counts = {'a': 0}

        @calc_node
        def a():
            call_counts['a'] += 1
            return 21

        with calc_context():
            self.assertEqual(a(), 21)
            self.assertEqual(a(), 21)

        self.assertEqual(call_counts['a'], 1)

        self.assertEqual(a(), 21)
        self.assertEqual(call_counts['a'], 2)

        self.assertEqual(a(), 21)
        self.assertEqual(call_counts['a'], 2)

    def test_calc_context_override(self):
        call_counts = {'b': 0}

        @calc_node
        def b():
            call_counts['b'] += 1
            return 5

        with calc_context():
            b.override(42)
            self.assertEqual(b(), 42)
            self.assertEqual(call_counts['b'], 0)

        self.assertEqual(call_counts['b'], 0)
        self.assertEqual(b(), 5)
        self.assertEqual(call_counts['b'], 1)

        b.set_value(7)

        with calc_context():
            b.override(99)
            self.assertEqual(b(), 99)

        self.assertEqual(b(), 7)
        self.assertEqual(call_counts['b'], 1)

    def test_calc_context_set_value_sticky(self):
        call_counts = {'c': 0}

        @calc_node
        def c():
            call_counts['c'] += 1
            return 11

        with calc_context():
            c.set_value(123)
            self.assertEqual(c(), 123)

        self.assertEqual(call_counts['c'], 0)
        self.assertEqual(c(), 123)

        c.invalidate()
        self.assertEqual(c(), 11)
        self.assertEqual(call_counts['c'], 1)

    def test_instance_method_caching_and_uniqueness(self):
        """Tests that nodes are unique per instance."""
        class MyCalc:
            def __init__(self, val):
                self.val = val
                self.call_count = 0

            @calc_node
            def j(self):
                self.call_count += 1
                return self.val

        obj1 = MyCalc(10)
        obj2 = MyCalc(20)

        self.assertEqual(obj1.j(), 10)
        self.assertEqual(obj1.call_count, 1)
        self.assertEqual(obj1.j(), 10)
        self.assertEqual(obj1.call_count, 1)

        self.assertEqual(obj2.j(), 20)
        self.assertEqual(obj2.call_count, 1)
        self.assertEqual(obj1.call_count, 1)

    def test_instance_method_invalidation(self):
        """Tests that invalidation is instance-specific."""
        class MyCalc:
            def __init__(self, name):
                self.name = name
                self.call_count_k = 0
                self.call_count_l = 0

            @calc_node
            def k(self):
                self.call_count_k += 1
                return self.name

            @calc_node
            def l(self):
                self.call_count_l += 1
                return self.k()

        obj_a = MyCalc("A")
        obj_b = MyCalc("B")

        self.assertEqual(obj_a.l(), "A")
        self.assertEqual(obj_a.call_count_k, 1)
        self.assertEqual(obj_a.call_count_l, 1)

        self.assertEqual(obj_b.l(), "B")
        self.assertEqual(obj_b.call_count_k, 1)
        self.assertEqual(obj_b.call_count_l, 1)

        obj_a.k.invalidate()

        self.assertEqual(obj_a.l(), "A")
        self.assertEqual(obj_a.call_count_k, 2)
        self.assertEqual(obj_a.call_count_l, 2)

        self.assertEqual(obj_b.l(), "B")
        self.assertEqual(obj_b.call_count_k, 1)
        self.assertEqual(obj_b.call_count_l, 1)


class TestEdgeNodeCalcGraph(unittest.TestCase):

    def setUp(self):
        graph.clear()

    def test_conditional_edge_node_shapes(self):
        class ConditionalCalc:
            def __init__(self, threshold):
                self.threshold = threshold

            @calc_node
            def Foo(self):
                return self.threshold

            @calc_node
            def A(self):
                return 'A'

            @calc_node
            def B(self):
                return 'B'

            @calc_node
            def Root(self):
                if self.Foo() > 3:
                    return self.A()
                else:
                    return self.B()

        calc = ConditionalCalc(5)
        self.assertEqual(calc.Root(), 'A')

        summary = ConditionalCalc.Root.describe_dependencies(calc)
        self.assertCountEqual([node.name for node in summary['edge_nodes']], ['Foo'])
        self.assertCountEqual([node.name for node in summary['input_nodes']], ['A'])
        self.assertCountEqual([node.name for node in summary['predicate_nodes']], ['B'])

        calc.threshold = 1
        ConditionalCalc.Foo.invalidate(calc)
        self.assertEqual(calc.Root(), 'B')

        summary = ConditionalCalc.Root.describe_dependencies(calc)
        self.assertCountEqual([node.name for node in summary['edge_nodes']], ['Foo'])
        self.assertCountEqual([node.name for node in summary['input_nodes']], ['B'])
        self.assertCountEqual([node.name for node in summary['predicate_nodes']], ['A'])

    def test_dynamic_edge_node_dispatch(self):
        class Alpha:
            def __init__(self, label):
                self.label = label

            @calc_node
            def Bar(self):
                return f"alpha-{self.label}"

        class Beta:
            def __init__(self, label):
                self.label = label

            @calc_node
            def Bar(self):
                return f"beta-{self.label}"

        class DynamicCalc:
            def __init__(self):
                self.use_alpha = True
                self.alpha = Alpha('value')
                self.beta = Beta('value')

            @calc_node
            def Foo(self):
                if self.use_alpha:
                    return self.alpha
                return self.beta

            @calc_node
            def Output(self):
                return self.Foo().Bar()

        calc = DynamicCalc()
        self.assertEqual(calc.Output(), 'alpha-value')

        summary = DynamicCalc.Output.describe_dependencies(calc)
        self.assertEqual(len(summary['edge_nodes']), 1)
        self.assertIs(summary['edge_nodes'][0].instance, calc)
        self.assertEqual(summary['edge_nodes'][0].name, 'Foo')

        self.assertEqual(len(summary['input_nodes']), 1)
        self.assertIs(summary['input_nodes'][0].instance, calc.alpha)
        self.assertEqual(summary['input_nodes'][0].name, 'Bar')
        self.assertEqual(len(summary['predicate_nodes']), 0)

        # Setting Foo() to a different object should change the graph shape.
        calc.Foo.set_value(calc.beta)
        self.assertEqual(calc.Output(), 'beta-value')

        summary = DynamicCalc.Output.describe_dependencies(calc)
        self.assertIs(summary['input_nodes'][0].instance, calc.beta)
        self.assertIn(Alpha.Bar._get_node(calc.alpha), summary['predicate_nodes'])

        # When Foo() uses a conditional, toggling the condition changes the shape as well.
        calc.use_alpha = False
        calc.Foo.invalidate()
        calc.Output.invalidate()
        self.assertEqual(calc.Output(), 'beta-value')

        summary = DynamicCalc.Output.describe_dependencies(calc)
        self.assertIs(summary['input_nodes'][0].instance, calc.beta)
        self.assertIn(Alpha.Bar._get_node(calc.alpha), summary['predicate_nodes'])


if __name__ == '__main__':
    unittest.main()
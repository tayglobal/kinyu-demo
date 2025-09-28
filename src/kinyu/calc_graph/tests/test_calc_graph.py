import unittest
from kinyu.calc_graph import calc_node
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

if __name__ == '__main__':
    unittest.main()
import pytest
import time
from kinyu.graph import g_func, clear_cache, get_edges, override_value

# --- Test Setup ---

@pytest.fixture(autouse=True)
def run_around_tests():
    # This runs before and after each test function
    clear_cache()
    yield
    clear_cache()

# --- Test Cases ---

def test_caching_and_expensive_function():
    """Tests that results are cached and expensive functions are not re-run."""
    expensive_call_count = 0

    @g_func
    def expensive_function(x):
        nonlocal expensive_call_count
        expensive_call_count += 1
        time.sleep(0.1)
        return x

    @g_func
    def a():
        return b() + 1

    @g_func
    def b():
        return expensive_function(10)

    start_time = time.time()
    assert a() == 11
    assert time.time() - start_time > 0.1
    assert expensive_call_count == 1

    start_time = time.time()
    assert a() == 11
    assert time.time() - start_time < 0.01
    assert expensive_call_count == 1

def test_dependency_tracking_and_overriding():
    """Tests that overriding a node invalidates its ancestors."""
    call_log = []

    @g_func
    def a():
        call_log.append('a')
        return b() * 2

    @g_func
    def b():
        call_log.append('b')
        return c() + 1

    @g_func
    def c():
        call_log.append('c')
        return 10

    assert a() == 22
    assert call_log == ['a', 'b', 'c'], "Execution order follows Python's standard call stack"

    # Test caching
    call_log.clear()
    assert a() == 22
    assert call_log == [], "All nodes should be cached"

    # Test override
    override_value('c', 100)
    call_log.clear()
    assert a() == 202
    # 'c' is overridden, so it's not called. 'a' and 'b' are re-executed.
    assert call_log == ['a', 'b'], "Ancestors of overridden node are re-executed"

def test_conditional_dependency():
    """Tests that dependencies are tracked correctly based on runtime conditions."""
    @g_func
    def d(condition):
        if condition:
            return e()
        else:
            return f()

    @g_func
    def e():
        return "e"

    @g_func
    def f():
        return "f"

    # First run with condition=True
    assert d(True) == "e"
    edges = get_edges()
    parent_d_true = "d(*(True,), **frozendict.frozendict({}))"
    child_e = "e(*(), **frozendict.frozendict({}))"
    child_f = "f(*(), **frozendict.frozendict({}))"
    assert (parent_d_true, child_e) in edges
    assert (parent_d_true, child_f) not in edges

    # Second run with condition=False
    assert d(False) == "f"
    edges = get_edges()
    parent_d_false = "d(*(False,), **frozendict.frozendict({}))"
    # The dependency on e() from the previous call should still exist in the graph
    assert (parent_d_false, child_f) in edges
    assert (parent_d_true, child_e) in edges


def test_argument_sensitivity():
    """Tests that calls with different arguments are treated as distinct nodes."""
    @g_func
    def h():
        return b(x=1) + b(x=2)

    @g_func
    def b(x):
        return x * x

    assert h() == 5
    edges = get_edges()
    parent_h = "h(*(), **frozendict.frozendict({}))"
    child_b1 = "b(*(), **frozendict.frozendict({'x': 1}))"
    child_b2 = "b(*(), **frozendict.frozendict({'x': 2}))"
    assert (parent_h, child_b1) in edges
    assert (parent_h, child_b2) in edges

def test_clear_cache():
    """Tests that the cache can be cleared."""
    call_count = 0

    @g_func
    def expensive_node():
        nonlocal call_count
        call_count += 1
        return 1

    expensive_node()
    assert call_count == 1
    expensive_node()
    assert call_count == 1

    clear_cache()
    expensive_node()
    assert call_count == 2
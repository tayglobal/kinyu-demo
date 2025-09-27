# kinyu.graph: A High-Performance Dependency Graph

This module provides a decorator, `@g_func`, that turns Python functions into nodes in a cacheable, overridable dependency graph. The core logic is implemented in Rust for high performance and thread safety.

## The Concept

In many computational workflows, especially in finance, calculations are often chained together. For example, the result of function `c()` is used by `b()`, and the result of `b()` is used by `a()`. This forms a dependency graph: `a -> b -> c`.

The `@g_func` decorator builds this graph automatically. When you call a decorated function, it does two things:
1.  **Caches the Result**: The return value of the function is stored. On subsequent calls with the same arguments, the cached result is returned instantly, avoiding re-computation. This is invaluable for expensive operations.
2.  **Tracks Dependencies**: The decorator records the relationships between functions. If you later override the value of a node (e.g., `c`), the graph automatically knows that its ancestors (`a` and `b`) are now stale and must be re-executed.

This provides a powerful and efficient way to manage complex calculations, perform "what-if" analysis by overriding base values, and visualize the flow of data.

## Installation

The project uses a standard Python build system with a Rust extension. To install the `kinyu.graph` module and its dependencies, follow these steps from the root of the repository:

1.  **Install build dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install the project in editable mode:**
    This command will compile the Rust extension and make the `kinyu` package available in your environment.
    ```bash
    pip install -e .
    ```

## Demonstration

The following script demonstrates the key features of the dependency graph: caching, dependency tracking, and value overriding.

### Demo Code (`demo.py`)

```python
import time
import pprint
from kinyu.graph import g_func, get_edges, override_value, clear_cache

def main():
    """A demonstration of the dependency graph functionality."""

    print("--- DEMONSTRATION: Caching and Dependency Tracking ---\n")

    # Define a few functions that form a dependency graph.
    # a() -> b() -> c()
    # c() is an "expensive" function that takes time to execute.

    @g_func
    def a():
        print("Executing a()...")
        return b() * 2

    @g_func
    def b():
        print("Executing b()...")
        return c() + 10

    @g_func
    def c():
        print("Executing c() (expensive operation)...")
        time.sleep(0.5)
        return 5

    # --- First execution ---
    print("1. First execution of a():")
    start_time = time.time()
    result = a()
    duration = time.time() - start_time
    print(f"   Result: {result}")
    print(f"   Duration: {duration:.2f}s\n")
    # Expected output: a(), b(), and c() are all executed. Duration > 0.5s.

    # --- Second execution (from cache) ---
    print("2. Second execution of a() (should be cached):")
    start_time = time.time()
    result = a()
    duration = time.time() - start_time
    print(f"   Result: {result}")
    print(f"   Duration: {duration:.2f}s\n")
    # Expected output: No functions are executed. Duration should be near zero.

    # --- Show the dependency graph ---
    print("3. Current dependency graph edges:")
    pprint.pprint(get_edges())
    print("")

    # --- Override a value ---
    print("--- DEMONSTRATION: Value Overriding ---\n")
    print("4. Overriding the value of c() to 100.")
    override_value('c', 100)
    print("   The cache for ancestors of c() (i.e., a and b) should be invalidated.\n")

    # --- Execution after override ---
    print("5. Executing a() after overriding c():")
    result = a()
    print(f"   New Result: {result}")
    print("   Note that only a() and b() were re-executed.\n")
    # Expected output: a() and b() are executed, but c() is not.

    # --- Final graph ---
    print("6. Final dependency graph edges:")
    pprint.pprint(get_edges())

if __name__ == "__main__":
    clear_cache()
    main()
```

### Demo Output

Running the script produces the following output, clearly showing the caching and invalidation logic at work.

```
--- DEMONSTRATION: Caching and Dependency Tracking ---

1. First execution of a():
Executing a()...
Executing b()...
Executing c() (expensive operation)...
   Result: 30
   Duration: 0.52s

2. Second execution of a() (should be cached):
   Result: 30
   Duration: 0.00s

3. Current dependency graph edges:
[('a(*(), **frozendict.frozendict({}))', 'b(*(), **frozendict.frozendict({}))'),
 ('b(*(), **frozendict.frozendict({}))', 'c(*(), **frozendict.frozendict({}))')]

--- DEMONSTRATION: Value Overriding ---

4. Overriding the value of c() to 100.
   The cache for ancestors of c() (i.e., a and b) should be invalidated.

5. Executing a() after overriding c():
Executing a()...
Executing b()...
   New Result: 220
   Note that only a() and b() were re-executed.

6. Final dependency graph edges:
[('a(*(), **frozendict.frozendict({}))', 'b(*(), **frozendict.frozendict({}))'),
 ('b(*(), **frozendict.frozendict({}))', 'c(*(), **frozendict.frozendict({}))')]
```
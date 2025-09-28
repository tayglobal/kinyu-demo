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

## Interactive Demonstration

For a detailed walkthrough of the concepts and features, please see the demonstration notebook:

[**notebooks/graph_demonstration.ipynb**](../../../../notebooks/graph_demonstration.ipynb)

This notebook covers:
-   **Caching and Performance**: See how `@g_func` avoids re-running expensive functions.
-   **Dependency Tracking**: Understand how the graph is built and how `override_value` intelligently invalidates the cache.
-   **Use Cases**: Explore examples with both standalone functions and instance methods to see how the decorator works in different scenarios.
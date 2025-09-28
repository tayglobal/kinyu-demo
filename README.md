# Kinyu Demos

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tayglobal/kinyu-demo)

This is a repository that contains interesting demos that can be applied in the financial industry. 

See
[Documentation](https://kydb.readthedocs.io)
to find out more.

## Demos

[Dependency Graph](https://tayglobal.github.io/kinyu-demo/html/examples/Dependency_Graph.html)

Demonstrate the dependency graph heavily used in Goldman Sache (SecDB), JPMorgan (Athena), Bank of America (Quartz) and Beacon.io.

[IR Risk in Multicurve Env](https://tayglobal.github.io/kinyu-demo/html/examples/IR_Risk_in_Multicurve_Env.html)

For a single swap and a portfolio, show how we could display a risk ladder. The tenors would be based on market data used to build the curves. Two curves are involved in this demo, a projection curve and a funding curve.

[Remote Import](https://tayglobal.github.io/kinyu-demo/html/examples/Remote_Import.html)

Python is an extremely dynamic and flexible language. In this demo, the goal is to show how we can use python’s path_hooks mechanism to change imports from using local filesystem to a remote source.

As a bonus, this article also demonstrates creating necessary AWS resources via Cloudformation and how to remove them with a single command. There will be some basic performance analysis of the each remote source.

## Core Libraries

This repository is structured as a collection of specialized libraries, each targeting a specific domain within financial engineering. Below is an overview of the core components:

- **[`calc_graph`](src/kinyu/calc_graph)**: Implements a powerful calculation dependency graph in Python. It analyzes function Abstract Syntax Trees (ASTs) to automatically determine execution order, manage caching, and optimize re-computation, similar to systems like SecDB or Athena.

- **[`corr`](src/kinyu/corr)**: A high-performance Rust library for correlation calculations. It is designed to be used in various financial models where the correlation between different assets or risk factors is required.

- **[`credit`](src/kinyu/credit)**: Provides Python tools for credit analysis. A key feature is the ability to build survival curves from Credit Default Swap (CDS) spreads, a fundamental task in credit risk modeling.

- **[`fixture`](src/kinyu/fixture)**: Contains Python scripts for setting up test data and environments. This is essential for ensuring the reliability and accuracy of the financial models, with examples like setting up data for interest rate swaps.

- **[`rates`](src/kinyu/rates)**: A Rust library focused on building and managing interest rate curves. It provides the core components for pricing and risk management of interest rate-sensitive instruments.

- **[`rimport`](src/kinyu/rimport)**: A non-standard Python module importer that can load code from a remote `kydb` database. This allows for dynamic code management and deployment, decoupling the application from the local filesystem.

- **[`vol`](src/kinyu/vol)**: A comprehensive suite for volatility modeling, including:
  - **[`historical`](src/kinyu/vol/historical)**: A Rust library for calculating historical volatility from time series data.
  - **[`implied`](src/kinyu/vol/implied)**: A Rust library for calibrating implied volatility surfaces from option market prices.
  - **[`nn`](src/kinyu/vol/nn)**: A PyTorch-based library for creating smooth, arbitrage-free volatility surfaces using neural networks.

- **[`warrants`](src/kinyu/warrants)**: A sophisticated Rust library for pricing exotic, callable warrants with path-dependent features. It can be compiled to WebAssembly (WASM) to run directly in the browser.
  - **Live Demo**: [Explore the interactive warrant pricing tool here](https://tayglobal.github.io/kinyu-demo/demo/warrant/).

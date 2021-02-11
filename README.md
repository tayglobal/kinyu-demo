# Kinyu Demos

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


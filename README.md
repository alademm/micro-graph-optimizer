# Micro Graph Optimizer
Micro graph optimizer is a minimal least squares solver for problems represented as factor graphs.

The solver is contained in `micro_graph_optimizer.h` and `micro_graph_optimizer.cpp` and depends only on Eigen. It has no special build requirements. That said, a CMake build file is provided for convenience and to quickly build the examples.

Two examples are provided:
1. hello_slam: constructs a simple linear pose graph and optimizes it.
2. slam2d: demonstrates optimzing a non-linear 2D SLAM problem read from a g2o file.

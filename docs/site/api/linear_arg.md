# Core data structures

This page describes the primary graph objects and constructors behind `linear-dag` genotype operators.

`LinearARG` is the core sample-by-variant `LinearOperator` abstraction. It stores sparse graph state and metadata
needed for filtering, transpose-aware algebra, and HDF5 persistence.

For lower-level graph construction, `BrickGraph` defines the inference path from phased
genotype matrices into linearized graph form.

::: linear_dag.LinearARG
    options:
        show_bases: true

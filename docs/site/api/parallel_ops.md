# Parallel operators

`linear-dag` includes process-parallel operators for blockwise genotype and GRM algebra on HDF5-backed data.

At a high level, these operators:

1. Load block metadata and optional variant filters from HDF5.
2. Spawn worker processes backed by shared memory arrays.
3. Execute `matmat` and `rmatmat` operations blockwise without materializing dense genotype matrices.

!!! note

    Use these operators as context managers so workers and shared memory are
    cleaned up deterministically.

::: linear_dag.ParallelOperator
    options:
        show_bases: true
        members:
            - from_hdf5

---

::: linear_dag.GRMOperator
    options:
        show_bases: true
        members:
            - from_hdf5

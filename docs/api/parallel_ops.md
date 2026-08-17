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

## JAX exact-ragged fallback

[`linear_dag.core.jaxlinarg.JaxParallelOperator`][] is the public JAX
compatibility path for multi-block data. It keeps exact block shapes, places
blocks on their assigned devices, and runs cached per-device-range programs. It
doesn't create worker processes or shared-memory segments, so it isn't a context
manager.

The packed representation remains an internal candidate during promotion
testing. Select `JaxParallelOperator` explicitly when whole-block packing exceeds
its configured padding limit; a failed packed construction doesn't switch to the
exact-ragged path automatically. The [JAX operator guide](jax.md) documents the
compiled-call, backend, ingress, and fallback contracts.

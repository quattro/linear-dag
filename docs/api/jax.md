# JAX operators

The public JAX API provides [`linear_dag.core.jaxlinarg.JaxLinearARG`][] for one
exact-shape LinearARG block,
[`linear_dag.core.jaxlinarg.JaxParallelOperator`][] for an exact-ragged set of
blocks, and [`linear_dag.core.jaxlinarg.JaxGRMOperator`][] for
relatedness-matrix products.
"Exact-ragged" means that each source block keeps its natural array shapes and
runs through a cached program assigned to one device.

Cross-platform promotion review recorded `continue_coexistence`. An internal
packed candidate concatenates ragged source blocks into fixed-shape device
shards, but it remains private and experimental because every collected warm
product ratio exceeded the promotion threshold and x86_64 CPU and GPU evidence
is missing. It isn't a public class or import path. Renaming or exporting it
requires a separate approved promotion plan after all gates pass.

## Current public exact-ragged API

Load one block with `JaxLinearARG`, or use `JaxParallelOperator` for a multi-block
HDF5 file:

```python
import jax
import jax.numpy as jnp
import numpy as np

from jax.sharding import Mesh
from linear_dag import Backend, JaxParallelOperator

mesh = Mesh(np.asarray(jax.devices()), ("blocks",))
operator = JaxParallelOperator.from_hdf5(
    "lineararg.h5",
    mesh=mesh,
    backend=Backend.AUTO,
)

variant_weights = jnp.ones((operator.shape[1], 4), dtype=jnp.float32)
sample_scores = operator.matmat(variant_weights)
```

`JaxParallelOperator` places each exact block on its assigned device. Its bound
`matmat` and `rmatmat` methods coordinate cached exact-shape programs from
Python. Call those methods directly. Wrapping a bound multi-block method in an
additional `jax.jit` captures its block arrays as constants and bypasses the
placement boundary. The same restriction applies to a `JaxGRMOperator` backed
by `JaxParallelOperator`.

## Backends

[`linear_dag.core.jaxlinarg.Backend`][] exposes the backends implemented in
this branch:

- `Backend.PURE_JAX` uses portable pure JAX and is always available.
- `Backend.FFI_CPU` uses optional native CPU FFI targets. An explicit request
  fails during construction if the active platform isn't CPU or the complete
  target set required by the representation is unavailable.
- `Backend.AUTO` selects CPU FFI only when the representation's complete target
  set is available on CPU. Otherwise it silently uses pure JAX.

There is no Pallas backend in this branch. GPU uses the portable pure-JAX path;
this branch doesn't advertise an accelerator-specific kernel.

## Packed candidate compilation contract

The intended packed functional contract passes the operator as an explicit
argument:

```text
lineararg_matmat(operator, values) -> sample_values
lineararg_rmatmat(operator, values) -> variant_values
loss(parameters, operator, phenotype) -> scalar
```

This form keeps the packed graph arrays as dynamic operands when `loss` is
transformed. The candidate supports `jit`, JVP, VJP, `grad`, `value_and_grad`,
higher-order differentiation, `vmap`, `scan`, and `remat` for dense operands and
surrounding learnable parameters. Graph state is opaque and non-learnable:
topology, edge data, allele metadata, and packing decisions don't receive
tangents or cotangents.

Bound `matmat` and `rmatmat` methods remain available for eager calls. The
candidate's `compile_matmat()` and `compile_rmatmat()` helpers retain bound-call
convenience while supplying the operator to a module-level JIT as a dynamic
argument; raw bound-method closure capture is outside the graph-memory
guarantee. Use the explicit-operator form or a safe compilation helper instead;
the implementation doesn't promise to detect unsafe closures by inspecting
JAX tracers.

!!! note

    The functional operations and safe helpers in this section describe the
    internal candidate's promotion contract. They aren't additions to the
    current public exact `JaxLinearARG` API.

## Ingress and durable storage

HDF5 remains the durable source format. The internal candidate streams existing
root-level or multi-block HDF5 data into final assigned shards without changing
the HDF5 schema. In-memory construction uses the same packing boundary. This
branch does not define a packed serialization format: there is no packed
`write`, pickle, or Equinox-serialization contract. Reconstruct a candidate from
the durable source.

Real Zarr support remains on the downstream `genoio` integration path. Before a
later merge can claim Zarr support, the real reader must pass reconstruction,
peak residency, transform, and schema-parity gates. Generic group fixtures are
not durable Zarr integration coverage; they test only the host-array adapter
shape.

## Padding diagnostics and fallback

Packing assigns whole source blocks to device shards. It doesn't subdivide an
oversized source graph. The default `max_padding_ratio=1.25` compares padded
canonical graph bytes with unpadded canonical graph bytes. Construction reports
the measured ratio and configured limit, then rejects a plan above the limit.
A caller testing a known skewed input may supply a larger value or `None`, but
the override doesn't bypass descriptor, shape, or indexing validation.

A failed packed construction does not automatically fall back to the
exact-ragged implementation. Choose `JaxParallelOperator` explicitly when the
packing diagnostics show that exact-ragged execution is the appropriate
fallback.

## Reproduce promotion evidence

Use the portable runner from a clean checkout with an explicit platform label,
device count, representative HDF5 input, and output directory:

```console
mkdir -p /tmp/linear-dag-jax-promotion
scripts/run_jax_promotion.sh \
  --repo-root "$PWD" \
  --hdf5-path "$PWD/1kg_chromosomes_n3202_blocks.h5" \
  --output-dir /tmp/linear-dag-jax-promotion \
  --platform-label forced-two-device-cpu \
  --device-count 2
```

The runner builds the CPU FFI targets when collecting CPU evidence, runs the
correctness and transform suites, then writes separate fresh-cache and
reused-cache JSON evidence. The committed decision at
`.plans/implementation-plans/2026-08-13-jax-packed-sharded-lineararg/promotion-decision.md`
records the current blockers. Planning artifacts aren't part of the MkDocs
site.

## API reference

::: linear_dag.core.jaxlinarg.Backend

---

::: linear_dag.core.jaxlinarg.JaxLinearARG
    options:
        members:
            - from_lineararg
            - from_hdf5_block
            - matmat
            - rmatmat

---

::: linear_dag.core.jaxlinarg.JaxParallelOperator
    options:
        members:
            - from_linearargs
            - from_hdf5
            - matmat
            - rmatmat

---

::: linear_dag.core.jaxlinarg.JaxGRMOperator
    options:
        members:
            - matmat
            - rmatmat
            - matmat_blockwise

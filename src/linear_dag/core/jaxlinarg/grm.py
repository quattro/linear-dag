# pattern: Functional Core

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import cached_property
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jax.sharding import PartitionSpec as P
from jaxtyping import Array, ArrayLike

from .operator import Backend, JaxLinearARG
from .wrapper import (
    _as_rank2_matrix,
    _device_put_if_needed,
    _mesh_assembly_device,
    _mesh_block_devices,
    _shard_map,
    JaxParallelOperator,
)


class JaxGRMOperator(eqx.Module):
    r"""JAX-compatible genetic relatedness matrix operator.

    This operator is an algebraic view over an already-constructed JAX genotype
    operator $X$. It computes sample-space products of the form
    $(X - \mathbf{1} f^\top) D (X - \mathbf{1} f^\top)^\top y$, where $f$ is
    the vector of allele frequencies. By default $D_j = (f_j(1-f_j))^\alpha$,
    with zero weight for monomorphic variants. A supplied `diagonal` replaces
    these weights and may depend on differentiable model parameters. Signed
    weights are accepted; positive semidefiniteness then depends on the caller.

    !!! info

        Construct and validate operators on the host before applying JAX
        transformations. Use `matmat(..., diagonal=weights)` to vary weights
        inside a traced model. This class does not load or filter graph data.
        Sample rows follow the wrapped operator (usually haplotypes); diploid
        pairing and phenotype alignment are separate operations.

    !!! Example
        ```python
        import jax
        from linear_dag.core.jaxlinarg import JaxGRMOperator

        grm = JaxGRMOperator(genotypes, center=True)
        product = jax.jit(lambda weights, rhs: grm.matmat(rhs, diagonal=weights))
        ```

    **Arguments:**

    - `operator`: JAX genotype operator providing `matmat`, `rmatmat`, and
      per-variant allele counts.
    - `diagonal`: Optional dynamic variant weights, differentiable on the native
      CPU path along with the input vectors. Graph topology stays fixed.
    - `alpha`: Static allele-frequency weight exponent, used when no diagonal is supplied.
    - `center`: Whether to center genotypes by allele frequency before forming
      the GRM.
    - `iids`: Optional host-side sample identifiers for phenotype alignment.
    """

    operator: JaxLinearARG | JaxParallelOperator
    alpha: float = eqx.field(default=-1.0, converter=float, static=True)
    center: bool = eqx.field(default=True, converter=bool, static=True)
    iids: Any = eqx.field(default=None, static=True)
    diagonal: Array | None = None

    def __check_init__(self) -> None:
        if not isinstance(self.operator, (JaxLinearARG, JaxParallelOperator)):
            raise TypeError("operator must be a JaxLinearARG or JaxParallelOperator")
        n_samples = self.operator.shape[0]
        if n_samples <= 0:
            raise ValueError("JaxGRMOperator requires a positive sample count")
        if self.diagonal is not None and jnp.shape(self.diagonal) != (self.operator.shape[1],):
            raise ValueError("diagonal length must match the variant count")
        if self.center or self.diagonal is None:
            counts = np.asarray(self.operator.allele_counts)
            if counts.size and (counts.min() < 0 or counts.max() > n_samples):
                raise ValueError("JaxGRMOperator requires allele_counts in [0, n_samples] for every variant")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the sample-space GRM shape `(n_samples, n_samples)`."""
        return (self.operator.shape[0], self.operator.shape[0])

    @property
    def dtype(self) -> object:
        """Return the wrapped operator dtype."""
        return self.operator.dtype

    def matmat(self, x: ArrayLike, *, diagonal: ArrayLike | None = None) -> Array:
        """Multiply by the implicit genetic relatedness matrix.

        **Arguments:**

        - `x`: Rank-1 sample-space vector, or rank-2 sample-by-trait matrix.
        - `diagonal`: Optional dynamic variant weights overriding the stored
          diagonal or frequency-based weights. Gradients flow through these values.

        **Returns:**

        - GRM product with the same rank convention as the input.
        """
        x, was_vector = _as_rank2_matrix(x, expected_rows=self.shape[0], dtype=self.dtype)

        selected_diagonal = self.diagonal if diagonal is None else diagonal
        native = self._native_state()
        if native is not None:
            from ._packed import _weighted

            graph, config = native
            if selected_diagonal is None:
                frequencies = self.operator.allele_counts.astype(x.dtype) / self.shape[0]
                pq = frequencies * (1 - frequencies)
                # Keep the unselected branch finite for negative exponents.
                # Monomorphic variants contribute zero, not 0**alpha.
                safe_pq = jnp.where(pq > 0, pq, 1)
                selected_diagonal = jnp.where(pq > 0, safe_pq**self.alpha, 0)
            result = _weighted(
                graph, jnp.asarray(selected_diagonal, dtype=x.dtype), x, replace(config, center=self.center)
            )
            return result[:, 0] if was_vector else result
        if selected_diagonal is not None:
            weights = jnp.asarray(selected_diagonal, dtype=x.dtype)
            if weights.shape != (self.operator.shape[1],):
                raise ValueError("diagonal length must match the variant count")
            scores = self.operator.rmatmat(x)
            frequencies = self.operator.allele_counts.astype(x.dtype) / self.shape[0]
            if self.center:
                scores = scores - frequencies[:, None] * jnp.sum(x, axis=0)
            scores = weights[:, None] * scores
            result = self.operator.matmat(scores)
            if self.center:
                result = result - jnp.sum(frequencies[:, None] * scores, axis=0)
            return result[:, 0] if was_vector else result

        @jax.custom_vjp
        def product(values: Array) -> Array:
            return self._matmat_rank2(values)

        def product_fwd(values: Array) -> tuple[Array, None]:
            return self._matmat_rank2(values), None

        def product_bwd(_residual: None, cotangent: Array) -> tuple[Array]:
            # The GRM is symmetric, so its input cotangent is the same GRM
            # product applied to the output cotangent. Operator fields,
            # including `alpha`, are treated as fixed configuration here.
            return (self._matmat_rank2(cotangent),)

        product.defvjp(product_fwd, product_bwd)
        result = product(x)
        return result[:, 0] if was_vector else result

    def _matmat_rank2(self, x: Array) -> Array:
        if isinstance(self.operator, JaxLinearARG):
            result = _block_grm_product(
                self.operator,
                x,
                alpha=self.alpha,
                center=self.center,
            )
        elif len(self.operator.block_ranges) > 1:
            result = self._sharded_matmat(x)
        else:
            result = jnp.zeros_like(x)
            for block in self.operator.blocks:
                result = result + _block_grm_product(
                    block,
                    x,
                    alpha=self.alpha,
                    center=self.center,
                )
        return result

    def _native_state(self):
        if isinstance(self.operator, JaxParallelOperator):
            if self.operator._graph is not None:
                return self.operator._graph, self.operator._config
        elif self.operator.backend is Backend.FFI_CPU and self.operator.data.dtype == jnp.int32:
            return self.operator._packed_state()
        return None

    def matmat_blockwise(self, x: ArrayLike) -> Array:
        """Apply the GRM with backend-specific block execution.

        Native CPU execution delegates to the fused packed product, whose
        scratch is already bounded. The pure-JAX default-weight path keeps
        a host loop over cached per-block JIT functions to limit simultaneous
        block temporaries. Explicit diagonal weights use `matmat` directly.

        **Arguments:**

        - `x`: Sample-space vector or sample-by-trait matrix.

        **Returns:**

        - GRM product with the same rank as the input.
        """
        if self._native_state() is not None or self.diagonal is not None:
            return self.matmat(x)
        x, was_vector = _as_rank2_matrix(x, expected_rows=self.shape[0], dtype=self.dtype)
        result = self._matmat_blockwise_rank2(x)
        return result[:, 0] if was_vector else result

    def _matmat_blockwise_rank2(self, x: Array) -> Array:
        assembly_device = (
            _mesh_assembly_device(self.operator.mesh) if isinstance(self.operator, JaxParallelOperator) else None
        )
        result = _device_put_if_needed(jnp.zeros_like(x), assembly_device)
        for device, product in self._blockwise_grm_products:
            contribution = product(_device_put_if_needed(x, device))
            result = result + _device_put_if_needed(contribution, assembly_device)
        return result

    @cached_property
    def _blockwise_grm_products(self) -> tuple[tuple[jax.Device | None, Callable[[Array], Array]], ...]:
        blocks = self.operator.blocks if isinstance(self.operator, JaxParallelOperator) else (self.operator,)
        devices = _grm_block_devices(self.operator) if isinstance(self.operator, JaxParallelOperator) else (None,)
        return tuple(
            (device, _blockwise_grm_product(block, alpha=self.alpha, center=self.center))
            for device, block in zip(devices, blocks, strict=True)
        )

    def matvec(self, x: ArrayLike) -> Array:
        """Multiply a vector by the implicit genetic relatedness matrix."""
        return self.matmat(x)

    def rmatmat(self, x: ArrayLike) -> Array:
        """Multiply by the GRM transpose.

        The GRM is symmetric by construction, so this dispatches to
        [`linear_dag.core.jaxlinarg.JaxGRMOperator.matmat`][].
        """
        return self.matmat(x)

    def rmatvec(self, x: ArrayLike) -> Array:
        """Multiply a vector by the GRM transpose."""
        return self.rmatmat(x)

    def __matmul__(self, x: ArrayLike) -> Array:
        return self.matmat(x)

    @property
    def T(self) -> "JaxGRMOperator":
        """Return this symmetric operator."""
        return self

    def _sharded_matmat(self, x: Array) -> Array:
        branches = tuple(self._grm_branch(start, end) for start, end in self.operator.block_ranges)

        def mapped(values: Array) -> Array:
            axis_index = jax.lax.axis_index("blocks")
            local = jax.lax.switch(axis_index, branches, values)
            # Every GRM block-range contribution is sample-shaped, so unlike
            # `JaxParallelOperator.rmatmat` there is no ragged variant output to
            # assemble. Devices can reduce partial sample products directly.
            return jax.lax.psum(local, "blocks")

        product = _shard_map(
            mapped,
            mesh=self.operator.mesh,
            in_specs=P(),
            out_specs=P(),
            axis_names={"blocks"},
        )
        return product(x)

    def _grm_branch(self, start: int, end: int) -> Callable[[Array], Array]:
        def branch(values: Array) -> Array:
            local = jnp.zeros_like(values)
            for block_index in range(start, end):
                local = local + _block_grm_product(
                    self.operator.blocks[block_index],
                    values,
                    alpha=self.alpha,
                    center=self.center,
                )
            return local

        return branch


def _grm_block_devices(operator: JaxParallelOperator) -> tuple[jax.Device | None, ...]:
    devices = _mesh_block_devices(operator.mesh)
    if devices is None:
        return tuple(None for _ in operator.blocks)

    block_devices: list[jax.Device | None] = [None] * len(operator.blocks)
    for range_index, (start, end) in enumerate(operator.block_ranges):
        for block_index in range(start, end):
            block_devices[block_index] = devices[range_index]
    return tuple(block_devices)


def _blockwise_grm_product(block: JaxLinearARG, *, alpha: float, center: bool) -> Callable[[Array], Array]:
    def product(values: Array) -> Array:
        return _block_grm_product_jit(block, values, alpha=alpha, center=center)

    return product


@eqx.filter_jit
def _block_grm_product_jit(
    block: JaxLinearARG,
    values: Array,
    *,
    alpha: float,
    center: bool,
) -> Array:
    return _block_grm_product(block, values, alpha=alpha, center=center)


def _block_grm_product(
    block: JaxLinearARG,
    values: Array,
    *,
    alpha: float,
    center: bool,
) -> Array:
    frequencies = jnp.asarray(block.allele_counts, dtype=values.dtype) / jnp.asarray(
        block.n_samples, dtype=values.dtype
    )
    pq = frequencies * (1 - frequencies)
    weights = jnp.where(pq > 0, pq**alpha, jnp.zeros_like(pq))

    variant_scores = block.rmatmat(values)
    if center:
        # Apply `(X - 1 f^T)^T values` without forming centered genotypes:
        # `X^T values - f (1^T values)`.
        value_sums = jnp.sum(values, axis=0, keepdims=True)
        weighted_scores = weights[:, None] * (variant_scores - frequencies[:, None] * value_sums)
        correction = jnp.sum(frequencies[:, None] * weighted_scores, axis=0, keepdims=True)
    else:
        correction = None
        weighted_scores = weights[:, None] * variant_scores

    result = block.matmat(weighted_scores)
    if center:
        # Apply the left centering factor:
        # `(X - 1 f^T) weighted_scores = X weighted_scores - 1 (f^T weighted_scores)`.
        result = result - correction
    return result

# pattern: Functional Core

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P
from jaxtyping import Array, ArrayLike

from .operator import JaxLinearARG
from .wrapper import (
    _as_rank2_matrix,
    _device_put_if_needed,
    _mesh_assembly_device,
    _mesh_block_devices,
    JaxParallelOperator,
)


class JaxGRMOperator(eqx.Module):
    r"""JAX-compatible genetic relatedness matrix operator.

    This operator is an algebraic view over an already-constructed JAX genotype
    operator $X$. It computes sample-space products of the form
    $(X - \mathbf{1} f^\top) D (X - \mathbf{1} f^\top)^\top y$, where $f$ is
    the vector of allele frequencies and $D_j = (f_j(1-f_j))^\alpha$.

    !!! info

        Loading, filtering, pruning, and device placement belong to
        [`linear_dag.core.jaxlinarg.JaxLinearARG`][] or
        [`linear_dag.core.jaxlinarg.JaxParallelOperator`][]. This class only
        defines the GRM algebra on top of those operators.

    **Arguments:**

    - `operator`: JAX genotype operator providing `matmat`, `rmatmat`, and
      per-variant allele counts.
    - `alpha`: Allele-frequency weight exponent.
    - `center`: Whether to center genotypes by allele frequency before forming
      the GRM.
    """

    operator: JaxLinearARG | JaxParallelOperator
    alpha: float = eqx.field(default=-1.0, converter=float, static=True)
    center: bool = eqx.field(default=True, converter=bool, static=True)
    iids: Any = eqx.field(default=None, static=True)

    def __check_init__(self) -> None:
        if not isinstance(self.operator, (JaxLinearARG, JaxParallelOperator)):
            raise TypeError("operator must be a JaxLinearARG or JaxParallelOperator")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the sample-space GRM shape `(n_samples, n_samples)`."""
        return (self.operator.shape[0], self.operator.shape[0])

    @property
    def dtype(self) -> object:
        """Return the wrapped operator dtype."""
        return self.operator.blocks[0].dtype if isinstance(self.operator, JaxParallelOperator) else self.operator.dtype

    def matmat(self, x: ArrayLike) -> Array:
        """Multiply by the implicit genetic relatedness matrix.

        **Arguments:**

        - `x`: Rank-1 sample-space vector, or rank-2 sample-by-trait matrix.

        **Returns:**

        - GRM product with the same rank convention as the input.
        """
        x, was_vector = _as_rank2_matrix(x, expected_rows=self.shape[0], dtype=self.dtype)

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

    def matmat_blockwise(self, x: ArrayLike) -> Array:
        """Multiply by the GRM using cached per-block JIT entrypoints.

        This path preserves the same algebra as
        [`linear_dag.core.jaxlinarg.JaxGRMOperator.matmat`][], but keeps the
        Python loop over LinearARG blocks outside a single XLA program. It is
        useful for workflows such as RHE where a full multi-block GRM product
        inside a larger `jax.jit` can make all block temporaries live in one HLO
        module.
        """
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

        product = jax.shard_map(
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

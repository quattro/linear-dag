# pattern: Mixed (unavoidable)
# Reason: GRM algebra is pure, while multi-block execution coordinates
# device-local operands and output assembly.

"""Implicit genetic relatedness matrix products over JAX LinearARG operators."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike

from .operator import _as_rank2_matrix, JaxLinearARG
from .wrapper import (
    _device_put_if_needed,
    _devices_for_blocks,
    _mesh_assembly_device,
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

        When `operator` is a multi-block
        [`linear_dag.core.jaxlinarg.JaxParallelOperator`][], call `matmat`
        directly. An additional outer `jax.jit` around the bound GRM method
        captures every block as a constant and bypasses device-local ownership.

    !!! Example

        ```python
        import jax.numpy as jnp

        from linear_dag import JaxGRMOperator, JaxLinearARG

        genotype = JaxLinearARG.from_hdf5_block("lineararg.h5", "block_0")
        grm = JaxGRMOperator(genotype, alpha=-1.0, center=True)
        relatedness_scores = grm @ jnp.ones(genotype.shape[0])
        ```

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
        else:
            result = self._device_local_matmat_rank2(x)
        return result

    def matmat_blockwise(self, x: ArrayLike) -> Array:
        """Multiply by the GRM using cached per-block JIT entrypoints.

        This path preserves the same algebra as
        [`linear_dag.core.jaxlinarg.JaxGRMOperator.matmat`][], but keeps the
        Python loop over LinearARG blocks outside a single XLA program. It is
        useful for workflows such as RHE where a full multi-block GRM product
        inside a larger `jax.jit` can make all block temporaries live in one HLO
        module.

        **Arguments:**

        - `x`: Rank-1 sample-space vector, or rank-2 sample-by-trait matrix.

        **Returns:**

        - GRM product with the same rank convention as the input.

        **Raises:**

        - `ValueError`: If `x` has an incompatible rank or leading dimension.
        """
        x, was_vector = _as_rank2_matrix(x, expected_rows=self.shape[0], dtype=self.dtype)
        result = self._matmat_blockwise_rank2(x)
        return result[:, 0] if was_vector else result

    def _matmat_blockwise_rank2(self, x: Array) -> Array:
        assembly_device = (
            _mesh_assembly_device(self.operator.mesh) if isinstance(self.operator, JaxParallelOperator) else None
        )
        result = _device_put_if_needed(jnp.zeros_like(x), assembly_device)
        # Keep each block product behind its own JIT boundary. This limits the
        # lifetime of block intermediates instead of building one monolithic HLO.
        for device, block in self._device_blocks:
            contribution = _block_grm_product_jit(
                block,
                _device_put_if_needed(x, device),
                alpha=self.alpha,
                center=self.center,
            )
            result = result + _device_put_if_needed(contribution, assembly_device)
        return result

    @cached_property
    def _device_blocks(self) -> tuple[tuple[jax.Device | None, JaxLinearARG], ...]:
        blocks = self.operator.blocks if isinstance(self.operator, JaxParallelOperator) else (self.operator,)
        devices = (
            _devices_for_blocks(
                self.operator.mesh,
                self.operator.block_ranges,
                n_blocks=len(blocks),
            )
            if isinstance(self.operator, JaxParallelOperator)
            else (None,)
        )
        return tuple(zip(devices, blocks, strict=True))

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

    def _device_local_matmat_rank2(self, x: Array) -> Array:
        if not isinstance(self.operator, JaxParallelOperator):
            raise TypeError("device-local GRM execution requires a JaxParallelOperator")
        contributions = [
            _grm_range_product_jit(
                blocks,
                _device_put_if_needed(x, device),
                alpha=self.alpha,
                center=self.center,
            )
            for device, blocks, _variant_start, _variant_end in self.operator._device_block_ranges
        ]
        assembly_device = _mesh_assembly_device(self.operator.mesh)
        result = _device_put_if_needed(jnp.zeros_like(x), assembly_device)
        for contribution in contributions:
            result = result + _device_put_if_needed(contribution, assembly_device)
        return result


@eqx.filter_jit
def _grm_range_product_jit(
    blocks: tuple[JaxLinearARG, ...],
    values: Array,
    *,
    alpha: float,
    center: bool,
) -> Array:
    result = jnp.zeros_like(values)
    for block in blocks:
        result = result + _block_grm_product(block, values, alpha=alpha, center=center)
    return result


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
        assert correction is not None
        result = result - correction
    return result

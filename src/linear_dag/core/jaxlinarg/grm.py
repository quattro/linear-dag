# pattern: Functional Core

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P
from jaxtyping import Array, ArrayLike

from .operator import JaxLinearARG
from .wrapper import (
    _as_rank2_matrix,
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
    safe_pq = jnp.where(pq > 0, pq, jnp.ones_like(pq))
    weights = jnp.where(pq > 0, safe_pq**alpha, jnp.zeros_like(pq))

    variant_scores = block.rmatmat(values)
    if center:
        # Apply `(X - 1 f^T)^T values` without forming centered genotypes:
        # `X^T values - f (1^T values)`.
        variant_scores = variant_scores - frequencies[:, None] * jnp.sum(values, axis=0, keepdims=True)
    weighted_scores = weights[:, None] * variant_scores

    result = block.matmat(weighted_scores)
    if center:
        # Apply the left centering factor:
        # `(X - 1 f^T) weighted_scores = X weighted_scores - 1 (f^T weighted_scores)`.
        correction = jnp.sum(frequencies[:, None] * weighted_scores, axis=0, keepdims=True)
        result = result - correction
    return result

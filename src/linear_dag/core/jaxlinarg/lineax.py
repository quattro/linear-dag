# pattern: Functional Core

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from .grm import JaxGRMOperator
from .operator import JaxLinearARG
from .wrapper import JaxParallelOperator


class _LinearARGOperator(lx.AbstractLinearOperator):
    operator: JaxLinearARG | JaxParallelOperator | JaxGRMOperator
    transposed: bool = eqx.field(default=False, static=True)

    def mv(self, vector):
        return self.operator.rmatvec(vector) if self.transposed else self.operator.matvec(vector)

    def as_matrix(self):
        # Use the smaller identity basis for rectangular genotype matrices.
        if self.in_size() > self.out_size():
            return self.transpose().as_matrix().T
        identity = jnp.eye(self.in_size(), dtype=self.operator.dtype)
        return self.operator.rmatmat(identity) if self.transposed else self.operator.matmat(identity)

    def transpose(self):
        if isinstance(self.operator, JaxGRMOperator):
            return self
        return _LinearARGOperator(self.operator, not self.transposed)

    def in_structure(self):
        return jax.ShapeDtypeStruct((self.operator.shape[0 if self.transposed else 1],), self.operator.dtype)

    def out_structure(self):
        return jax.ShapeDtypeStruct((self.operator.shape[1 if self.transposed else 0],), self.operator.dtype)


@lx.linearise.register(_LinearARGOperator)
def _linearise(operator):
    return operator


@lx.materialise.register(_LinearARGOperator)
def _materialise(operator):
    return lx.MatrixLinearOperator(operator.as_matrix(), tags=(lx.symmetric_tag,) if lx.is_symmetric(operator) else ())


@lx.diagonal.register(_LinearARGOperator)
def _diagonal(operator):
    return jnp.diag(operator.as_matrix())


@lx.tridiagonal.register(_LinearARGOperator)
def _tridiagonal(operator):
    matrix = operator.as_matrix()
    return jnp.diag(matrix), jnp.diag(matrix, -1), jnp.diag(matrix, 1)


@lx.is_symmetric.register(_LinearARGOperator)
def _is_symmetric(operator):
    return isinstance(operator.operator, JaxGRMOperator)


def _unknown_structure(operator):
    return False


# False means the structural property is not guaranteed, rather than a test
# of the current numerical values. In particular, dynamic GRM weights can be signed.
for _predicate in (
    lx.is_diagonal,
    lx.is_tridiagonal,
    lx.is_lower_triangular,
    lx.is_upper_triangular,
    lx.is_positive_semidefinite,
    lx.is_negative_semidefinite,
):
    _predicate.register(_LinearARGOperator)(_unknown_structure)


def as_lineax_operator(operator: JaxLinearARG | JaxParallelOperator | JaxGRMOperator) -> lx.AbstractLinearOperator:
    """Adapt a JAX genotype or GRM operator for Lineax composition and solvers.

    Graph arrays and dynamic GRM weights remain PyTree leaves. Products and
    transposes delegate directly to the wrapped operator without materializing
    its matrix. The wrapped backend determines supported AD transformations.

    !!! info
        GRMs are marked symmetric, but no positive-semidefinite tag is inferred:
        supplied diagonal weights may be signed. Choose solver tags only when
        guaranteed by the model. Explicit materialization, diagonal extraction,
        and direct solvers may allocate a dense matrix; use an iterative solver
        for matrix-free execution.

    !!! Example
        ```python
        import lineax as lx
        from linear_dag import as_lineax_operator

        x = as_lineax_operator(genotypes)
        scores = x.T.mv(residuals)
        identity = lx.IdentityLinearOperator(x.in_structure())
        regularized = lx.TaggedLinearOperator(
            x.T @ x + identity * 0.1, lx.positive_semidefinite_tag
        )
        solution = lx.linear_solve(regularized, rhs, lx.CG(rtol=1e-6, atol=1e-6))
        ```

    **Arguments:**

    - `operator`: A JAX single-block, multi-block, or GRM operator.

    **Returns:**

    - A Lineax linear operator over rank-1 arrays with matching shape and dtype.

    **Raises:**

    - `TypeError`: If the input is not a supported JAX operator.
    """
    if not isinstance(operator, (JaxLinearARG, JaxParallelOperator, JaxGRMOperator)):
        raise TypeError("expected JaxLinearARG, JaxParallelOperator, or JaxGRMOperator")
    return _LinearARGOperator(operator)

from typing import Optional, Union

import numpy as np

from numpy.random import BitGenerator, Generator, SeedSequence
from scipy.sparse.linalg import LinearOperator


def simulate_phenotype(
    genotypes: LinearOperator,
    heritability: float,
    fraction_causal: float = 1,
    num_traits: int = 1,
    return_genetic_component: bool = False,
    return_beta: bool = False,
    variant_effect_variance: Optional[np.ndarray] = None,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
):
    """Simulate quantitative phenotypes from a genotype linear operator.

    !!! info

        Traits are generated from $y = X\\beta + \\epsilon$, where $X$ is the
        genotype operator, $\\beta$ are sampled variant effects, and
        $\\epsilon$ is Gaussian noise scaled to match the requested
        heritability.

    **Arguments:**

    - `genotypes`: Genotype linear operator with shape
      `(n_samples, n_variants)`.
    - `heritability`: Target narrow-sense heritability in `[0, 1]`.
    - `fraction_causal`: Fraction of variants with non-zero effects.
    - `num_traits`: Number of traits to simulate.
    - `return_genetic_component`: If `True`, also return the genetic
      component `y_bar`.
    - `return_beta`: If `True`, also return sampled effects `beta`.
    - `variant_effect_variance`: Optional per-variant scaling for effect-size
      variance.
    - `seed`: Optional NumPy-compatible RNG seed or generator.

    **Returns:**

    - Simulated phenotype matrix `y`, or tuples `(y, beta)` / `(y, y_bar)`
      depending on output flags.

    **Raises:**

    - `ValueError`: If sampled effects produce zero genetic variance and
      scaling would divide by zero.
    """
    N, M = genotypes.shape
    rng = np.random.default_rng(seed)

    beta = rng.standard_normal(size=(M, num_traits), dtype=np.float32)

    # randomly select what is causal and zero-out the corresponding effects at non-causals
    if fraction_causal < 1:
        is_causal = rng.binomial(1, fraction_causal, size=M)
        beta *= is_causal.reshape(-1, 1)

    if variant_effect_variance is not None:
        beta *= np.sqrt(variant_effect_variance).reshape(-1, 1)

    y_bar = genotypes @ beta
    y_bar -= np.mean(y_bar, axis=0)

    # Standardize y_bar
    multiplier = np.sqrt(heritability) / np.std(y_bar, axis=0)
    if np.any(np.isinf(multiplier)):
        raise ValueError("Divide by zero error occurred, perhaps because no causal variants were sampled.")

    multiplier[np.isnan(multiplier)] = 0  # heritability == 0
    y_bar *= multiplier
    beta *= multiplier

    y = y_bar + rng.normal(scale=np.sqrt(1 - heritability), size=y_bar.shape)

    if return_beta:
        return y, beta
    if return_genetic_component:
        return y, y_bar

    return y

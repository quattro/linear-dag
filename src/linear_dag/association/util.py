import numpy as np

from scipy.sparse.linalg import LinearOperator


def _backslash(A: np.ndarray, b: np.ndarray, lam: float = 1e-5) -> np.ndarray:
    """MATLAB-style backslash"""
    # return np.linalg.solve(A.T @ A, A.T @ b)
    return np.linalg.pinv(A.T @ A, rcond=lam) @ (A.T @ b)


def _residualize_phenotypes_mar(
    phenotypes: np.ndarray, covariates: np.ndarray, phenotypes_missing: np.ndarray
) -> np.ndarray:
    """Residualize phenotypes on covariates assuming missing at random (MAR),
    i.e., covariance matrix of the covariates is the same for individuals with and without missing data.
    Input phenotypes should be zero for missing values."""
    beta = _backslash(covariates, phenotypes)
    residuals = phenotypes - (covariates @ beta) / (1 - np.mean(phenotypes_missing, axis=0))
    residuals.ravel()[phenotypes_missing.ravel()] = 0
    return residuals


def residualize_phenotypes(
    phenotypes: np.ndarray,
    covariates: np.ndarray,
    phenotypes_missing: np.ndarray,
    missingness_threshold_mar: float = 0,
) -> np.ndarray:
    """Residualize phenotypes on covariates with missingness-aware handling.

    !!! info

        Phenotypes with missingness below `missingness_threshold_mar` are
        processed with the MAR approximation; higher-missingness traits are
        residualized trait-wise on non-missing rows.

    **Arguments:**

    - `phenotypes`: Phenotype matrix of shape `(n_samples, n_traits)`.
    - `covariates`: Covariate matrix whose first column is typically intercept.
    - `phenotypes_missing`: Boolean matrix marking missing phenotype entries.
    - `missingness_threshold_mar`: Maximum missingness fraction for MAR path.

    **Returns:**

    - Residualized phenotype matrix with the same shape as `phenotypes`.
    """
    residuals = _residualize_phenotypes_mar(phenotypes, covariates, phenotypes_missing)
    for i in range(phenotypes.shape[1]):
        nonmissing = ~phenotypes_missing[:, i]
        if np.mean(nonmissing) >= 1 - missingness_threshold_mar:
            continue
        beta = _backslash(covariates[nonmissing, :], phenotypes[nonmissing, i])
        residuals[nonmissing, i] = phenotypes[nonmissing, i] - (covariates[nonmissing, :] @ beta)
    return residuals


def get_genotype_variance_explained(
    XtC: np.ndarray,
    C: np.ndarray,
    num_heterozygotes: np.ndarray | None,
    batch_size: int = 100_000,
    lam: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute covariate-explained genotype variance terms.

    !!! info

        This computes
        $\\mathrm{diag}(X^\\top C(C^\\top C)^{-1}C^\\top X)$ in batches, where
        $X$ is genotype dosage and $C$ is the covariate matrix.

    **Arguments:**

    - `XtC`: Matrix product $X^\\top C$.
    - `C`: Covariate matrix (intercept in first column for standard usage).
    - `num_heterozygotes`: Optional heterozygote counts for non-HWE denominator
      adjustment.
    - `batch_size`: Number of variants to process per batch.
    - `lam`: Pseudoinverse regularization parameter.

    **Returns:**

    - Tuple `(denominator, allele_counts)` used by downstream association
      variance calculations.
    """
    num_covar = C.shape[1]
    covariate_inner = C.T @ C

    num_snps = XtC.shape[0]
    total_var_explained = np.zeros((num_snps, 1), dtype=np.float32)

    for start_idx in range(0, num_snps, batch_size):
        end_idx = min(start_idx + batch_size, num_snps)
        XtC_batch = XtC[start_idx:end_idx, :num_covar]
        # C_backslash_XtC_batch = np.linalg.solve(covariate_inner, XtC_batch.T).astype(np.float32)
        C_backslash_XtC_batch = (np.linalg.pinv(covariate_inner, rcond=lam) @ XtC_batch.T).astype(np.float32)
        total_var_explained[start_idx:end_idx] = np.sum(XtC_batch.T * C_backslash_XtC_batch, axis=0).reshape(-1, 1)

    allele_counts = XtC[:, 0:1]
    denominator = allele_counts - total_var_explained
    if num_heterozygotes is not None:  # else assume HWE
        # assumes diploid
        num_homozygotes = (allele_counts - num_heterozygotes.reshape(-1, 1)) / 2
        denominator = denominator + 2 * num_homozygotes - total_var_explained

    return denominator, allele_counts


def get_genotype_variance_explained_recompute_AC(
    XtCD: np.ndarray,
    C: np.ndarray,
    num_heterozygotes: np.ndarray | None,
    num_nonmissing: np.ndarray | None = None,
    batch_size: int = 100_000,
    lam: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute covariate-explained genotype variance with per-trait AC updates.

    !!! info

        Input matrix `XtCD` is partitioned as
        $[X^\\top C\\;\\;X^\\top D]$, where $D$ encodes non-missing phenotype
        indicators. This allows trait-specific allele-count recomputation.

    **Arguments:**

    - `XtCD`: Concatenated matrix $[X^\\top C\\;\\;X^\\top D]$.
    - `C`: Covariate matrix.
    - `num_heterozygotes`: Optional heterozygote counts for non-HWE adjustment.
    - `num_nonmissing`: Non-missing counts per trait (column sums of $D$).
    - `batch_size`: Number of variants processed per chunk.
    - `lam`: Pseudoinverse regularization parameter.

    **Returns:**

    - Tuple `(denominator, total_allele_counts)` aligned to variant order.
    """
    n, num_covar = C.shape
    if num_covar + len(num_nonmissing) != XtCD.shape[1]:
        raise ValueError("XtCD must have the same number of columns as C and D")

    # Flip cases where AF > 0.5 to avoid approximation error when AF~1
    # Denominator is insensitive to allele flipping
    # (1-X)'C == sum(C) - X'C
    mask = XtCD[:, 0] > n / 2
    XtCD[mask, :] *= -1
    XtCD[mask, :num_covar] += np.sum(C, axis=0)
    XtCD[mask, num_covar:] += num_nonmissing

    # total_var_explained: misleading variable name for in-place operations to work
    total_var_explained, total_allele_counts = get_genotype_variance_explained(
        XtCD[:, :num_covar], C, None, batch_size, lam
    )

    # n * v
    total_var_explained *= -1
    total_var_explained += total_allele_counts

    # n' * p'
    allele_counts = XtCD[:, num_covar:]

    # (p'/p)^2 * n' * v
    # one large memory allocation
    denominator = allele_counts.copy() ** 2
    denominator *= (n / num_nonmissing).astype(np.float32)
    denominator *= total_var_explained / total_allele_counts.reshape(-1, 1) ** 2
    np.nan_to_num(denominator, copy=False, nan=0.0)

    if num_heterozygotes is None:  # assume HWE
        denominator *= -1
        denominator += allele_counts
    else:
        # assumes diploid
        num_homozygotes = (total_allele_counts - num_heterozygotes.reshape(-1, 1)) / 2
        denominator *= -2
        # avoid large memory allocation
        num_snps = XtCD.shape[0]

        for start_idx in range(0, num_snps, batch_size):
            end_idx = min(start_idx + batch_size, num_snps)

            batch_homozygotes = num_homozygotes[start_idx:end_idx]
            batch_allele_counts = allele_counts[start_idx:end_idx]

            denominator[start_idx:end_idx] += batch_allele_counts + 2 * batch_homozygotes * num_nonmissing / n

    # un-flip alleles
    total_allele_counts[mask] = n - total_allele_counts[mask]

    assert denominator.dtype == np.float32
    return denominator, total_allele_counts


def _get_genotype_variance(
    genotypes: LinearOperator,
    allele_counts: np.ndarray,
    individuals_to_include: np.ndarray,
) -> tuple[np.int64, np.ndarray]:
    """Get genotype variance without Hardy-Weinberg assumptions.

    !!! info

        Uses carrier counts to evaluate
        $\\mathrm{diag}(X^\\top X)$ from individual-level genotype operators.

    **Arguments:**

    - `genotypes`: Unnormalized genotype operator with individual nodes.
    - `allele_counts`: Per-variant allele counts.
    - `individuals_to_include`: Indices of non-missing individuals.

    **Returns:**

    - Tuple `(var_genotypes, carrier_counts)` for each variant.
    """
    carrier_counts = genotypes.number_of_carriers(individuals_to_include).reshape(-1, 1)
    assert np.all(allele_counts - carrier_counts >= 0)
    var_genotypes = 3 * allele_counts - 2 * carrier_counts  # 4 * num_homozygotes + num_heterozygotes
    return var_genotypes, carrier_counts


def impute_missing_with_mean(data: np.ndarray) -> np.ndarray:
    """Impute missing entries column-wise with observed means.

    **Arguments:**

    - `data`: Dense matrix that may contain `NaN` values.

    **Returns:**

    - Copy of `data` with missing values replaced by per-column means.
    """
    data = data.copy()
    for col in range(data.shape[1]):
        is_missing = np.isnan(data[:, col])
        col_mean = np.mean(data[~is_missing, col])
        data[is_missing, col] = col_mean
    return data

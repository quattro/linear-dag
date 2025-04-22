from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

# annoying 'as' notation to avoid warnings/errors about unused imports...
from .association import (
    randomized_haseman_elston as randomized_haseman_elston,
)
from .core import (
    BrickGraph as BrickGraph,
    linear_arg_from_genotypes as linear_arg_from_genotypes,
    LinearARG as LinearARG,
    list_blocks as list_blocks,
)
from .genotype import (
    apply_maf_threshold as apply_maf_threshold,
    binarize as binarize,
    compute_af as compute_af,
    flip_alleles as flip_alleles,
    read_vcf as read_vcf,
)
from .structure import (
    pca as pca,
    svd as svd,
)

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

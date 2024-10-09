from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
from importlib.util import find_spec

# annoying 'as' notation to avoid warnings/errors about unused imports...
from .brick_graph import BrickGraph as BrickGraph
from .data_structures import (
    CountingArray as CountingArray,
    DiGraph as DiGraph,
    LinkedListArray as LinkedListArray,
    ModHeap as ModHeap,
    Queue as Queue,
    Stack as Stack,
)
from .genotype import (
    apply_maf_threshold as apply_maf_threshold,
    binarize as binarize,
    compute_af as compute_af,
    flip_alleles as flip_alleles,
    read_vcf as read_vcf,
)
from .lineararg import LinearARG as LinearARG, VariantInfo as VariantInfo
from .recombination import Recombination as Recombination
from .sample_info import SampleInfo as SampleInfo
from .simulate import Simulate as Simulate
from .solve import (
    spinv_triangular as spinv_triangular,
    topological_sort as topological_sort,
)

# dna_nexus and pyspark are optional dependencies
if find_spec("dna_nexus") and find_spec("pyspark"):
    from .dna_nexus import (
        download_vcf as download_vcf,
        find_shapeit200k_vcf as find_shapeit200k_vcf,
    )

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

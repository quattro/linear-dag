# SPDX-FileCopyrightText: 2024-present Nicholas Mancuso <nmancuso@usc.edu>
#
# SPDX-License-Identifier: MIT
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

from .brick_graph import (
    BrickGraph as BrickGraph,
)
from .brick_graph_py import (
    BrickGraphPy as BrickGraphPy,
)
from .data_structures import (
    CountingArray as CountingArray,
    DiGraph as DiGraph,
    LinkedListArray as LinkedListArray,
    ModHeap as ModHeap,
    Queue as Queue,
    Stack as Stack,
)
from .genotype_processing import (
    apply_maf_threshold as apply_maf_threshold,
    binarize as binarize,
    compute_af as compute_af,
    flip_alleles as flip_alleles,
)

# annoying 'as' notation to avoid warnings/errors about unused imports...
from .linarg_workflow import (
    run_linarg_workflow as run_linarg_workflow,
)
from .linear_arg_inference import (
    add_samples_to_linear_arg as add_samples_to_linear_arg,
    add_singleton_variants as add_singleton_variants,
    infer_brick_graph_using_containment as infer_brick_graph_using_containment,
    linearize_brick_graph as linearize_brick_graph,
)
from .lineararg import (
    LinearARG as LinearARG,
)
from .one_summed import (
    compute_path_sums as compute_path_sums,
    construct_1_summed_DAG_fast as construct_1_summed_DAG_fast,
    construct_1_summed_DAG_slow as construct_1_summed_DAG_slow,
)
from .simulate import (
    Simulate as Simulate,
)
from .solve import (
    spinv_triangular as spinv_triangular,
    topological_sort as topological_sort,
)
from .trios import (
    Trios as Trios,
)


try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

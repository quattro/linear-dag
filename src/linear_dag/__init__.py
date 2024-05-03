# SPDX-FileCopyrightText: 2024-present Nicholas Mancuso <nmancuso@usc.edu>
#
# SPDX-License-Identifier: MIT
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

from .linarg_workflow import (
    run_linarg_workflow as run_linarg_workflow,
)
from .lineararg import (
    LinearARG as LinearARG,
)

# from .intervals import (
#     Intervals as Intervals,
# )
# annoying 'as' notation to avoid warnings/errors about unused imports...
from .one_summed import (
    compute_path_sums as compute_path_sums,
    construct_1_summed_DAG_fast as construct_1_summed_DAG_fast,
    construct_1_summed_DAG_slow as construct_1_summed_DAG_slow,
)
from .pathsumdag import (
    PathSumDAG as PathSumDAG,
)

# from .pathsumdag_old import (
#     PathSumDAGTesting as PathSumDAGTesting,
# )
from .simulate import (
    Simulate as Simulate,
)
from .solve import (
    spinv_triangular as spinv_triangular,
)
from .trios import (
    LinkedListArray as LinkedListArray,
    Trios as Trios,
)
from .utils import (
    apply_maf_threshold as apply_maf_threshold,
    binarize as binarize,
    compute_af as compute_af,
    flip_alleles as flip_alleles,
)


try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

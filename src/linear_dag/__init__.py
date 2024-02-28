# SPDX-FileCopyrightText: 2024-present Nicholas Mancuso <nmancuso@usc.edu>
#
# SPDX-License-Identifier: MIT
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

# annoying 'as' notation to avoid warnings/errors about unused imports...
from .core import (
    compute_path_sums as compute_path_sums,
    construct_1_summed_DAG_fast as construct_1_summed_DAG_fast,
    construct_1_summed_DAG_slow as construct_1_summed_DAG_slow,
)
from .trios import (
    Trios as Trios,
    LinkedListArray as LinkedListArray,
)


try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

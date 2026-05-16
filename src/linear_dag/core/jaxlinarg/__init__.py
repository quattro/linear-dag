# pattern: Functional Core

from .build_config import show_build_config as show_build_config
from .grm import JaxGRMOperator as JaxGRMOperator
from .operator import Backend as Backend, JaxLinearARG as JaxLinearARG
from .wrapper import (
    JaxParallelOperator as JaxParallelOperator,
    split_blocks_by_n_entries as split_blocks_by_n_entries,
    variant_offsets_from_metadata as variant_offsets_from_metadata,
)

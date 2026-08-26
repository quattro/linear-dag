# pattern: Imperative Shell

from __future__ import annotations

import pytest

from tests.jax.oracle import make_oracle_cases


@pytest.fixture(params=("unflipped_k1", "unflipped_k3", "unflipped_vector", "flipped_k3"))
def oracle_case(request, linarg_h5_path, first_block_name):
    cases = {case.name: case for case in make_oracle_cases(linarg_h5_path, first_block_name)}
    return cases[request.param]

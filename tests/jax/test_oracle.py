# pattern: Imperative Shell

from __future__ import annotations

import numpy as np

from tests.jax.oracle import make_oracle_cases


def test_oracle_cases_match_lineararg_operator_shapes(linarg_h5_path, first_block_name):
    cases = make_oracle_cases(linarg_h5_path, first_block_name)

    for case in cases:
        assert case.Xw.shape == (case.linarg.shape[0],) + case.w.shape[1:]
        assert case.XTy.shape == (case.linarg.shape[1],) + case.y.shape[1:]
        assert case.Xw.shape == (case.linarg @ case.w).shape
        assert case.XTy.shape == (case.linarg.T @ case.y).shape


def test_oracle_cases_include_flipped_variant_case(linarg_h5_path, first_block_name):
    cases = make_oracle_cases(linarg_h5_path, first_block_name)

    assert any(case.flip_prob > 0 for case in cases)


def test_oracle_case_arrays_are_float32_unless_operator_widens(oracle_case):
    assert oracle_case.w.dtype == np.float32
    assert oracle_case.y.dtype == np.float32
    assert oracle_case.Xw.dtype in (np.float32, np.float64)
    assert oracle_case.XTy.dtype in (np.float32, np.float64)

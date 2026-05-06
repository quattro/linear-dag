# pattern: Imperative Shell

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from linear_dag.core.lineararg import LinearARG
from tests.helpers.linarg_fixtures import load_lineararg_block


class OracleCase(NamedTuple):
    name: str
    linarg: LinearARG
    w: np.ndarray
    y: np.ndarray
    Xw: np.ndarray
    XTy: np.ndarray
    flip_prob: float


def make_oracle_cases(linarg_h5_path: Path, first_block_name: str) -> list[OracleCase]:
    rng = np.random.default_rng(20260506)
    linarg = load_lineararg_block(linarg_h5_path, block_name=first_block_name)

    cases = [
        _make_case("unflipped_k1", linarg, _draw_inputs(rng, linarg, k=1)),
        _make_case("unflipped_k3", linarg, _draw_inputs(rng, linarg, k=3)),
        _make_case("unflipped_vector", linarg, _draw_inputs(rng, linarg, k=None)),
    ]

    flipped = linarg.copy()
    flipped.flip = _deterministic_flip_mask(flipped.shape[1])
    cases.append(_make_case("flipped_k3", flipped, _draw_inputs(rng, flipped, k=3)))
    return cases


def _draw_inputs(
    rng: np.random.Generator,
    linarg: LinearARG,
    *,
    k: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if k is None:
        w_shape = (linarg.shape[1],)
        y_shape = (linarg.shape[0],)
    else:
        w_shape = (linarg.shape[1], k)
        y_shape = (linarg.shape[0], k)

    w = rng.normal(size=w_shape).astype(np.float32)
    y = rng.normal(size=y_shape).astype(np.float32)
    return w, y


def _make_case(
    name: str,
    linarg: LinearARG,
    inputs: tuple[np.ndarray, np.ndarray],
) -> OracleCase:
    w, y = inputs
    return OracleCase(
        name=name,
        linarg=linarg,
        w=w,
        y=y,
        Xw=np.asarray(linarg @ w),
        XTy=np.asarray(linarg.T @ y),
        flip_prob=float(np.mean(linarg.flip)),
    )


def _deterministic_flip_mask(n_variants: int) -> np.ndarray:
    mask = np.zeros(n_variants, dtype=np.bool_)
    mask[::3] = True
    if n_variants > 0:
        mask[0] = True
    return mask

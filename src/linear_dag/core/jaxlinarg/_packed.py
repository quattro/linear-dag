# pattern: Mixed (host packing and pure numerical primitive rules)

import operator
import os

from dataclasses import dataclass
from enum import IntEnum

import jax
import jax.numpy as jnp
import numpy as np

from jax.experimental.hijax import GradAccum, HiPrim, instantiate_zeros, linearize_from_jvp, ShapedArray, Zero


# Integer values and graph tuple order form the ABI with kernels/packed_cpu.cc.
# Graph buffers: metadata, CSC indptr, edges, edge weights, workspace mapping,
# variant nodes, sample nodes, flips, allele counts. Metadata rows contain:
# [ptr_offset, node_offset, edge_offset, variant_offset, nodes, variants,
#  workspace_rows, min_index_to_keep]. Node/edge indices stay block-local.
class _Mode(IntEnum):
    FORWARD = 0
    REVERSE = 1
    WEIGHTED = 2
    DIAGONAL_ADJOINT = 3


@dataclass(frozen=True)
class _ExecutionConfig:
    samples: int
    variants: int
    max_state: int
    max_variants: int
    workers: int
    chunk: int
    center: bool = False

    def scratch_bytes(self, itemsize: int = 4) -> int:
        return self.workers * (self.max_state + 2 * self.max_variants + self.samples + 2) * self.chunk * itemsize


def _execution_config(
    samples: int,
    variants: int,
    max_state: int,
    max_variants: int,
    blocks: int,
    *,
    workers: int | None = None,
    chunk: int = 8,
    workspace_bytes: int = 256 * 1024**2,
    dtype=np.float32,
) -> _ExecutionConfig:
    workers = min(os.cpu_count() or 1, blocks) if workers is None else operator.index(workers)
    chunk = operator.index(chunk)
    workspace_bytes = operator.index(workspace_bytes)
    if workers < 1 or chunk < 1 or workspace_bytes < 1:
        raise ValueError("num_processes, max_num_traits and workspace_bytes must be positive")
    # Each worker owns graph state, two variant-score tiles, a sample accumulator,
    # and two reserved RHS rows. This matches the native scratch layout.
    per_column = (max_state + 2 * max_variants + samples + 2) * np.dtype(dtype).itemsize
    if workspace_bytes < per_column:
        raise ValueError(f"workspace_bytes must be at least {per_column} for one worker and RHS column")
    workers = min(workers, blocks, workspace_bytes // per_column)
    chunk = min(chunk, workspace_bytes // (workers * per_column))
    return _ExecutionConfig(samples, variants, max_state, max_variants, workers, chunk)


def _pack(blocks, *, workers=None, chunk=8, workspace_bytes=256 * 1024**2, dtype=np.float32):
    # Host-only packing: callers establish sample identity/order across blocks.
    # Validate int32 capacity before narrowing; the packed format is in-memory
    # execution state, not a new on-disk schema.
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("blocks must be nonempty")
    keys = ("indptr", "indices", "data", "nonunique_indices", "variant_indices", "sample_indices", "flip")
    arrays = [[] for _ in keys]
    counts = []
    metadata = []
    po = no = eo = vo = 0
    n_samples = None
    max_state = max_variants = 0
    for block in blocks:
        raw = [np.asarray(block[key]) for key in keys]
        for key, value in zip(keys[:-1], raw[:-1]):
            if value.ndim != 1 or not np.all(np.isfinite(value)) or np.any(value != np.round(value)):
                raise ValueError(f"{key} must contain finite integers")
            if value.size and (value.min() < np.iinfo(np.int32).min or value.max() > np.iinfo(np.int32).max):
                raise ValueError(f"{key} exceeds int32 capacity")
        p, idx, data, mapping, v, s = [a.astype(np.int32) for a in raw[:-1]]
        flip = raw[-1]
        n = len(p) - 1
        if n < 1 or p[0] != 0 or p[-1] != len(idx) or np.any(np.diff(p.astype(np.int64)) < 0) or len(data) != len(idx):
            raise ValueError("invalid CSC structure")
        if len(mapping) != n or np.any(mapping < 0):
            raise ValueError("invalid workspace mapping")
        for a in (idx, v, s):
            if np.any(a < 0) or np.any(a >= n):
                raise ValueError("node index out of bounds")
        sources = np.repeat(np.arange(n, dtype=np.int32), np.diff(p))
        if np.any(idx < sources) or np.any((idx == sources) & (data != 0)):
            raise ValueError("graph must be strictly forward ordered")
        if flip.shape != v.shape or np.any((flip != 0) & (flip != 1)):
            raise ValueError("flip must be aligned booleans")
        if n_samples is None:
            n_samples = len(s)
        if not len(s) or len(s) != n_samples or len(np.unique(mapping[s])) != len(s):
            raise ValueError("sample rows must be nonempty, aligned and distinct")
        ns = int(mapping.max()) + 1
        keep = int(block["min_index_to_keep"])
        if keep < 0 or keep >= n:
            raise ValueError("invalid retained-node boundary")
        metadata.append([po, no, eo, vo, n, len(v), ns, keep])
        for target, value in zip(arrays, (p, idx, data, mapping, v, s, flip.astype(bool))):
            target.append(value)
        po += len(p)
        no += n
        eo += len(idx)
        vo += len(v)
        if max(po, no, eo, vo, ns, len(blocks) * n_samples) > np.iinfo(np.int32).max:
            raise ValueError("packed offsets exceed int32 capacity")
        max_state = max(max_state, ns)
        max_variants = max(max_variants, len(v))
        count = np.asarray(block.get("allele_counts", np.full(len(v), -1)), dtype=np.int32)
        if count.shape != v.shape:
            raise ValueError("allele_counts must be aligned with variants")
        counts.append(count)
    config = _execution_config(
        n_samples,
        vo,
        max_state,
        max_variants,
        len(blocks),
        workers=workers,
        chunk=chunk,
        workspace_bytes=workspace_bytes,
        dtype=dtype,
    )
    graph = (
        jnp.asarray(np.asarray(metadata, dtype=np.int32)),
        *(jnp.asarray(np.concatenate(a)) for a in arrays),
        jnp.asarray(np.concatenate(counts)),
    )
    return graph, config


def _call(graph, a, b, cfg, mode):
    from .kernels import ffi_cpu

    if not ffi_cpu.is_ffi_cpu_available():
        raise RuntimeError("Packed CPU FFI handler is unavailable; rebuild linear-dag")
    k = (b if mode == _Mode.WEIGHTED else a).shape[1]
    rows = cfg.variants if mode in (_Mode.REVERSE, _Mode.DIAGONAL_ADJOINT) else cfg.samples
    result_shape = (rows, 1 if mode == _Mode.DIAGONAL_ADJOINT else k)
    scratch_rows = cfg.max_state + 2 * cfg.max_variants + cfg.samples + 2
    target = "linear_dag_packed_f32" if a.dtype == jnp.float32 else "linear_dag_packed_f64"
    call = jax.ffi.ffi_call(
        target,
        (jax.ShapeDtypeStruct(result_shape, a.dtype), jax.ShapeDtypeStruct((cfg.workers, scratch_rows, k), a.dtype)),
        vmap_method="sequential",
    )
    result, _ = call(
        *graph,
        a,
        b,
        mode=int(mode),
        n_samples=cfg.samples,
        max_state=cfg.max_state,
        max_variants=cfg.max_variants,
        center=int(cfg.center),
    )
    return result


def _execute(graph, a, b, cfg, mode):
    # A sequential XLA loop bounds scratch width regardless of input width.
    k = (b if mode == _Mode.WEIGHTED else a).shape[1]
    rows = cfg.variants if mode in (_Mode.REVERSE, _Mode.DIAGONAL_ADJOINT) else cfg.samples
    if k == 0:
        return jnp.zeros((rows, 1 if mode == _Mode.DIAGONAL_ADJOINT else 0), dtype=a.dtype)
    width = min(k, cfg.chunk)
    if k <= width:
        return _call(graph, a, b, cfg, mode)
    tiles = (k + width - 1) // width
    pad = tiles * width - k
    aa = a if mode == _Mode.WEIGHTED else jnp.pad(a, ((0, 0), (0, pad)))
    bb = jnp.pad(b, ((0, 0), (0, pad))) if mode in (_Mode.WEIGHTED, _Mode.DIAGONAL_ADJOINT) else b
    result = jnp.zeros((rows, 1 if mode == _Mode.DIAGONAL_ADJOINT else tiles * width), a.dtype)

    def step(i, output):
        x = aa if mode == _Mode.WEIGHTED else jax.lax.dynamic_slice_in_dim(aa, i * width, width, axis=1)
        y = (
            jax.lax.dynamic_slice_in_dim(bb, i * width, width, axis=1)
            if mode in (_Mode.WEIGHTED, _Mode.DIAGONAL_ADJOINT)
            else bb
        )
        tile = _call(graph, x, y, cfg, mode)
        return (
            output + tile
            if mode == _Mode.DIAGONAL_ADJOINT
            else jax.lax.dynamic_update_slice(output, tile, (0, i * width))
        )

    result = jax.lax.fori_loop(0, tiles, step, result)
    return result if mode == _Mode.DIAGONAL_ADJOINT else result[:, :k]


def _zero_graph(graph):
    return jax.tree.map(lambda x: Zero(jax.typeof(x).to_tangent_aval()), graph)


def _bind(graph, a, b, cfg, mode):
    return _PackedOperation(jax.tree.map(jax.typeof, graph), jax.typeof(a), jax.typeof(b), cfg, mode)(graph, a, b)


class _PackedOperation(HiPrim):
    def __init__(self, graph_types, a_type, b_type, cfg, mode):
        self.in_avals = (graph_types, a_type, b_type)
        k = (b_type if mode == _Mode.WEIGHTED else a_type).shape[1]
        rows = cfg.variants if mode in (_Mode.REVERSE, _Mode.DIAGONAL_ADJOINT) else cfg.samples
        self.out_aval = ShapedArray((rows, 1 if mode == _Mode.DIAGONAL_ADJOINT else k), a_type.dtype)
        self.params = dict(cfg=cfg, mode=mode)
        super().__init__()

    def expand(self, graph, a, b):
        return _execute(graph, a, b, self.cfg, self.mode)

    # Modes 0/1 are linear; modes 2/3 are bilinear in the floating operands.
    # The diagonal adjoint is sum_r (X.T a)[j,r] * (X.T b)[j,r].
    def jvp(self, primals, tangents):
        graph, a, b = primals
        _, da, db = tangents
        result = self(graph, a, b)
        if self.mode in (_Mode.FORWARD, _Mode.REVERSE):
            tangent = _bind(graph, instantiate_zeros(da), b, self.cfg, self.mode)
        else:
            tangent = _bind(graph, instantiate_zeros(da), b, self.cfg, self.mode)
            tangent = tangent + _bind(graph, a, instantiate_zeros(db), self.cfg, self.mode)
        return result, tangent

    lin, linearized = linearize_from_jvp

    def vjp_fwd(self, nzs_in, graph, a, b):
        # Save only original inputs; native workspaces and variant scores are
        # not reverse-mode residuals. Backward recomputes per-block scores.
        return self(graph, a, b), (graph, a, b)

    def vjp_bwd_retval(self, residual, cotangent):
        graph, a, b = residual
        g = instantiate_zeros(cotangent)
        dummy = jnp.zeros((1, 1), a.dtype)
        if self.mode in (_Mode.FORWARD, _Mode.REVERSE):
            da = _bind(graph, g, dummy, self.cfg, _Mode.REVERSE if self.mode == _Mode.FORWARD else _Mode.FORWARD)
            db = jnp.zeros_like(b)
        elif self.mode == _Mode.WEIGHTED:
            da = _bind(graph, b, g, self.cfg, _Mode.DIAGONAL_ADJOINT)
            db = _bind(graph, a, g, self.cfg, _Mode.WEIGHTED)
        else:
            da = _bind(graph, g, b, self.cfg, _Mode.WEIGHTED)
            db = _bind(graph, g, a, self.cfg, _Mode.WEIGHTED)
        return _zero_graph(graph), da, db

    def batch(self, axis_data, args, dims):
        graph, a, b = args
        graph_dims, ad, bd = dims
        if any(d is not None for d in jax.tree.leaves(graph_dims)):
            raise ValueError("mapping over different graphs is not supported")
        if ad is None and bd is None:
            return self(graph, a, b), None
        # Collapse shared-graph/shared-diagonal RHS batches into matrix columns.
        rhs_axis = ad if self.mode in (_Mode.FORWARD, _Mode.REVERSE) else bd
        if self.mode in (_Mode.FORWARD, _Mode.REVERSE) or (self.mode == _Mode.WEIGHTED and ad is None):
            rhs = a if self.mode in (_Mode.FORWARD, _Mode.REVERSE) else b
            if rhs_axis is None:
                return self(graph, a, b), None
            rhs = jnp.moveaxis(rhs, rhs_axis, 0)
            batch, rows, width = rhs.shape
            flat = jnp.transpose(rhs, (1, 0, 2)).reshape(rows, batch * width)
            result = (
                _bind(graph, flat, b, self.cfg, self.mode)
                if self.mode in (_Mode.FORWARD, _Mode.REVERSE)
                else _bind(graph, a, flat, self.cfg, _Mode.WEIGHTED)
            )
            return result.reshape(result.shape[0], batch, width).transpose(1, 0, 2), 0
        # Independently varying diagonals and diagonal adjoints are mapped as
        # whole operations. This is correct but intentionally not optimized yet.
        size = a.shape[ad] if ad is not None else b.shape[bd]
        aa = jnp.broadcast_to(a, (size, *a.shape)) if ad is None else jnp.moveaxis(a, ad, 0)
        bb = jnp.broadcast_to(b, (size, *b.shape)) if bd is None else jnp.moveaxis(b, bd, 0)
        return jax.lax.map(lambda pair: _bind(graph, pair[0], pair[1], self.cfg, self.mode), (aa, bb)), 0

    def transpose(self, out_ct, graph, a, b):
        g = instantiate_zeros(out_ct)
        if self.mode in (_Mode.FORWARD, _Mode.REVERSE):
            if isinstance(a, GradAccum):
                a.accum(
                    _bind(
                        graph,
                        g,
                        jnp.zeros((1, 1), g.dtype),
                        self.cfg,
                        _Mode.REVERSE if self.mode == _Mode.FORWARD else _Mode.FORWARD,
                    )
                )
        elif self.mode == _Mode.WEIGHTED:
            if isinstance(a, GradAccum):
                a.accum(_bind(graph, b, g, self.cfg, _Mode.DIAGONAL_ADJOINT))
            if isinstance(b, GradAccum):
                b.accum(_bind(graph, a, g, self.cfg, _Mode.WEIGHTED))
        else:
            if isinstance(a, GradAccum):
                a.accum(_bind(graph, g, b, self.cfg, _Mode.WEIGHTED))
            if isinstance(b, GradAccum):
                b.accum(_bind(graph, g, a, self.cfg, _Mode.WEIGHTED))


def _matrix(value, rows):
    value = jnp.asarray(value)
    if value.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("float32 or float64 inputs required")
    if value.ndim not in (1, 2) or value.shape[0] != rows:
        raise ValueError("incompatible product input shape")
    return (value[:, None], True) if value.ndim == 1 else (value, False)


def _product(graph, values, cfg, *, transpose=False):
    values, vector = _matrix(values, cfg.samples if transpose else cfg.variants)
    result = _bind(graph, values, jnp.zeros((1, 1), values.dtype), cfg, _Mode.REVERSE if transpose else _Mode.FORWARD)
    return result[:, 0] if vector else result


def _weighted(graph, diagonal, values, cfg):
    values, vector = _matrix(values, cfg.samples)
    diagonal = jnp.asarray(diagonal)
    if diagonal.shape != (cfg.variants,) or diagonal.dtype != values.dtype:
        raise ValueError("diagonal must match variant count and input dtype")
    result = _bind(graph, diagonal[:, None], values, cfg, _Mode.WEIGHTED)
    return result[:, 0] if vector else result

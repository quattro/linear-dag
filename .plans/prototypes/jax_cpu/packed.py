"""Internal HiJAX/CPU experiment; not a supported linear_dag API.

Graph arrays are dynamic integer/bool PyTree leaves. Only dimensions and execution
policy are static. Packing is host-only; numerical execution uses no callbacks.
"""

from dataclasses import dataclass
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from jax.experimental.hijax import GradAccum, HiPrim, instantiate_zeros, linearize_from_jvp, ShapedArray, Zero


@dataclass(frozen=True)
class Config:
    samples: int
    variants: int
    max_state: int
    max_variants: int
    workers: int
    chunk: int

    def scratch_bytes(self, itemsize=4):
        return self.workers * (self.max_state + 2 * self.max_variants + self.samples + 2) * self.chunk * itemsize


def pack(blocks, *, workers=1, chunk=8):
    """Pack already-constructed, validated compressed graph blocks on the host.

    Copies each source array into one packed buffer. This experimental format
    uses int32 offsets and rejects overflow. Sample row order must agree across
    blocks; callers establish sample identity alignment before packing.
    """
    blocks = tuple(blocks)
    if not blocks or workers < 1 or chunk < 1:
        raise ValueError("nonempty blocks and positive workers/chunk required")
    keys = ("indptr", "indices", "data", "nonunique_indices", "variant_indices", "sample_indices", "flip")
    arrays = [[] for _ in keys]
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
        if np.any(idx <= sources):
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
    graph = (jnp.asarray(np.asarray(metadata, dtype=np.int32)), *(jnp.asarray(np.concatenate(a)) for a in arrays))
    return graph, Config(n_samples, vo, max_state, max_variants, min(workers, len(blocks)), chunk)


@cache
def _register():
    import _packed_cpu

    for name, capsule in _packed_cpu.registrations().items():
        jax.ffi.register_ffi_target(name, capsule, platform="cpu")


def _call(graph, a, b, cfg, mode):
    _register()
    k = (b if mode == 2 else a).shape[1]
    rows = cfg.variants if mode in (1, 3) else cfg.samples
    result_shape = (rows, 1 if mode == 3 else k)
    scratch_rows = cfg.max_state + 2 * cfg.max_variants + cfg.samples + 2
    target = "packed_f32" if a.dtype == jnp.float32 else "packed_f64"
    call = jax.ffi.ffi_call(
        target,
        (jax.ShapeDtypeStruct(result_shape, a.dtype), jax.ShapeDtypeStruct((cfg.workers, scratch_rows, k), a.dtype)),
        vmap_method="sequential",
    )
    result, _ = call(
        *graph, a, b, mode=mode, n_samples=cfg.samples, max_state=cfg.max_state, max_variants=cfg.max_variants
    )
    return result


def _execute(graph, a, b, cfg, mode):
    """Bound scratch width using a sequential XLA loop over RHS tiles."""
    k = (b if mode == 2 else a).shape[1]
    rows = cfg.variants if mode in (1, 3) else cfg.samples
    if k == 0:
        return jnp.zeros((rows, 1 if mode == 3 else 0), dtype=a.dtype)
    width = min(k, cfg.chunk)
    if k <= width:
        return _call(graph, a, b, cfg, mode)
    tiles = (k + width - 1) // width
    pad = tiles * width - k
    aa = a if mode == 2 else jnp.pad(a, ((0, 0), (0, pad)))
    bb = jnp.pad(b, ((0, 0), (0, pad))) if mode in (2, 3) else b
    result = jnp.zeros((rows, 1 if mode == 3 else tiles * width), a.dtype)

    def step(i, output):
        x = aa if mode == 2 else jax.lax.dynamic_slice_in_dim(aa, i * width, width, axis=1)
        y = jax.lax.dynamic_slice_in_dim(bb, i * width, width, axis=1) if mode in (2, 3) else bb
        tile = _call(graph, x, y, cfg, mode)
        return output + tile if mode == 3 else jax.lax.dynamic_update_slice(output, tile, (0, i * width))

    result = jax.lax.fori_loop(0, tiles, step, result)
    return result if mode == 3 else result[:, :k]


def _zero_graph(graph):
    return jax.tree.map(lambda x: Zero(jax.typeof(x).to_tangent_aval()), graph)


def _bind(graph, a, b, cfg, mode):
    return Operation(jax.tree.map(jax.typeof, graph), jax.typeof(a), jax.typeof(b), cfg, mode)(graph, a, b)


class Operation(HiPrim):
    """Forward, transpose, weighted GRM, or diagonal-adjoint operation."""

    def __init__(self, graph_types, a_type, b_type, cfg, mode):
        self.in_avals = (graph_types, a_type, b_type)
        k = (b_type if mode == 2 else a_type).shape[1]
        rows = cfg.variants if mode in (1, 3) else cfg.samples
        self.out_aval = ShapedArray((rows, 1 if mode == 3 else k), a_type.dtype)
        self.params = dict(cfg=cfg, mode=mode)
        super().__init__()

    def expand(self, graph, a, b):
        return _execute(graph, a, b, self.cfg, self.mode)

    def jvp(self, primals, tangents):
        graph, a, b = primals
        _, da, db = tangents
        result = self(graph, a, b)
        if self.mode in (0, 1):
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
        if self.mode in (0, 1):
            da = _bind(graph, g, dummy, self.cfg, 1 - self.mode)
            db = jnp.zeros_like(b)
        elif self.mode == 2:
            da = _bind(graph, b, g, self.cfg, 3)
            db = _bind(graph, a, g, self.cfg, 2)
        else:
            da = _bind(graph, g, b, self.cfg, 2)
            db = _bind(graph, g, a, self.cfg, 2)
        return _zero_graph(graph), da, db

    def batch(self, axis_data, args, dims):
        graph, a, b = args
        graph_dims, ad, bd = dims
        if any(d is not None for d in jax.tree.leaves(graph_dims)):
            raise ValueError("mapping over different graphs is not supported")
        if ad is None and bd is None:
            return self(graph, a, b), None
        # Collapse shared-graph/shared-diagonal RHS batches into matrix columns.
        rhs_axis = ad if self.mode in (0, 1) else bd
        if self.mode in (0, 1) or (self.mode == 2 and ad is None):
            rhs = a if self.mode in (0, 1) else b
            if rhs_axis is None:
                return self(graph, a, b), None
            rhs = jnp.moveaxis(rhs, rhs_axis, 0)
            batch, rows, width = rhs.shape
            flat = jnp.transpose(rhs, (1, 0, 2)).reshape(rows, batch * width)
            result = (
                _bind(graph, flat, b, self.cfg, self.mode)
                if self.mode in (0, 1)
                else _bind(graph, a, flat, self.cfg, 2)
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
        if self.mode in (0, 1):
            if isinstance(a, GradAccum):
                a.accum(_bind(graph, g, jnp.zeros((1, 1), g.dtype), self.cfg, 1 - self.mode))
        elif self.mode == 2:
            if isinstance(a, GradAccum):
                a.accum(_bind(graph, b, g, self.cfg, 3))
            if isinstance(b, GradAccum):
                b.accum(_bind(graph, a, g, self.cfg, 2))
        else:
            if isinstance(a, GradAccum):
                a.accum(_bind(graph, g, b, self.cfg, 2))
            if isinstance(b, GradAccum):
                b.accum(_bind(graph, g, a, self.cfg, 2))


def _matrix(value, rows):
    value = jnp.asarray(value)
    if value.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("float32 or float64 inputs required")
    if value.ndim not in (1, 2) or value.shape[0] != rows:
        raise ValueError("incompatible product input shape")
    return (value[:, None], True) if value.ndim == 1 else (value, False)


def product(graph, values, cfg, *, transpose=False):
    """Multiply by a fixed graph or its transpose; supports vectors and matrices."""
    values, vector = _matrix(values, cfg.samples if transpose else cfg.variants)
    result = _bind(graph, values, jnp.zeros((1, 1), values.dtype), cfg, int(transpose))
    return result[:, 0] if vector else result


def weighted(graph, diagonal, values, cfg):
    """Compute X diag(diagonal) X.T values, differentiable in both arguments."""
    values, vector = _matrix(values, cfg.samples)
    diagonal = jnp.asarray(diagonal)
    if diagonal.shape != (cfg.variants,) or diagonal.dtype != values.dtype:
        raise ValueError("diagonal must match variant count and input dtype")
    result = _bind(graph, diagonal[:, None], values, cfg, 2)
    return result[:, 0] if vector else result

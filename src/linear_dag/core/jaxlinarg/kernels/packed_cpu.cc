// Packed ragged CPU execution. Graph contents are validated at host ingress.
#include "xla/ffi/api/ffi.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
namespace ffi = xla::ffi;

namespace {
// Keep these codes and metadata columns in sync with ../_packed.py. The graph
// payload remains block-local; offsets locate each block in the packed arrays.
constexpr int64_t kForward = 0;
constexpr int64_t kReverse = 1;
constexpr int64_t kWeighted = 2;
constexpr int64_t kDiagonalAdjoint = 3;
enum MetadataColumn {
  kPtrOffset,
  kNodeOffset,
  kEdgeOffset,
  kVariantOffset,
  kNodes,
  kVariants,
  kStateRows,
  kKeepBoundary,
  kMetadataWidth
};

template <typename T> void Add(int64_t k, T scale, const T *src, T *dst) {
  for (int64_t c = 0; c < k; ++c)
    dst[c] += scale * src[c];
}

template <ffi::DataType dtype>
ffi::Future
Execute(ffi::ThreadPool pool, ffi::BufferR2<ffi::S32> metadata,
        ffi::BufferR1<ffi::S32> ptr, ffi::BufferR1<ffi::S32> indices,
        ffi::BufferR1<ffi::S32> weights, ffi::BufferR1<ffi::S32> mapping,
        ffi::BufferR1<ffi::S32> variants, ffi::BufferR1<ffi::S32> samples,
        ffi::BufferR1<ffi::PRED> flips, ffi::BufferR1<ffi::S32> counts,
        ffi::BufferR2<dtype> a, ffi::BufferR2<dtype> b,
        ffi::ResultBufferR2<dtype> output, ffi::ResultBufferR3<dtype> scratch,
        int64_t mode, int64_t n_samples, int64_t max_state,
        int64_t max_variants, int64_t center) {
  using T = ffi::NativeType<dtype>;
  ffi::Promise promise;
  ffi::Future future(promise);
  const int64_t workers = scratch->dimensions()[0];
  const int64_t blocks = metadata.dimensions()[0];
  const int64_t k = (mode == kWeighted ? b : a).dimensions()[1];
  const int64_t rows = max_state + 2 * max_variants + n_samples + 2;
  if (mode < kForward || mode > kDiagonalAdjoint || workers < 1 ||
      workers > blocks || metadata.dimensions()[1] != kMetadataWidth ||
      scratch->dimensions()[1] != rows || scratch->dimensions()[2] != k) {
    promise.SetError(
        ffi::Error::InvalidArgument("invalid packed execution shape"));
    return future;
  }
  const int64_t n_variants = variants.dimensions()[0];
  const int64_t result_rows =
      (mode == kReverse || mode == kDiagonalAdjoint) ? n_variants : n_samples;
  if (n_samples < 0 || max_state < 0 || max_variants < 0 ||
      counts.dimensions()[0] != n_variants ||
      flips.dimensions()[0] != n_variants ||
      samples.dimensions()[0] != blocks * n_samples ||
      weights.dimensions()[0] != indices.dimensions()[0] ||
      output->dimensions()[0] != result_rows ||
      output->dimensions()[1] != (mode == kDiagonalAdjoint ? 1 : k) ||
      a.dimensions()[0] !=
          ((mode == kForward || mode == kWeighted) ? n_variants : n_samples) ||
      (mode == kWeighted &&
       (a.dimensions()[1] != 1 || b.dimensions()[0] != n_samples)) ||
      (mode == kDiagonalAdjoint &&
       (b.dimensions()[0] != n_samples || b.dimensions()[1] != k)) ||
      (center && n_samples == 0)) {
    promise.SetError(
        ffi::Error::InvalidArgument("inconsistent packed buffer dimensions"));
    return future;
  }
  // Check per-block bounds here, but do not rescan all edges on every product.
  // Graph contents/topological order are validated once by host construction.
  const int32_t *meta = metadata.typed_data();
  for (int64_t block = 0; block < blocks; ++block) {
    const int32_t *m = meta + kMetadataWidth * block;
    if (m[kPtrOffset] < 0 || m[kNodeOffset] < 0 || m[kEdgeOffset] < 0 ||
        m[kVariantOffset] < 0 || m[kNodes] < 0 || m[kVariants] < 0 ||
        m[kStateRows] < 0 || m[kKeepBoundary] < 0 ||
        int64_t(m[kPtrOffset]) + m[kNodes] + 1 > ptr.dimensions()[0] ||
        int64_t(m[kNodeOffset]) + m[kNodes] > mapping.dimensions()[0] ||
        int64_t(m[kVariantOffset]) + m[kVariants] > n_variants ||
        m[kVariants] > max_variants || m[kStateRows] > max_state ||
        m[kKeepBoundary] > m[kNodes] || ptr.typed_data()[m[kPtrOffset]] != 0 ||
        int64_t(m[kEdgeOffset]) +
                ptr.typed_data()[int64_t(m[kPtrOffset]) + m[kNodes]] >
            indices.dimensions()[0]) {
      promise.SetError(
          ffi::Error::InvalidArgument("invalid packed block bounds"));
      return future;
    }
  }
  const int32_t *ptrs = ptr.typed_data();
  const int32_t *edges = indices.typed_data();
  const int32_t *vals = weights.typed_data();
  const int32_t *maps = mapping.typed_data();
  const int32_t *vars = variants.typed_data();
  const int32_t *samps = samples.typed_data();
  const bool *flip = flips.typed_data();
  const int32_t *ac = counts.typed_data();
  const T *av = a.typed_data();
  const T *bv = b.typed_data();
  T *out = output->typed_data();
  T *work = scratch->typed_data();
  auto remaining = std::make_shared<std::atomic<int64_t>>(workers);
  for (int64_t worker = 0; worker < workers; ++worker) {
    pool.Schedule([=]() mutable {
      // XLA owns these buffers through completion of the returned Future.
      // Each worker has disjoint state/scores/accumulator storage. Two trailing
      // RHS rows are reserved; the first holds flip/centering totals.
      T *state = work + worker * rows * k;
      T *scores = state + max_state * k;
      T *scores2 = scores + max_variants * k;
      T *accumulator = scores2 + max_variants * k;
      T *totals = accumulator + n_samples * k;
      std::fill(accumulator, accumulator + n_samples * k, T(0));
      // Fixed round-robin assignment and fixed final worker reduction order.
      for (int64_t block = worker; block < blocks; block += workers) {
        const int32_t *m = meta + block * kMetadataWidth;
        const int64_t po = m[kPtrOffset], no = m[kNodeOffset],
                      eo = m[kEdgeOffset], vo = m[kVariantOffset];
        const int64_t nodes = m[kNodes], nv = m[kVariants], ns = m[kStateRows],
                      keep = m[kKeepBoundary];
        const int32_t *map = maps + no;
        const int32_t *vp = vars + vo;
        const int32_t *sp = samps + block * n_samples;
        auto reverse = [&](const T *input, T *result) {
          std::fill(state, state + ns * k, T(0));
          std::fill(totals, totals + k, T(0));
          for (int64_t s = 0; s < n_samples; ++s) {
            std::copy(input + s * k, input + (s + 1) * k,
                      state + int64_t(map[sp[s]]) * k);
            Add(k, T(1), input + s * k, totals);
          }
          for (int64_t node = nodes - 1; node >= 0; --node) {
            const int64_t start = ptrs[po + node], stop = ptrs[po + node + 1];
            T *dst = state + int64_t(map[node]) * k;
            // Reused workspace slots must be cleared before reverse
            // accumulation; retained sample slots (node >= keep) preserve the
            // seeded RHS.
            if (start != stop && node < keep)
              std::fill(dst, dst + k, T(0));
            for (int64_t e = stop - 1; e >= start; --e)
              Add(k, T(vals[eo + e]), state + int64_t(map[edges[eo + e]]) * k,
                  dst);
          }
          for (int64_t v = 0; v < nv; ++v) {
            const T *src = state + int64_t(map[vp[v]]) * k;
            const T frequency = center ? T(ac[vo + v]) / T(n_samples) : T(0);
            for (int64_t c = 0; c < k; ++c)
              result[v * k + c] = (flip[vo + v] ? totals[c] - src[c] : src[c]) -
                                  frequency * totals[c];
          }
        };
        auto forward = [&](const T *input) {
          std::fill(state, state + ns * k, T(0));
          std::fill(totals, totals + k, T(0));
          for (int64_t v = 0; v < nv; ++v) {
            Add(k, flip[vo + v] ? T(-1) : T(1), input + v * k,
                state + int64_t(map[vp[v]]) * k);
            const T correction = (flip[vo + v] ? T(1) : T(0)) -
                                 (center ? T(ac[vo + v]) / T(n_samples) : T(0));
            if (correction != T(0))
              Add(k, correction, input + v * k, totals);
          }
          for (int64_t node = 0; node < nodes; ++node) {
            const int64_t start = ptrs[po + node], stop = ptrs[po + node + 1];
            T *src = state + int64_t(map[node]) * k;
            for (int64_t e = start; e < stop; ++e)
              Add(k, T(vals[eo + e]), src,
                  state + int64_t(map[edges[eo + e]]) * k);
            // Forward traversal releases a reused slot only after its outgoing
            // edges have consumed the value.
            if (start != stop && node < keep)
              std::fill(src, src + k, T(0));
          }
          for (int64_t s = 0; s < n_samples; ++s) {
            Add(k, T(1), state + int64_t(map[sp[s]]) * k, accumulator + s * k);
            Add(k, T(1), totals, accumulator + s * k);
          }
        };
        if (mode == kForward)
          forward(av + vo * k);
        if (mode == kReverse)
          reverse(av, out + vo * k);
        if (mode == kWeighted) {
          reverse(bv, scores);
          for (int64_t v = 0; v < nv; ++v)
            for (int64_t c = 0; c < k; ++c)
              scores[v * k + c] *= av[vo + v];
          forward(scores);
        }
        if (mode == kDiagonalAdjoint) {
          reverse(av, scores);
          reverse(bv, scores2);
          for (int64_t v = 0; v < nv; ++v) {
            T sum = 0;
            for (int64_t c = 0; c < k; ++c)
              sum += scores[v * k + c] * scores2[v * k + c];
            out[vo + v] = sum;
          }
        }
      }
      // Reverse/diagonal-adjoint blocks write disjoint variant ranges directly.
      // Forward/GRM products reduce private accumulators in fixed worker order.
      // Acquire/release publishes every worker's writes before the final
      // reducer signals completion; no task blocks waiting for another pool
      // task.
      if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (mode == kForward || mode == kWeighted) {
          std::fill(out, out + n_samples * k, T(0));
          for (int64_t w = 0; w < workers; ++w)
            Add(n_samples * k, T(1),
                work + (w * rows + max_state + 2 * max_variants) * k, out);
        }
        promise.SetAvailable();
      }
    });
  }
  return future;
}

} // namespace

#define BIND(T)                                                                \
  ffi::Ffi::Bind()                                                             \
      .Ctx<ffi::ThreadPool>()                                                  \
      .Arg<ffi::BufferR2<ffi::S32>>()                                          \
      .Arg<ffi::BufferR1<ffi::S32>>()                                          \
      .Arg<ffi::BufferR1<ffi::S32>>()                                          \
      .Arg<ffi::BufferR1<ffi::S32>>()                                          \
      .Arg<ffi::BufferR1<ffi::S32>>()                                          \
      .Arg<ffi::BufferR1<ffi::S32>>()                                          \
      .Arg<ffi::BufferR1<ffi::S32>>()                                          \
      .Arg<ffi::BufferR1<ffi::PRED>>()                                         \
      .Arg<ffi::BufferR1<ffi::S32>>()                                          \
      .Arg<ffi::BufferR2<T>>()                                                 \
      .Arg<ffi::BufferR2<T>>()                                                 \
      .Ret<ffi::BufferR2<T>>()                                                 \
      .Ret<ffi::BufferR3<T>>()                                                 \
      .Attr<int64_t>("mode")                                                   \
      .Attr<int64_t>("n_samples")                                              \
      .Attr<int64_t>("max_state")                                              \
      .Attr<int64_t>("max_variants")                                           \
      .Attr<int64_t>("center")
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_packed_f32, Execute<ffi::F32>,
                              BIND(ffi::F32));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_packed_f64, Execute<ffi::F64>,
                              BIND(ffi::F64));

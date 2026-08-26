// pattern: Imperative Shell

#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#ifdef LINEAR_DAG_HAVE_CBLAS
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#include "xla/ffi/api/ffi.h"

#ifndef LINEAR_DAG_FFI_CPU_BLAS_BACKEND
#define LINEAR_DAG_FFI_CPU_BLAS_BACKEND "unknown"
#endif

#ifndef LINEAR_DAG_FFI_CPU_NATIVE_TUNING
#define LINEAR_DAG_FFI_CPU_NATIVE_TUNING 0
#endif

namespace ffi = xla::ffi;

template <typename T>
void AxpyScalar(int64_t n, T alpha, const T* x, T* y) {
  for (int64_t i = 0; i < n; ++i) {
    y[i] += x[i] * alpha;
  }
}

#ifdef LINEAR_DAG_HAVE_CBLAS
void AxpyBlas(int64_t n, float alpha, const float* x, float* y) {
  cblas_saxpy(static_cast<int>(n), alpha, x, 1, y, 1);
}

void AxpyBlas(int64_t n, double alpha, const double* x, double* y) {
  cblas_daxpy(static_cast<int>(n), alpha, x, 1, y, 1);
}
#endif

template <typename T>
void Axpy(int64_t n, T alpha, const T* x, T* y) {
#ifdef LINEAR_DAG_HAVE_CBLAS
  // Each edge update is a row-contiguous vector update over RHS columns:
  // dst_row += weight * src_row. That is exactly BLAS AXPY with unit strides.
  // CBLAS uses int lengths, so very wide RHS buffers keep the scalar fallback.
  if (n <= std::numeric_limits<int>::max()) {
    AxpyBlas(n, alpha, x, y);
    return;
  }
#endif
  AxpyScalar(n, alpha, x, y);
}

template <typename T>
void ZeroRow(int64_t n, T* row) {
  std::memset(row, 0, static_cast<size_t>(n) * sizeof(T));
}

template <ffi::DataType dtype>
void CopyBuffer(ffi::BufferR2<dtype> b, ffi::ResultBufferR2<dtype> out) {
  const int64_t element_count = b.dimensions()[0] * b.dimensions()[1];
  if (b.typed_data() != out->typed_data()) {
    std::copy(b.typed_data(), b.typed_data() + element_count, out->typed_data());
  }
}

template <ffi::DataType dtype, ffi::DataType weight_dtype>
void SolveForwardInPlace(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<weight_dtype> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    int64_t n_cols,
    ffi::NativeType<dtype>* values,
    int64_t min_index_to_keep) {
  using T = ffi::NativeType<dtype>;
  const int64_t n_nodes = indptr.dimensions()[0] - 1;
  const int32_t* indptr_data = indptr.typed_data();
  const int32_t* indices_data = indices.typed_data();
  const ffi::NativeType<weight_dtype>* weights = data.typed_data();
  const int32_t* nonunique = nonunique_indices.typed_data();

  int64_t edge = 0;
  for (int64_t node = 0; node < n_nodes; ++node) {
    const int64_t edge_stop = indptr_data[node + 1];
    if (edge == edge_stop) {
      continue;
    }
    T* src_row = values + static_cast<int64_t>(nonunique[node]) * n_cols;
    while (edge < edge_stop) {
      T* dst_row = values + static_cast<int64_t>(nonunique[indices_data[edge]]) * n_cols;
      Axpy(n_cols, static_cast<T>(weights[edge]), src_row, dst_row);
      ++edge;
    }
    if (node < min_index_to_keep) {
      ZeroRow(n_cols, src_row);
    }
  }
}

template <ffi::DataType dtype>
ffi::Error SolveForwardCompressed(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<dtype> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<dtype> b,
    ffi::ResultBufferR2<dtype> out,
    int64_t min_index_to_keep) {
  // This handler is intentionally a hot numerical kernel, not a metadata
  // validator. Graph construction is responsible for producing consistent
  // CSC arrays and nonunique row mappings before values reach this FFI call.
  CopyBuffer(b, out);
  SolveForwardInPlace<dtype, dtype>(
      indptr, indices, data, nonunique_indices, b.dimensions()[1],
      out->typed_data(), min_index_to_keep);
  return ffi::Error::Success();
}

template <ffi::DataType dtype, ffi::DataType weight_dtype>
void SolveBackwardInPlace(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<weight_dtype> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    int64_t n_cols,
    ffi::NativeType<dtype>* values,
    int64_t min_index_to_keep) {
  using T = ffi::NativeType<dtype>;
  const int64_t n_nodes = indptr.dimensions()[0] - 1;
  const int32_t* indptr_data = indptr.typed_data();
  const int32_t* indices_data = indices.typed_data();
  const ffi::NativeType<weight_dtype>* weights = data.typed_data();
  const int32_t* nonunique = nonunique_indices.typed_data();

  int64_t edge = indptr_data[n_nodes];
  for (int64_t node = n_nodes - 1; node >= 0; --node) {
    const int64_t edge_start = indptr_data[node];
    if (edge == edge_start) {
      continue;
    }
    T* dst_row = values + static_cast<int64_t>(nonunique[node]) * n_cols;
    if (node < min_index_to_keep) {
      ZeroRow(n_cols, dst_row);
    }
    while (edge > edge_start) {
      --edge;
      T* src_row = values + static_cast<int64_t>(nonunique[indices_data[edge]]) * n_cols;
      Axpy(n_cols, static_cast<T>(weights[edge]), src_row, dst_row);
    }
  }
}

template <ffi::DataType dtype>
ffi::Error SolveBackwardCompressed(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<dtype> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<dtype> b,
    ffi::ResultBufferR2<dtype> out,
    int64_t min_index_to_keep) {
  // Match the Cython solve loop shape and trust prevalidated graph metadata.
  CopyBuffer(b, out);
  SolveBackwardInPlace<dtype, dtype>(
      indptr, indices, data, nonunique_indices, b.dimensions()[1],
      out->typed_data(), min_index_to_keep);
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error MatmatCompressed(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::S32> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR1<ffi::S32> variant_indices,
    ffi::BufferR1<ffi::S32> sample_indices,
    ffi::BufferR1<ffi::PRED> flip,
    ffi::BufferR2<dtype> input,
    ffi::ResultBufferR2<dtype> out,
    ffi::ResultBufferR2<dtype> workspace,
    int64_t min_index_to_keep) {
  using T = ffi::NativeType<dtype>;
  const int64_t n_variants = input.dimensions()[0];
  const int64_t n_cols = input.dimensions()[1];
  const int64_t n_samples = sample_indices.dimensions()[0];
  const int64_t n_nodes = nonunique_indices.dimensions()[0];
  const int64_t workspace_rows = workspace->dimensions()[0];
  const int64_t n_state = workspace_rows - 1;
  if (variant_indices.dimensions()[0] != n_variants ||
      flip.dimensions()[0] != n_variants ||
      out->dimensions()[0] != n_samples ||
      out->dimensions()[1] != n_cols || workspace_rows < 1 ||
      workspace->dimensions()[1] != n_cols) {
    return ffi::Error::InvalidArgument("inconsistent fused matmat shapes");
  }

  const T* input_data = input.typed_data();
  const int32_t* variant_nodes = variant_indices.typed_data();
  const int32_t* sample_nodes = sample_indices.typed_data();
  const int32_t* nonunique = nonunique_indices.typed_data();
  const bool* flips = flip.typed_data();
  T* state = workspace->typed_data();
  T* flip_totals = state + n_state * n_cols;
  std::fill(state, flip_totals + n_cols, static_cast<T>(0));

  for (int64_t variant = 0; variant < n_variants; ++variant) {
    const int64_t node = variant_nodes[variant];
    if (node < 0 || node >= n_nodes) {
      return ffi::Error::InvalidArgument(
          "variant_indices contains an out-of-range node index");
    }
    const int64_t state_index = nonunique[node];
    if (state_index < 0 || state_index >= n_state) {
      return ffi::Error::InvalidArgument(
          "variant_indices maps to an out-of-range state index");
    }
    const T* input_row = input_data + variant * n_cols;
    T* state_row = state + state_index * n_cols;
    if (flips[variant]) {
      Axpy(n_cols, static_cast<T>(-1), input_row, state_row);
      Axpy(n_cols, static_cast<T>(1), input_row, flip_totals);
    } else {
      Axpy(n_cols, static_cast<T>(1), input_row, state_row);
    }
  }

  SolveForwardInPlace<dtype, ffi::S32>(
      indptr, indices, data, nonunique_indices, n_cols, state,
      min_index_to_keep);

  T* result = out->typed_data();
  for (int64_t sample = 0; sample < n_samples; ++sample) {
    const int64_t node = sample_nodes[sample];
    if (node < 0 || node >= n_nodes) {
      return ffi::Error::InvalidArgument(
          "sample_indices contains an out-of-range node index");
    }
    const int64_t state_index = nonunique[node];
    if (state_index < 0 || state_index >= n_state) {
      return ffi::Error::InvalidArgument(
          "sample_indices maps to an out-of-range state index");
    }
    const T* state_row = state + state_index * n_cols;
    T* result_row = result + sample * n_cols;
    std::copy(state_row, state_row + n_cols, result_row);
    Axpy(n_cols, static_cast<T>(1), flip_totals, result_row);
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error RmatmatCompressed(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::S32> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR1<ffi::S32> variant_indices,
    ffi::BufferR1<ffi::S32> sample_indices,
    ffi::BufferR1<ffi::PRED> flip,
    ffi::BufferR2<dtype> input,
    ffi::ResultBufferR2<dtype> out,
    ffi::ResultBufferR2<dtype> workspace,
    int64_t min_index_to_keep) {
  using T = ffi::NativeType<dtype>;
  const int64_t n_samples = input.dimensions()[0];
  const int64_t n_cols = input.dimensions()[1];
  const int64_t n_variants = variant_indices.dimensions()[0];
  const int64_t n_nodes = nonunique_indices.dimensions()[0];
  const int64_t workspace_rows = workspace->dimensions()[0];
  const int64_t n_state = workspace_rows - 1;
  if (sample_indices.dimensions()[0] != n_samples ||
      flip.dimensions()[0] != n_variants ||
      out->dimensions()[0] != n_variants ||
      out->dimensions()[1] != n_cols || workspace_rows < 1 ||
      workspace->dimensions()[1] != n_cols) {
    return ffi::Error::InvalidArgument("inconsistent fused rmatmat shapes");
  }

  const T* input_data = input.typed_data();
  const int32_t* variant_nodes = variant_indices.typed_data();
  const int32_t* sample_nodes = sample_indices.typed_data();
  const int32_t* nonunique = nonunique_indices.typed_data();
  const bool* flips = flip.typed_data();
  T* state = workspace->typed_data();
  T* totals = state + n_state * n_cols;
  std::fill(state, totals + n_cols, static_cast<T>(0));

  for (int64_t sample = 0; sample < n_samples; ++sample) {
    const int64_t node = sample_nodes[sample];
    if (node < 0 || node >= n_nodes) {
      return ffi::Error::InvalidArgument(
          "sample_indices contains an out-of-range node index");
    }
    const int64_t state_index = nonunique[node];
    if (state_index < 0 || state_index >= n_state) {
      return ffi::Error::InvalidArgument(
          "sample_indices maps to an out-of-range state index");
    }
    const T* input_row = input_data + sample * n_cols;
    T* state_row = state + state_index * n_cols;
    std::copy(input_row, input_row + n_cols, state_row);
    Axpy(n_cols, static_cast<T>(1), input_row, totals);
  }

  SolveBackwardInPlace<dtype, ffi::S32>(
      indptr, indices, data, nonunique_indices, n_cols, state,
      min_index_to_keep);

  T* result = out->typed_data();
  for (int64_t variant = 0; variant < n_variants; ++variant) {
    const int64_t node = variant_nodes[variant];
    if (node < 0 || node >= n_nodes) {
      return ffi::Error::InvalidArgument(
          "variant_indices contains an out-of-range node index");
    }
    const int64_t state_index = nonunique[node];
    if (state_index < 0 || state_index >= n_state) {
      return ffi::Error::InvalidArgument(
          "variant_indices maps to an out-of-range state index");
    }
    const T* state_row = state + state_index * n_cols;
    T* result_row = result + variant * n_cols;
    if (flips[variant]) {
      for (int64_t column = 0; column < n_cols; ++column) {
        result_row[column] = totals[column] - state_row[column];
      }
    } else {
      std::copy(state_row, state_row + n_cols, result_row);
    }
  }
  return ffi::Error::Success();
}

static ffi::Error SolveForwardF32(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::F32> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<ffi::F32> b,
    ffi::ResultBufferR2<ffi::F32> out,
    int64_t min_index_to_keep) {
  return SolveForwardCompressed<ffi::F32>(
      indptr, indices, data, nonunique_indices, b, out, min_index_to_keep);
}

static ffi::Error SolveBackwardF32(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::F32> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<ffi::F32> b,
    ffi::ResultBufferR2<ffi::F32> out,
    int64_t min_index_to_keep) {
  return SolveBackwardCompressed<ffi::F32>(
      indptr, indices, data, nonunique_indices, b, out, min_index_to_keep);
}

static ffi::Error SolveForwardF64(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::F64> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<ffi::F64> b,
    ffi::ResultBufferR2<ffi::F64> out,
    int64_t min_index_to_keep) {
  return SolveForwardCompressed<ffi::F64>(
      indptr, indices, data, nonunique_indices, b, out, min_index_to_keep);
}

static ffi::Error SolveBackwardF64(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::F64> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<ffi::F64> b,
    ffi::ResultBufferR2<ffi::F64> out,
    int64_t min_index_to_keep) {
  return SolveBackwardCompressed<ffi::F64>(
      indptr, indices, data, nonunique_indices, b, out, min_index_to_keep);
}

static ffi::Error MatmatF32(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::S32> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR1<ffi::S32> variant_indices,
    ffi::BufferR1<ffi::S32> sample_indices,
    ffi::BufferR1<ffi::PRED> flip,
    ffi::BufferR2<ffi::F32> input,
    ffi::ResultBufferR2<ffi::F32> out,
    ffi::ResultBufferR2<ffi::F32> workspace,
    int64_t min_index_to_keep) {
  return MatmatCompressed<ffi::F32>(
      indptr, indices, data, nonunique_indices, variant_indices,
      sample_indices, flip, input, out, workspace, min_index_to_keep);
}

static ffi::Error MatmatF64(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::S32> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR1<ffi::S32> variant_indices,
    ffi::BufferR1<ffi::S32> sample_indices,
    ffi::BufferR1<ffi::PRED> flip,
    ffi::BufferR2<ffi::F64> input,
    ffi::ResultBufferR2<ffi::F64> out,
    ffi::ResultBufferR2<ffi::F64> workspace,
    int64_t min_index_to_keep) {
  return MatmatCompressed<ffi::F64>(
      indptr, indices, data, nonunique_indices, variant_indices,
      sample_indices, flip, input, out, workspace, min_index_to_keep);
}

static ffi::Error RmatmatF32(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::S32> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR1<ffi::S32> variant_indices,
    ffi::BufferR1<ffi::S32> sample_indices,
    ffi::BufferR1<ffi::PRED> flip,
    ffi::BufferR2<ffi::F32> input,
    ffi::ResultBufferR2<ffi::F32> out,
    ffi::ResultBufferR2<ffi::F32> workspace,
    int64_t min_index_to_keep) {
  return RmatmatCompressed<ffi::F32>(
      indptr, indices, data, nonunique_indices, variant_indices,
      sample_indices, flip, input, out, workspace, min_index_to_keep);
}

static ffi::Error RmatmatF64(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::S32> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR1<ffi::S32> variant_indices,
    ffi::BufferR1<ffi::S32> sample_indices,
    ffi::BufferR1<ffi::PRED> flip,
    ffi::BufferR2<ffi::F64> input,
    ffi::ResultBufferR2<ffi::F64> out,
    ffi::ResultBufferR2<ffi::F64> workspace,
    int64_t min_index_to_keep) {
  return RmatmatCompressed<ffi::F64>(
      indptr, indices, data, nonunique_indices, variant_indices,
      sample_indices, flip, input, out, workspace, min_index_to_keep);
}

#define LINEAR_DAG_BINDING(dtype)                                               \
  ffi::Ffi::Bind()                                                              \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<dtype>>()                                              \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR2<dtype>>()                                              \
      .Ret<ffi::BufferR2<dtype>>()                                              \
      .Attr<int64_t>("min_index_to_keep")

#define LINEAR_DAG_PRODUCT_BINDING(dtype)                                       \
  ffi::Ffi::Bind()                                                              \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<ffi::PRED>>()                                          \
      .Arg<ffi::BufferR2<dtype>>()                                              \
      .Ret<ffi::BufferR2<dtype>>()                                              \
      .Ret<ffi::BufferR2<dtype>>()                                              \
      .Attr<int64_t>("min_index_to_keep")

XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_solve_forward_f32, SolveForwardF32,
                              LINEAR_DAG_BINDING(ffi::F32));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_solve_backward_f32, SolveBackwardF32,
                              LINEAR_DAG_BINDING(ffi::F32));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_solve_forward_f64, SolveForwardF64,
                              LINEAR_DAG_BINDING(ffi::F64));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_solve_backward_f64, SolveBackwardF64,
                              LINEAR_DAG_BINDING(ffi::F64));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_matmat_f32, MatmatF32,
                              LINEAR_DAG_PRODUCT_BINDING(ffi::F32));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_matmat_f64, MatmatF64,
                              LINEAR_DAG_PRODUCT_BINDING(ffi::F64));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_rmatmat_f32, RmatmatF32,
                              LINEAR_DAG_PRODUCT_BINDING(ffi::F32));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_rmatmat_f64, RmatmatF64,
                              LINEAR_DAG_PRODUCT_BINDING(ffi::F64));

#undef LINEAR_DAG_BINDING
#undef LINEAR_DAG_PRODUCT_BINDING

static int AddRegistration(PyObject* dict, const char* name, void* ptr) {
  PyObject* capsule = PyCapsule_New(ptr, nullptr, nullptr);
  if (capsule == nullptr) {
    return -1;
  }
  const int status = PyDict_SetItemString(dict, name, capsule);
  Py_DECREF(capsule);
  return status;
}

static PyObject* Registrations(PyObject*, PyObject*) {
  PyObject* dict = PyDict_New();
  if (dict == nullptr) {
    return nullptr;
  }
  if (AddRegistration(dict, "linear_dag_jaxlinarg_solve_forward_f32",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_solve_forward_f32)) < 0 ||
      AddRegistration(dict, "linear_dag_jaxlinarg_solve_backward_f32",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_solve_backward_f32)) < 0 ||
      AddRegistration(dict, "linear_dag_jaxlinarg_solve_forward_f64",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_solve_forward_f64)) < 0 ||
      AddRegistration(dict, "linear_dag_jaxlinarg_solve_backward_f64",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_solve_backward_f64)) < 0 ||
      AddRegistration(dict, "linear_dag_jaxlinarg_matmat_f32",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_matmat_f32)) < 0 ||
      AddRegistration(dict, "linear_dag_jaxlinarg_matmat_f64",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_matmat_f64)) < 0 ||
      AddRegistration(dict, "linear_dag_jaxlinarg_rmatmat_f32",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_rmatmat_f32)) < 0 ||
      AddRegistration(dict, "linear_dag_jaxlinarg_rmatmat_f64",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_rmatmat_f64)) < 0) {
    Py_DECREF(dict);
    return nullptr;
  }
  return dict;
}

static PyObject* BlasEnabled(PyObject*, PyObject*) {
#ifdef LINEAR_DAG_HAVE_CBLAS
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

static PyObject* BlasBackend(PyObject*, PyObject*) {
  return PyUnicode_FromString(LINEAR_DAG_FFI_CPU_BLAS_BACKEND);
}

static PyObject* NativeTuningEnabled(PyObject*, PyObject*) {
#if LINEAR_DAG_FFI_CPU_NATIVE_TUNING
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

static PyMethodDef Methods[] = {
    {"registrations", Registrations, METH_NOARGS, "Return CPU FFI target registrations."},
    {"blas_enabled", BlasEnabled, METH_NOARGS, "Return whether the CPU FFI extension was built with CBLAS."},
    {"blas_backend", BlasBackend, METH_NOARGS, "Return the CPU FFI BLAS backend selected at build time."},
    {"native_tuning_enabled", NativeTuningEnabled, METH_NOARGS, "Return whether the CPU FFI extension was built with native CPU tuning."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_ffi_cpu_impl",
    nullptr,
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit__ffi_cpu_impl(void) {
  return PyModule_Create(&Module);
}

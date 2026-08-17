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

constexpr int32_t kPackedDescriptorVersion = 1;
enum PackedDescriptorColumn : int64_t {
  kVersion = 0,
  kValid = 1,
  kNodeStart = 2,
  kNodeLength = 3,
  kIndptrStart = 4,
  kIndptrLength = 5,
  kEdgeStart = 6,
  kEdgeLength = 7,
  kCompressedStart = 8,
  kCompressedLength = 9,
  kMinIndexToKeep = 10,
  kPackedDescriptorColumnCount = 11,
};

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
  using T = ffi::NativeType<dtype>;
  CopyBuffer(b, out);

  const int64_t n_nodes = indptr.dimensions()[0] - 1;
  const int64_t n_cols = b.dimensions()[1];
  T* values = out->typed_data();
  const int32_t* indptr_data = indptr.typed_data();
  const int32_t* indices_data = indices.typed_data();
  const T* weights = data.typed_data();
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
      Axpy(n_cols, weights[edge], src_row, dst_row);
      ++edge;
    }
    if (node < min_index_to_keep) {
      ZeroRow(n_cols, src_row);
    }
  }
  return ffi::Error::Success();
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
  using T = ffi::NativeType<dtype>;
  CopyBuffer(b, out);

  const int64_t n_nodes = indptr.dimensions()[0] - 1;
  const int64_t n_cols = b.dimensions()[1];
  T* values = out->typed_data();
  const int32_t* indptr_data = indptr.typed_data();
  const int32_t* indices_data = indices.typed_data();
  const T* weights = data.typed_data();
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
      Axpy(n_cols, weights[edge], src_row, dst_row);
    }
  }
  return ffi::Error::Success();
}

bool SpanInRange(int64_t start, int64_t length, int64_t capacity) {
  return start >= 0 && length >= 0 && start <= capacity && length <= capacity - start;
}

template <ffi::DataType dtype>
ffi::Error ValidatePackedSolve(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<dtype> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<ffi::S32> descriptors,
    ffi::BufferR2<dtype> b) {
  if (descriptors.dimensions()[1] != kPackedDescriptorColumnCount) {
    return ffi::Error::InvalidArgument(
        "packed descriptor buffer has an unsupported column count");
  }
  if (indices.dimensions()[0] != data.dimensions()[0]) {
    return ffi::Error::InvalidArgument(
        "packed indices and data buffers must have equal lengths");
  }

  const int64_t descriptor_count = descriptors.dimensions()[0];
  const int32_t* rows = descriptors.typed_data();
  const int32_t* indptr_data = indptr.typed_data();
  const int32_t* indices_data = indices.typed_data();
  const int32_t* nonunique = nonunique_indices.typed_data();
  int64_t expected_node_start = 0;
  int64_t expected_indptr_start = 0;
  int64_t expected_edge_start = 0;
  int64_t expected_compressed_start = 0;
  bool saw_padding = false;

  for (int64_t slot = 0; slot < descriptor_count; ++slot) {
    const int32_t* row = rows + slot * kPackedDescriptorColumnCount;
    if (row[kVersion] != kPackedDescriptorVersion) {
      return ffi::Error::InvalidArgument(
          "unsupported packed descriptor version");
    }
    if (row[kValid] != 0 && row[kValid] != 1) {
      return ffi::Error::InvalidArgument(
          "packed descriptor valid flag must be zero or one");
    }
    if (row[kValid] == 0) {
      saw_padding = true;
      for (int64_t column = kNodeStart;
           column < kPackedDescriptorColumnCount; ++column) {
        if (row[column] != 0) {
          return ffi::Error::InvalidArgument(
              "padded packed descriptor fields must be inert zeros");
        }
      }
      continue;
    }
    if (saw_padding) {
      return ffi::Error::InvalidArgument(
          "valid packed descriptors must precede padded descriptors");
    }

    const int64_t node_start = row[kNodeStart];
    const int64_t node_length = row[kNodeLength];
    const int64_t indptr_start = row[kIndptrStart];
    const int64_t indptr_length = row[kIndptrLength];
    const int64_t edge_start = row[kEdgeStart];
    const int64_t edge_length = row[kEdgeLength];
    const int64_t compressed_start = row[kCompressedStart];
    const int64_t compressed_length = row[kCompressedLength];
    const int64_t min_index_to_keep = row[kMinIndexToKeep];

    if (!SpanInRange(node_start, node_length,
                     nonunique_indices.dimensions()[0]) ||
        node_start != expected_node_start || node_length == 0) {
      return ffi::Error::InvalidArgument(
          "packed descriptor node span is out of range or nonmonotonic");
    }
    if (!SpanInRange(indptr_start, indptr_length,
                     indptr.dimensions()[0]) ||
        indptr_start != expected_indptr_start) {
      return ffi::Error::InvalidArgument(
          "packed descriptor indptr span is out of range or nonmonotonic");
    }
    if (indptr_length != node_length + 1) {
      return ffi::Error::InvalidArgument(
          "packed descriptor indptr length must equal node length plus one");
    }
    if (!SpanInRange(edge_start, edge_length, indices.dimensions()[0]) ||
        edge_start != expected_edge_start) {
      return ffi::Error::InvalidArgument(
          "packed descriptor edge span is out of range or nonmonotonic");
    }
    if (!SpanInRange(compressed_start, compressed_length,
                     b.dimensions()[0]) ||
        compressed_start != expected_compressed_start ||
        compressed_length == 0) {
      return ffi::Error::InvalidArgument(
          "packed descriptor compressed span is out of range or nonmonotonic");
    }
    if (min_index_to_keep < node_start ||
        min_index_to_keep >= node_start + node_length) {
      return ffi::Error::InvalidArgument(
          "packed descriptor min_index_to_keep is outside its node span");
    }
    if (indptr_data[indptr_start] != edge_start ||
        indptr_data[indptr_start + indptr_length - 1] !=
            edge_start + edge_length) {
      return ffi::Error::InvalidArgument(
          "packed indptr endpoints do not match the descriptor edge span");
    }
    for (int64_t offset = 0; offset + 1 < indptr_length; ++offset) {
      const int64_t current = indptr_data[indptr_start + offset];
      const int64_t next = indptr_data[indptr_start + offset + 1];
      if (current < edge_start || current > next ||
          next > edge_start + edge_length) {
        return ffi::Error::InvalidArgument(
            "packed indptr entries must be monotonic within the edge span");
      }
    }
    for (int64_t edge = edge_start; edge < edge_start + edge_length; ++edge) {
      if (indices_data[edge] < node_start ||
          indices_data[edge] >= node_start + node_length) {
        return ffi::Error::InvalidArgument(
            "packed graph index is outside its descriptor node span");
      }
    }
    for (int64_t node = node_start; node < node_start + node_length; ++node) {
      if (nonunique[node] < compressed_start ||
          nonunique[node] >= compressed_start + compressed_length) {
        return ffi::Error::InvalidArgument(
            "packed nonunique index is outside its descriptor compressed span");
      }
    }

    expected_node_start += node_length;
    expected_indptr_start += indptr_length;
    expected_edge_start += edge_length;
    expected_compressed_start += compressed_length;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error SolvePackedForwardCompressed(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<dtype> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<ffi::S32> descriptors,
    ffi::BufferR2<dtype> b,
    ffi::ResultBufferR2<dtype> out) {
  ffi::Error error = ValidatePackedSolve(
      indptr, indices, data, nonunique_indices, descriptors, b);
  if (!error.success()) {
    return error;
  }
  using T = ffi::NativeType<dtype>;
  CopyBuffer(b, out);
  const int64_t n_cols = b.dimensions()[1];
  const int32_t* rows = descriptors.typed_data();
  const int32_t* indptr_data = indptr.typed_data();
  const int32_t* indices_data = indices.typed_data();
  const T* weights = data.typed_data();
  const int32_t* nonunique = nonunique_indices.typed_data();
  T* values = out->typed_data();

  for (int64_t slot = 0; slot < descriptors.dimensions()[0]; ++slot) {
    const int32_t* row = rows + slot * kPackedDescriptorColumnCount;
    if (row[kValid] == 0) {
      break;
    }
    const int64_t node_start = row[kNodeStart];
    const int64_t node_stop = node_start + row[kNodeLength];
    const int64_t indptr_start = row[kIndptrStart];
    const int64_t min_index_to_keep = row[kMinIndexToKeep];
    int64_t edge = row[kEdgeStart];
    for (int64_t node = node_start; node < node_stop; ++node) {
      const int64_t local_node = node - node_start;
      const int64_t edge_stop = indptr_data[indptr_start + local_node + 1];
      if (edge == edge_stop) {
        continue;
      }
      T* src_row = values + static_cast<int64_t>(nonunique[node]) * n_cols;
      while (edge < edge_stop) {
        T* dst_row = values +
            static_cast<int64_t>(nonunique[indices_data[edge]]) * n_cols;
        Axpy(n_cols, weights[edge], src_row, dst_row);
        ++edge;
      }
      if (node < min_index_to_keep) {
        ZeroRow(n_cols, src_row);
      }
    }
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error SolvePackedBackwardCompressed(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<dtype> data,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<ffi::S32> descriptors,
    ffi::BufferR2<dtype> b,
    ffi::ResultBufferR2<dtype> out) {
  ffi::Error error = ValidatePackedSolve(
      indptr, indices, data, nonunique_indices, descriptors, b);
  if (!error.success()) {
    return error;
  }
  using T = ffi::NativeType<dtype>;
  CopyBuffer(b, out);
  const int64_t n_cols = b.dimensions()[1];
  const int32_t* rows = descriptors.typed_data();
  const int32_t* indptr_data = indptr.typed_data();
  const int32_t* indices_data = indices.typed_data();
  const T* weights = data.typed_data();
  const int32_t* nonunique = nonunique_indices.typed_data();
  T* values = out->typed_data();

  for (int64_t slot = 0; slot < descriptors.dimensions()[0]; ++slot) {
    const int32_t* row = rows + slot * kPackedDescriptorColumnCount;
    if (row[kValid] == 0) {
      break;
    }
    const int64_t node_start = row[kNodeStart];
    const int64_t node_stop = node_start + row[kNodeLength];
    const int64_t indptr_start = row[kIndptrStart];
    const int64_t min_index_to_keep = row[kMinIndexToKeep];
    int64_t edge = row[kEdgeStart] + row[kEdgeLength];
    for (int64_t node = node_stop - 1; node >= node_start; --node) {
      const int64_t local_node = node - node_start;
      const int64_t edge_start = indptr_data[indptr_start + local_node];
      if (edge == edge_start) {
        continue;
      }
      T* dst_row = values + static_cast<int64_t>(nonunique[node]) * n_cols;
      if (node < min_index_to_keep) {
        ZeroRow(n_cols, dst_row);
      }
      while (edge > edge_start) {
        --edge;
        T* src_row = values +
            static_cast<int64_t>(nonunique[indices_data[edge]]) * n_cols;
        Axpy(n_cols, weights[edge], src_row, dst_row);
      }
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

#define LINEAR_DAG_PACKED_HANDLER(name, dtype, solve)                           \
  static ffi::Error name(                                                       \
      ffi::BufferR1<ffi::S32> indptr,                                           \
      ffi::BufferR1<ffi::S32> indices,                                          \
      ffi::BufferR1<dtype> data,                                                \
      ffi::BufferR1<ffi::S32> nonunique_indices,                                \
      ffi::BufferR2<ffi::S32> descriptors,                                      \
      ffi::BufferR2<dtype> b, ffi::ResultBufferR2<dtype> out) {                 \
    return solve<dtype>(                                                        \
        indptr, indices, data, nonunique_indices, descriptors, b, out);         \
  }

LINEAR_DAG_PACKED_HANDLER(SolvePackedForwardF32, ffi::F32,
                          SolvePackedForwardCompressed)
LINEAR_DAG_PACKED_HANDLER(SolvePackedBackwardF32, ffi::F32,
                          SolvePackedBackwardCompressed)
LINEAR_DAG_PACKED_HANDLER(SolvePackedForwardF64, ffi::F64,
                          SolvePackedForwardCompressed)
LINEAR_DAG_PACKED_HANDLER(SolvePackedBackwardF64, ffi::F64,
                          SolvePackedBackwardCompressed)

#undef LINEAR_DAG_PACKED_HANDLER

#define LINEAR_DAG_BINDING(dtype)                                               \
  ffi::Ffi::Bind()                                                              \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<dtype>>()                                              \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR2<dtype>>()                                              \
      .Ret<ffi::BufferR2<dtype>>()                                              \
      .Attr<int64_t>("min_index_to_keep")

#define LINEAR_DAG_PACKED_BINDING(dtype)                                        \
  ffi::Ffi::Bind()                                                              \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<dtype>>()                                              \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR2<ffi::S32>>()                                           \
      .Arg<ffi::BufferR2<dtype>>()                                              \
      .Ret<ffi::BufferR2<dtype>>()

XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_solve_forward_f32, SolveForwardF32,
                              LINEAR_DAG_BINDING(ffi::F32));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_solve_backward_f32, SolveBackwardF32,
                              LINEAR_DAG_BINDING(ffi::F32));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_solve_forward_f64, SolveForwardF64,
                              LINEAR_DAG_BINDING(ffi::F64));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_solve_backward_f64, SolveBackwardF64,
                              LINEAR_DAG_BINDING(ffi::F64));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_packed_solve_forward_f32,
                              SolvePackedForwardF32,
                              LINEAR_DAG_PACKED_BINDING(ffi::F32));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_packed_solve_backward_f32,
                              SolvePackedBackwardF32,
                              LINEAR_DAG_PACKED_BINDING(ffi::F32));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_packed_solve_forward_f64,
                              SolvePackedForwardF64,
                              LINEAR_DAG_PACKED_BINDING(ffi::F64));
XLA_FFI_DEFINE_HANDLER_SYMBOL(linear_dag_jaxlinarg_packed_solve_backward_f64,
                              SolvePackedBackwardF64,
                              LINEAR_DAG_PACKED_BINDING(ffi::F64));

#undef LINEAR_DAG_BINDING
#undef LINEAR_DAG_PACKED_BINDING

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
      AddRegistration(dict, "linear_dag_jaxlinarg_packed_solve_forward_f32",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_packed_solve_forward_f32)) < 0 ||
      AddRegistration(dict, "linear_dag_jaxlinarg_packed_solve_backward_f32",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_packed_solve_backward_f32)) < 0 ||
      AddRegistration(dict, "linear_dag_jaxlinarg_packed_solve_forward_f64",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_packed_solve_forward_f64)) < 0 ||
      AddRegistration(dict, "linear_dag_jaxlinarg_packed_solve_backward_f64",
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_packed_solve_backward_f64)) < 0) {
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

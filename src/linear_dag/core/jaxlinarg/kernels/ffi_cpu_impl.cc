// pattern: Imperative Shell

#include <Python.h>

#include <algorithm>
#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template <ffi::DataType dtype>
ffi::Error ValidateBuffers(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<dtype> data,
    ffi::BufferR1<ffi::S32> src_of_edge,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<dtype> b,
    ffi::ResultBufferR2<dtype> out) {
  if (indptr.dimensions()[0] == 0) {
    return ffi::Error::InvalidArgument("indptr must contain at least one entry");
  }
  const int64_t n_edges = indices.dimensions()[0];
  if (data.dimensions()[0] != n_edges) {
    return ffi::Error::InvalidArgument("data must have the same length as indices");
  }
  if (src_of_edge.dimensions()[0] != n_edges) {
    return ffi::Error::InvalidArgument("src_of_edge must have the same length as indices");
  }
  if (nonunique_indices.dimensions()[0] != indptr.dimensions()[0] - 1) {
    return ffi::Error::InvalidArgument("nonunique_indices length must match node count");
  }
  if (out->dimensions()[0] != b.dimensions()[0] || out->dimensions()[1] != b.dimensions()[1]) {
    return ffi::Error::InvalidArgument("output shape must match b");
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
void CopyBuffer(ffi::BufferR2<dtype> b, ffi::ResultBufferR2<dtype> out) {
  const int64_t element_count = b.dimensions()[0] * b.dimensions()[1];
  std::copy(b.typed_data(), b.typed_data() + element_count, out->typed_data());
}

template <ffi::DataType dtype>
ffi::Error SolveForwardCompressed(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<dtype> data,
    ffi::BufferR1<ffi::S32> src_of_edge,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<dtype> b,
    ffi::ResultBufferR2<dtype> out,
    int64_t min_index_to_keep) {
  if (auto error = ValidateBuffers(indptr, indices, data, src_of_edge, nonunique_indices, b, out);
      error.failure()) {
    return error;
  }
  using T = ffi::NativeType<dtype>;
  CopyBuffer(b, out);

  const int64_t n_edges = indices.dimensions()[0];
  const int64_t n_cols = b.dimensions()[1];
  T* values = out->typed_data();
  const int32_t* indptr_data = indptr.typed_data();
  const int32_t* indices_data = indices.typed_data();
  const T* weights = data.typed_data();
  const int32_t* src_data = src_of_edge.typed_data();
  const int32_t* nonunique = nonunique_indices.typed_data();

  for (int64_t edge = 0; edge < n_edges; ++edge) {
    const int32_t src = src_data[edge];
    const int32_t dst = indices_data[edge];
    const int32_t src_col = nonunique[src];
    const int32_t dst_col = nonunique[dst];
    const T weight = weights[edge];
    for (int64_t col = 0; col < n_cols; ++col) {
      values[dst_col * n_cols + col] += values[src_col * n_cols + col] * weight;
    }
    if (edge == indptr_data[src + 1] - 1 && src < min_index_to_keep) {
      for (int64_t col = 0; col < n_cols; ++col) {
        values[src_col * n_cols + col] = T{0};
      }
    }
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error SolveBackwardCompressed(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<dtype> data,
    ffi::BufferR1<ffi::S32> src_of_edge,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<dtype> b,
    ffi::ResultBufferR2<dtype> out,
    int64_t min_index_to_keep) {
  if (auto error = ValidateBuffers(indptr, indices, data, src_of_edge, nonunique_indices, b, out);
      error.failure()) {
    return error;
  }
  using T = ffi::NativeType<dtype>;
  CopyBuffer(b, out);

  const int64_t n_edges = indices.dimensions()[0];
  const int64_t n_cols = b.dimensions()[1];
  T* values = out->typed_data();
  const int32_t* indptr_data = indptr.typed_data();
  const int32_t* indices_data = indices.typed_data();
  const T* weights = data.typed_data();
  const int32_t* src_data = src_of_edge.typed_data();
  const int32_t* nonunique = nonunique_indices.typed_data();

  for (int64_t edge = n_edges - 1; edge >= 0; --edge) {
    const int32_t src = src_data[edge];
    const int32_t dst = indices_data[edge];
    const int32_t src_col = nonunique[src];
    const int32_t dst_col = nonunique[dst];
    if (edge == indptr_data[src + 1] - 1 && src < min_index_to_keep) {
      for (int64_t col = 0; col < n_cols; ++col) {
        values[src_col * n_cols + col] = T{0};
      }
    }
    const T weight = weights[edge];
    for (int64_t col = 0; col < n_cols; ++col) {
      values[src_col * n_cols + col] += values[dst_col * n_cols + col] * weight;
    }
  }
  return ffi::Error::Success();
}

static ffi::Error SolveForwardF32(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::F32> data,
    ffi::BufferR1<ffi::S32> src_of_edge,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<ffi::F32> b,
    ffi::ResultBufferR2<ffi::F32> out,
    int64_t min_index_to_keep) {
  return SolveForwardCompressed<ffi::F32>(
      indptr, indices, data, src_of_edge, nonunique_indices, b, out, min_index_to_keep);
}

static ffi::Error SolveBackwardF32(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::F32> data,
    ffi::BufferR1<ffi::S32> src_of_edge,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<ffi::F32> b,
    ffi::ResultBufferR2<ffi::F32> out,
    int64_t min_index_to_keep) {
  return SolveBackwardCompressed<ffi::F32>(
      indptr, indices, data, src_of_edge, nonunique_indices, b, out, min_index_to_keep);
}

static ffi::Error SolveForwardF64(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::F64> data,
    ffi::BufferR1<ffi::S32> src_of_edge,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<ffi::F64> b,
    ffi::ResultBufferR2<ffi::F64> out,
    int64_t min_index_to_keep) {
  return SolveForwardCompressed<ffi::F64>(
      indptr, indices, data, src_of_edge, nonunique_indices, b, out, min_index_to_keep);
}

static ffi::Error SolveBackwardF64(
    ffi::BufferR1<ffi::S32> indptr,
    ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::F64> data,
    ffi::BufferR1<ffi::S32> src_of_edge,
    ffi::BufferR1<ffi::S32> nonunique_indices,
    ffi::BufferR2<ffi::F64> b,
    ffi::ResultBufferR2<ffi::F64> out,
    int64_t min_index_to_keep) {
  return SolveBackwardCompressed<ffi::F64>(
      indptr, indices, data, src_of_edge, nonunique_indices, b, out, min_index_to_keep);
}

#define LINEAR_DAG_BINDING(dtype)                                               \
  ffi::Ffi::Bind()                                                              \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<dtype>>()                                              \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR1<ffi::S32>>()                                           \
      .Arg<ffi::BufferR2<dtype>>()                                              \
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

#undef LINEAR_DAG_BINDING

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
                      reinterpret_cast<void*>(linear_dag_jaxlinarg_solve_backward_f64)) < 0) {
    Py_DECREF(dict);
    return nullptr;
  }
  return dict;
}

static PyMethodDef Methods[] = {
    {"registrations", Registrations, METH_NOARGS, "Return CPU FFI target registrations."},
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

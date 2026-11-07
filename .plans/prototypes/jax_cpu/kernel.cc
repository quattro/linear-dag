// Experimental packed ragged CPU execution. Graph buffers are validated at ingress.
#include <Python.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

template <typename T>
void Add(int64_t k, T scale, const T* src, T* dst) {
  for (int64_t c = 0; c < k; ++c) dst[c] += scale * src[c];
}

template <ffi::DataType dtype>
ffi::Future Execute(
    ffi::ThreadPool pool, ffi::BufferR2<ffi::S32> metadata,
    ffi::BufferR1<ffi::S32> ptr, ffi::BufferR1<ffi::S32> indices,
    ffi::BufferR1<ffi::S32> weights, ffi::BufferR1<ffi::S32> mapping,
    ffi::BufferR1<ffi::S32> variants, ffi::BufferR1<ffi::S32> samples,
    ffi::BufferR1<ffi::PRED> flips,
    ffi::BufferR2<dtype> a, ffi::BufferR2<dtype> b,
    ffi::ResultBufferR2<dtype> output, ffi::ResultBufferR3<dtype> scratch,
    int64_t mode, int64_t n_samples, int64_t max_state, int64_t max_variants) {
  using T = ffi::NativeType<dtype>;
  ffi::Promise promise;
  ffi::Future future(promise);
  const int64_t workers = scratch->dimensions()[0];
  const int64_t blocks = metadata.dimensions()[0];
  const int64_t k = (mode == 2 ? b : a).dimensions()[1];
  const int64_t rows = max_state + 2 * max_variants + n_samples + 2;
  if (mode < 0 || mode > 3 || workers < 1 || workers > blocks ||
      metadata.dimensions()[1] != 8 || scratch->dimensions()[1] != rows ||
      scratch->dimensions()[2] != k) {
    promise.SetError(ffi::Error::InvalidArgument("invalid packed execution shape"));
    return future;
  }
  const int32_t* meta = metadata.typed_data();
  const int32_t* ptrs = ptr.typed_data();
  const int32_t* edges = indices.typed_data();
  const int32_t* vals = weights.typed_data();
  const int32_t* maps = mapping.typed_data();
  const int32_t* vars = variants.typed_data();
  const int32_t* samps = samples.typed_data();
  const bool* flip = flips.typed_data();
  const T* av = a.typed_data();
  const T* bv = b.typed_data();
  T* out = output->typed_data();
  T* work = scratch->typed_data();
  auto remaining = std::make_shared<std::atomic<int64_t>>(workers);
  for (int64_t worker = 0; worker < workers; ++worker) {
    pool.Schedule([=]() mutable {
      T* state = work + worker * rows * k;
      T* scores = state + max_state * k;
      T* scores2 = scores + max_variants * k;
      T* accumulator = scores2 + max_variants * k;
      T* totals = accumulator + n_samples * k;
      std::fill(accumulator, accumulator + n_samples * k, T(0));
      // Fixed round-robin assignment and fixed final worker reduction order.
      for (int64_t block = worker; block < blocks; block += workers) {
        const int32_t* m = meta + block * 8;
        const int64_t po=m[0], no=m[1], eo=m[2], vo=m[3];
        const int64_t nodes=m[4], nv=m[5], ns=m[6], keep=m[7];
        const int32_t* map = maps + no;
        const int32_t* vp = vars + vo;
        const int32_t* sp = samps + block * n_samples;
        auto reverse = [&](const T* input, T* result) {
          std::fill(state, state + ns*k, T(0));
          std::fill(totals, totals + k, T(0));
          for (int64_t s=0; s<n_samples; ++s) {
            std::copy(input+s*k, input+(s+1)*k, state+int64_t(map[sp[s]])*k);
            Add(k,T(1),input+s*k,totals);
          }
          for (int64_t node=nodes-1; node>=0; --node) {
            const int64_t start=ptrs[po+node], stop=ptrs[po+node+1];
            T* dst=state+int64_t(map[node])*k;
            if(start!=stop && node<keep) std::fill(dst,dst+k,T(0));
            for(int64_t e=stop-1;e>=start;--e)
              Add(k,T(vals[eo+e]),state+int64_t(map[edges[eo+e]])*k,dst);
          }
          for(int64_t v=0;v<nv;++v) {
            const T* src=state+int64_t(map[vp[v]])*k;
            for(int64_t c=0;c<k;++c) result[v*k+c]=flip[vo+v]?totals[c]-src[c]:src[c];
          }
        };
        auto forward = [&](const T* input) {
          std::fill(state,state+ns*k,T(0));
          std::fill(totals,totals+k,T(0));
          for(int64_t v=0;v<nv;++v) {
            Add(k,flip[vo+v]?T(-1):T(1),input+v*k,state+int64_t(map[vp[v]])*k);
            if(flip[vo+v]) Add(k,T(1),input+v*k,totals);
          }
          for(int64_t node=0;node<nodes;++node) {
            const int64_t start=ptrs[po+node],stop=ptrs[po+node+1];
            T* src=state+int64_t(map[node])*k;
            for(int64_t e=start;e<stop;++e)
              Add(k,T(vals[eo+e]),src,state+int64_t(map[edges[eo+e]])*k);
            if(start!=stop && node<keep) std::fill(src,src+k,T(0));
          }
          for(int64_t s=0;s<n_samples;++s) {
            Add(k,T(1),state+int64_t(map[sp[s]])*k,accumulator+s*k);
            Add(k,T(1),totals,accumulator+s*k);
          }
        };
        if(mode==0) forward(av+vo*k);
        if(mode==1) reverse(av,out+vo*k);
        if(mode==2) {
          reverse(bv,scores);
          for(int64_t v=0;v<nv;++v)
            for(int64_t c=0;c<k;++c) scores[v*k+c]*=av[vo+v];
          forward(scores);
        }
        if(mode==3) {
          reverse(av,scores); reverse(bv,scores2);
          for(int64_t v=0;v<nv;++v) {
            T sum=0;
            for(int64_t c=0;c<k;++c) sum+=scores[v*k+c]*scores2[v*k+c];
            out[vo+v]=sum;
          }
        }
      }
      if(remaining->fetch_sub(1,std::memory_order_acq_rel)==1) {
        if(mode==0 || mode==2) {
          std::fill(out,out+n_samples*k,T(0));
          for(int64_t w=0;w<workers;++w)
            Add(n_samples*k,T(1),work+(w*rows+max_state+2*max_variants)*k,out);
        }
        promise.SetAvailable();
      }
    });
  }
  return future;
}

#define BIND(T) ffi::Ffi::Bind().Ctx<ffi::ThreadPool>() \
  .Arg<ffi::BufferR2<ffi::S32>>() \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>() \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>() \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>() \
  .Arg<ffi::BufferR1<ffi::PRED>>() \
  .Arg<ffi::BufferR2<T>>().Arg<ffi::BufferR2<T>>() \
  .Ret<ffi::BufferR2<T>>().Ret<ffi::BufferR3<T>>() \
  .Attr<int64_t>("mode").Attr<int64_t>("n_samples") \
  .Attr<int64_t>("max_state").Attr<int64_t>("max_variants")
XLA_FFI_DEFINE_HANDLER_SYMBOL(packed_f32, Execute<ffi::F32>, BIND(ffi::F32));
XLA_FFI_DEFINE_HANDLER_SYMBOL(packed_f64, Execute<ffi::F64>, BIND(ffi::F64));

static PyObject* Registrations(PyObject*,PyObject*) {
  PyObject* result=PyDict_New();
  if(!result) return nullptr;
  for(auto pair : {std::pair<const char*, void*>("packed_f32",reinterpret_cast<void*>(packed_f32)),
                   std::pair<const char*, void*>("packed_f64",reinterpret_cast<void*>(packed_f64))}) {
    PyObject* capsule=PyCapsule_New(pair.second,nullptr,nullptr);
    if(!capsule || PyDict_SetItemString(result,pair.first,capsule)<0) {
      Py_XDECREF(capsule); Py_DECREF(result); return nullptr;
    }
    Py_DECREF(capsule);
  }
  return result;
}
static PyMethodDef methods[]={{"registrations",Registrations,METH_NOARGS,"FFI registrations"},{nullptr,nullptr,0,nullptr}};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_packed_cpu",nullptr,-1,methods};
PyMODINIT_FUNC PyInit__packed_cpu() {return PyModule_Create(&module);}

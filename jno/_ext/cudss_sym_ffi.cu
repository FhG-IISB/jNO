// cuDSS symmetric LDL^T as XLA custom calls: refactorise from DEVICE values and solve, both INSIDE a
// compiled program on XLA's stream, so jax.jit(jax.value_and_grad(f)) is GPU-resident end to end.
//
// Two handlers, tied by a token (an int32[1] result of the factor call that the solve calls take as an
// argument, so XLA orders them):
//
//   jno_cudsd_factor(values[nnz])        -> token[1]       first call: analysis + factorisation; after: refactorisation
//   jno_cudsd_solve (token, b[k, n])     -> x[k, n]        k right-hand sides, each contiguous (= column-major n x k)
//
// A SLOT holds the sparsity (upper triangle, CSR), the cuDSS handle/data and the factors. The pattern is
// registered once from the host by `jno_cudsd_create`; the VALUES arrive on the device at run time. The
// matrix is real symmetric (indefinite): A^T = A, so the adjoint solve of reverse mode is the same call.
//
// cuDSS runs on the slot's own stream, ordered against XLA's by events (binding the handle to XLA's stream
// per call returned wrong solves after the first -- see cudss_ffi.cu).
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

#include "cudss.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

struct Slot {
  cudssHandle_t h = nullptr;
  cudssConfig_t cfg = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t A = nullptr;
  int64_t n = 0, nnz = 0;
  int *ptr = nullptr, *idx = nullptr;
  double *val = nullptr, *scratch = nullptr;
  bool factored = false;
  cudaStream_t own = nullptr;
  cudaEvent_t in = nullptr, out = nullptr;
  std::map<int64_t, std::pair<cudssMatrix_t, cudssMatrix_t>> dn;  // nrhs -> (X, B), persistent descriptors
};

std::map<int64_t, Slot> g_slots;
std::mutex g_mu;

void release(Slot &s) {
  for (auto &kv : s.dn) {
    if (kv.second.first) cudssMatrixDestroy(kv.second.first);
    if (kv.second.second) cudssMatrixDestroy(kv.second.second);
  }
  if (s.A) cudssMatrixDestroy(s.A);
  if (s.data) cudssDataDestroy(s.h, s.data);
  if (s.cfg) cudssConfigDestroy(s.cfg);
  if (s.h) cudssDestroy(s.h);
  if (s.in) cudaEventDestroy(s.in);
  if (s.out) cudaEventDestroy(s.out);
  if (s.own) cudaStreamDestroy(s.own);
  cudaFree(s.ptr); cudaFree(s.idx); cudaFree(s.val); cudaFree(s.scratch);
  s = Slot();
}

}  // namespace

// Host-side registration of the sparsity: CSR of the UPPER triangle (cols >= rows), sorted, zero-based.
// Returns 0 on success, otherwise a cudaError, or 1000 + a cudssStatus.
extern "C" int jno_cudsd_create(int64_t slot, int64_t n, int64_t nnz, const int32_t *ptr, const int32_t *idx) {
  std::lock_guard<std::mutex> lock(g_mu);
  Slot &s = g_slots[slot];
  release(s);
  cudaError_t ce;
  cudssStatus_t st;
#define CK(x) do { ce = (x); if (ce != cudaSuccess) return (int)ce; } while (0)
#define CS(x) do { st = (x); if (st != CUDSS_STATUS_SUCCESS) return 1000 + (int)st; } while (0)
  s.n = n; s.nnz = nnz;
  CK(cudaMalloc(&s.ptr, sizeof(int32_t) * (n + 1)));
  CK(cudaMalloc(&s.idx, sizeof(int32_t) * nnz));
  CK(cudaMalloc(&s.val, sizeof(double) * nnz));
  CK(cudaMalloc(&s.scratch, sizeof(double) * n));
  CK(cudaMemcpy(s.ptr, ptr, sizeof(int32_t) * (n + 1), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s.idx, idx, sizeof(int32_t) * nnz, cudaMemcpyHostToDevice));
  CK(cudaMemset(s.val, 0, sizeof(double) * nnz));
  CK(cudaMemset(s.scratch, 0, sizeof(double) * n));
  CS(cudssCreate(&s.h));
  CK(cudaStreamCreateWithFlags(&s.own, cudaStreamNonBlocking));
  CK(cudaEventCreateWithFlags(&s.in, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&s.out, cudaEventDisableTiming));
  CS(cudssSetStream(s.h, s.own));
  CS(cudssConfigCreate(&s.cfg));
  CS(cudssDataCreate(s.h, &s.data));
  CS(cudssMatrixCreateCsr(&s.A, n, n, nnz, s.ptr, nullptr, s.idx, s.val, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                          CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO));
  return 0;
}

extern "C" void jno_cudsd_free(int64_t slot) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_slots.find(slot);
  if (it != g_slots.end()) {
    release(it->second);
    g_slots.erase(it);
  }
}

namespace {

Slot *find(int64_t slot) {
  auto it = g_slots.find(slot);
  return (it == g_slots.end() || !it->second.h) ? nullptr : &it->second;
}

// x = A^{-1} b is ordered after `stream`'s prior work and before its later work, with cuDSS on `own`.
ffi::Error FactorImpl(cudaStream_t stream, ffi::Buffer<ffi::F64> values, ffi::ResultBuffer<ffi::S32> token,
                      int64_t slot) {
  std::lock_guard<std::mutex> lock(g_mu);
  Slot *s = find(slot);
  if (!s) return ffi::Error::InvalidArgument("jno_cudsd_factor: empty slot");
  if ((int64_t)values.element_count() != s->nnz) return ffi::Error::InvalidArgument("jno_cudsd_factor: nnz mismatch");
  cudaEventRecord(s->in, stream);
  cudaStreamWaitEvent(s->own, s->in, 0);
  cudaMemcpyAsync(s->val, values.typed_data(), sizeof(double) * s->nnz, cudaMemcpyDeviceToDevice, s->own);
  cudssStatus_t st;
  if (!s->factored) {
    cudssMatrix_t X, B;
    cudssMatrixCreateDn(&X, s->n, 1, s->n, s->scratch, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&B, s->n, 1, s->n, s->scratch, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    st = cudssExecute(s->h, CUDSS_PHASE_ANALYSIS, s->cfg, s->data, s->A, X, B);
    if (st == CUDSS_STATUS_SUCCESS) st = cudssExecute(s->h, CUDSS_PHASE_FACTORIZATION, s->cfg, s->data, s->A, X, B);
    cudaStreamSynchronize(s->own);  // the scratch descriptors die here
    cudssMatrixDestroy(X);
    cudssMatrixDestroy(B);
    s->factored = (st == CUDSS_STATUS_SUCCESS);
  } else {
    cudssMatrix_t X, B;
    cudssMatrixCreateDn(&X, s->n, 1, s->n, s->scratch, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&B, s->n, 1, s->n, s->scratch, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    st = cudssExecute(s->h, CUDSS_PHASE_REFACTORIZATION, s->cfg, s->data, s->A, X, B);
    cudaStreamSynchronize(s->own);
    cudssMatrixDestroy(X);
    cudssMatrixDestroy(B);
  }
  if (st != CUDSS_STATUS_SUCCESS) return ffi::Error::Internal("jno_cudsd_factor: cuDSS (re)factorisation failed");
  cudaEventRecord(s->out, s->own);
  cudaStreamWaitEvent(stream, s->out, 0);
  cudaMemsetAsync(token->typed_data(), 0, sizeof(int32_t), stream);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

ffi::Error SolveImpl(cudaStream_t stream, ffi::Buffer<ffi::S32> token, ffi::Buffer<ffi::F64> b,
                     ffi::ResultBuffer<ffi::F64> x, int64_t slot) {
  (void)token;  // present only to order this call after the factorisation that produced it
  std::lock_guard<std::mutex> lock(g_mu);
  Slot *s = find(slot);
  if (!s || !s->factored) return ffi::Error::InvalidArgument("jno_cudsd_solve: slot not factored");
  auto dims = b.dimensions();
  if (dims.size() != 2 || dims[1] != s->n) return ffi::Error::InvalidArgument("jno_cudsd_solve: b must be (k, n)");
  int64_t k = dims[0];
  auto it = s->dn.find(k);
  if (it == s->dn.end()) {
    cudssMatrix_t X, B;
    if (cudssMatrixCreateDn(&X, s->n, k, s->n, x->typed_data(), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR) != CUDSS_STATUS_SUCCESS ||
        cudssMatrixCreateDn(&B, s->n, k, s->n, b.typed_data(), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR) != CUDSS_STATUS_SUCCESS)
      return ffi::Error::Internal("jno_cudsd_solve: descriptor creation failed");
    it = s->dn.emplace(k, std::make_pair(X, B)).first;
  } else {
    cudssMatrixSetValues(it->second.first, x->typed_data());
    cudssMatrixSetValues(it->second.second, const_cast<double *>(b.typed_data()));
  }
  cudaEventRecord(s->in, stream);
  cudaStreamWaitEvent(s->own, s->in, 0);
  if (cudssExecute(s->h, CUDSS_PHASE_SOLVE, s->cfg, s->data, s->A, it->second.first, it->second.second) != CUDSS_STATUS_SUCCESS)
    return ffi::Error::Internal("jno_cudsd_solve: cuDSS solve failed");
  cudaEventRecord(s->out, s->own);
  cudaStreamWaitEvent(stream, s->out, 0);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(jno_cudsd_factor, FactorImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Attr<int64_t>("slot"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(jno_cudsd_solve, SolveImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("slot"));

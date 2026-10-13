// cuDSS symmetric LDL^T as XLA custom calls: refactorise from DEVICE values and solve, both INSIDE a
// compiled program on XLA's stream, so jax.jit(jax.value_and_grad(f)) is GPU-resident end to end.
//
// Two handlers per value type, tied by a token (an int32[1] result of the factor call that the solve
// calls take as an argument, so XLA orders them):
//
//   jno_cudsd_factor / jno_cudsz_factor (values[nnz])       -> token[1]   real / complex
//   jno_cudsd_solve  / jno_cudsz_solve  (token, b[k, n])    -> x[k, n]    k right-hand sides, each contiguous
//
// First factor call: analysis + factorisation; after that: refactorisation (the analysis is kept).
// A SLOT holds the sparsity (upper triangle, CSR), the cuDSS handle/data and the factors. The pattern is
// registered once from the host by `jno_cudsd_create`; the VALUES arrive on the device at run time. The
// matrix is symmetric -- real symmetric indefinite, or COMPLEX SYMMETRIC (A^T = A, not Hermitian) -- so the
// adjoint solve of reverse mode (the plain transpose) is the same call.
//
// cuDSS runs on the slot's own stream, ordered against XLA's by events (binding the handle to XLA's stream
// per call returned wrong solves after the first -- see cudss_ffi.cu).
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <string>
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
  bool cplx = false;
  size_t es = sizeof(double);                 // bytes per value
  int *ptr = nullptr, *idx = nullptr;
  void *val = nullptr, *scratch = nullptr;
  bool factored = false;
  cudaStream_t own = nullptr;
  cudaEvent_t in = nullptr, out = nullptr;
  std::map<int64_t, std::pair<cudssMatrix_t, cudssMatrix_t>> dn;  // nrhs -> (X, B), persistent descriptors
  cudssDataType_t dt() const { return cplx ? CUDSS_C_64F : CUDSS_R_64F; }
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

Slot *find(int64_t slot) {
  auto it = g_slots.find(slot);
  return (it == g_slots.end() || !it->second.h) ? nullptr : &it->second;
}

}  // namespace

// Host-side registration of the sparsity: CSR of the UPPER triangle (cols >= rows), sorted, zero-based.
// `cplx` = 1 for complex double values. Returns 0, a cudaError, or 1000 + a cudssStatus.
extern "C" int jno_cudsd_create(int64_t slot, int64_t n, int64_t nnz, const int32_t *ptr, const int32_t *idx, int cplx) {
  std::lock_guard<std::mutex> lock(g_mu);
  Slot &s = g_slots[slot];
  release(s);
  cudaError_t ce;
  cudssStatus_t st;
#define CK(x) do { ce = (x); if (ce != cudaSuccess) return (int)ce; } while (0)
#define CS(x) do { st = (x); if (st != CUDSS_STATUS_SUCCESS) return 1000 + (int)st; } while (0)
  s.n = n; s.nnz = nnz; s.cplx = cplx != 0; s.es = s.cplx ? sizeof(cuDoubleComplex) : sizeof(double);
  CK(cudaMalloc(&s.ptr, sizeof(int32_t) * (n + 1)));
  CK(cudaMalloc(&s.idx, sizeof(int32_t) * nnz));
  CK(cudaMalloc(&s.val, s.es * nnz));
  CK(cudaMalloc(&s.scratch, s.es * n));
  CK(cudaMemcpy(s.ptr, ptr, sizeof(int32_t) * (n + 1), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s.idx, idx, sizeof(int32_t) * nnz, cudaMemcpyHostToDevice));
  CK(cudaMemset(s.val, 0, s.es * nnz));
  CK(cudaMemset(s.scratch, 0, s.es * n));
  CS(cudssCreate(&s.h));
  CK(cudaStreamCreateWithFlags(&s.own, cudaStreamNonBlocking));
  CK(cudaEventCreateWithFlags(&s.in, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&s.out, cudaEventDisableTiming));
  CS(cudssSetStream(s.h, s.own));
  CS(cudssConfigCreate(&s.cfg));
  CS(cudssDataCreate(s.h, &s.data));
  CS(cudssMatrixCreateCsr(&s.A, n, n, nnz, s.ptr, nullptr, s.idx, s.val, CUDSS_R_32I, CUDSS_R_32I, s.dt(),
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

// Host query after a factorisation: out[0] = nnz of the factors, out[1] = perturbed pivots,
// out[2] = permanent device bytes, out[3] = peak device bytes (cuDSS's own estimates). 0 on success.
extern "C" int jno_cudsd_stats(int64_t slot, int64_t *out) {
  std::lock_guard<std::mutex> lock(g_mu);
  Slot *s = find(slot);
  if (!s || !s->factored) return -1;
  cudaStreamSynchronize(s->own);
  size_t w = 0;
  int64_t lu = 0, est[16] = {0};
  int piv = 0;
  if (cudssDataGet(s->h, s->data, CUDSS_DATA_LU_NNZ, &lu, sizeof(lu), &w) != CUDSS_STATUS_SUCCESS) return -2;
  cudssDataGet(s->h, s->data, CUDSS_DATA_NPIVOTS, &piv, sizeof(piv), &w);
  cudssDataGet(s->h, s->data, CUDSS_DATA_MEMORY_ESTIMATES, est, sizeof(est), &w);
  out[0] = lu; out[1] = piv; out[2] = est[0]; out[3] = est[1];
  return 0;
}

namespace {

ffi::Error factor_core(cudaStream_t stream, const void *values, int64_t count, int32_t *token, int64_t slot, bool cplx) {
  std::lock_guard<std::mutex> lock(g_mu);
  Slot *s = find(slot);
  if (!s) return ffi::Error::InvalidArgument("jno_cudss_sym factor: empty slot");
  if (s->cplx != cplx) return ffi::Error::InvalidArgument("jno_cudss_sym factor: value type differs from the plan's");
  if (count != s->nnz) return ffi::Error::InvalidArgument("jno_cudss_sym factor: nnz mismatch");
  cudaEventRecord(s->in, stream);
  cudaStreamWaitEvent(s->own, s->in, 0);
  cudaMemcpyAsync(s->val, values, s->es * s->nnz, cudaMemcpyDeviceToDevice, s->own);
  cudssMatrix_t X, B;
  cudssMatrixCreateDn(&X, s->n, 1, s->n, s->scratch, s->dt(), CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&B, s->n, 1, s->n, s->scratch, s->dt(), CUDSS_LAYOUT_COL_MAJOR);
  cudssStatus_t st;
  if (!s->factored) {
    st = cudssExecute(s->h, CUDSS_PHASE_ANALYSIS, s->cfg, s->data, s->A, X, B);
    if (st == CUDSS_STATUS_SUCCESS) st = cudssExecute(s->h, CUDSS_PHASE_FACTORIZATION, s->cfg, s->data, s->A, X, B);
  } else {
    st = cudssExecute(s->h, CUDSS_PHASE_REFACTORIZATION, s->cfg, s->data, s->A, X, B);
  }
  cudaStreamSynchronize(s->own);  // the scratch descriptors die here
  cudssMatrixDestroy(X);
  cudssMatrixDestroy(B);
  if (st != CUDSS_STATUS_SUCCESS)
    return ffi::Error::Internal(std::string("jno_cudss_sym factor: cuDSS (re)factorisation failed, status ") + std::to_string((int)st) +
                                (st == CUDSS_STATUS_ALLOC_FAILED ? " (out of device memory)" : ""));
  s->factored = true;
  cudaEventRecord(s->out, s->own);
  cudaStreamWaitEvent(stream, s->out, 0);
  cudaMemsetAsync(token, 0, sizeof(int32_t), stream);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

ffi::Error solve_core(cudaStream_t stream, const void *b, void *x, int64_t k, int64_t ncol, int64_t slot, bool cplx) {
  std::lock_guard<std::mutex> lock(g_mu);
  Slot *s = find(slot);
  if (!s || !s->factored) return ffi::Error::InvalidArgument("jno_cudss_sym solve: slot not factored");
  if (s->cplx != cplx) return ffi::Error::InvalidArgument("jno_cudss_sym solve: value type differs from the plan's");
  if (ncol != s->n) return ffi::Error::InvalidArgument("jno_cudss_sym solve: b must be (k, n)");
  auto it = s->dn.find(k);
  if (it == s->dn.end()) {
    cudssMatrix_t X, B;
    if (cudssMatrixCreateDn(&X, s->n, k, s->n, x, s->dt(), CUDSS_LAYOUT_COL_MAJOR) != CUDSS_STATUS_SUCCESS ||
        cudssMatrixCreateDn(&B, s->n, k, s->n, const_cast<void *>(b), s->dt(), CUDSS_LAYOUT_COL_MAJOR) != CUDSS_STATUS_SUCCESS)
      return ffi::Error::Internal("jno_cudss_sym solve: descriptor creation failed");
    it = s->dn.emplace(k, std::make_pair(X, B)).first;
  } else {
    cudssMatrixSetValues(it->second.first, x);
    cudssMatrixSetValues(it->second.second, const_cast<void *>(b));
  }
  cudaEventRecord(s->in, stream);
  cudaStreamWaitEvent(s->own, s->in, 0);
  if (cudssExecute(s->h, CUDSS_PHASE_SOLVE, s->cfg, s->data, s->A, it->second.first, it->second.second) != CUDSS_STATUS_SUCCESS)
    return ffi::Error::Internal("jno_cudss_sym solve: cuDSS solve failed");
  cudaEventRecord(s->out, s->own);
  cudaStreamWaitEvent(stream, s->out, 0);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

ffi::Error FactorD(cudaStream_t st, ffi::Buffer<ffi::F64> v, ffi::ResultBuffer<ffi::S32> tok, int64_t slot) {
  return factor_core(st, v.typed_data(), (int64_t)v.element_count(), tok->typed_data(), slot, false);
}
ffi::Error FactorZ(cudaStream_t st, ffi::Buffer<ffi::C128> v, ffi::ResultBuffer<ffi::S32> tok, int64_t slot) {
  return factor_core(st, v.typed_data(), (int64_t)v.element_count(), tok->typed_data(), slot, true);
}
ffi::Error SolveD(cudaStream_t st, ffi::Buffer<ffi::S32>, ffi::Buffer<ffi::F64> b, ffi::ResultBuffer<ffi::F64> x, int64_t slot) {
  auto d = b.dimensions();
  if (d.size() != 2) return ffi::Error::InvalidArgument("jno_cudss_sym solve: b must be 2-D (k, n)");
  return solve_core(st, b.typed_data(), x->typed_data(), d[0], d[1], slot, false);
}
ffi::Error SolveZ(cudaStream_t st, ffi::Buffer<ffi::S32>, ffi::Buffer<ffi::C128> b, ffi::ResultBuffer<ffi::C128> x, int64_t slot) {
  auto d = b.dimensions();
  if (d.size() != 2) return ffi::Error::InvalidArgument("jno_cudss_sym solve: b must be 2-D (k, n)");
  return solve_core(st, b.typed_data(), x->typed_data(), d[0], d[1], slot, true);
}

}  // namespace

#define JNO_FACTOR_BIND(T) ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<T>>().Ret<ffi::Buffer<ffi::S32>>().Attr<int64_t>("slot")
#define JNO_SOLVE_BIND(T) ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<T>>().Ret<ffi::Buffer<T>>().Attr<int64_t>("slot")

XLA_FFI_DEFINE_HANDLER_SYMBOL(jno_cudsd_factor, FactorD, JNO_FACTOR_BIND(ffi::F64));
XLA_FFI_DEFINE_HANDLER_SYMBOL(jno_cudsd_solve, SolveD, JNO_SOLVE_BIND(ffi::F64));
XLA_FFI_DEFINE_HANDLER_SYMBOL(jno_cudsz_factor, FactorZ, JNO_FACTOR_BIND(ffi::C128));
XLA_FFI_DEFINE_HANDLER_SYMBOL(jno_cudsz_solve, SolveZ, JNO_SOLVE_BIND(ffi::C128));

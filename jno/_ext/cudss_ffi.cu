// cuDSS as an XLA custom call: factorise ONCE (from the host, on demand), solve INSIDE a compiled
// program on XLA's own CUDA stream -- no host round trip per apply.
//
// A factorisation lives in a numbered SLOT. `jno_cudss_factor` (plain C, called from Python) fills
// a slot from host CSR arrays; the FFI handler `jno_cudss_solve` applies it to a device vector.
// Each slot also carries an optional row/column scaling (an equilibration done by the caller), so
// the handler solves  A x = b  as  x = c .* B^{-1} (r .* b)  with  B = diag(r) A diag(c).
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>

#include "cudss.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

struct Slot {
  cudssHandle_t h = nullptr;
  cudssConfig_t cfg = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t A = nullptr, X = nullptr, B = nullptr;
  int64_t n = 0, nnz = 0;
  int *ptr = nullptr, *idx = nullptr;
  cuDoubleComplex *val = nullptr, *tmp = nullptr, *xbuf = nullptr;
  double *r = nullptr, *c = nullptr;
  // cuDSS works on its OWN stream, ordered against XLA's by events: switching the handle onto XLA's
  // stream (cudssSetStream per call) made every solve after the first in a process wrong -- measured,
  // 2e-4 then 2.4 relative residual for the same factorisation -- while the same calls from the host
  // on one stream stayed at 2e-4.
  cudaStream_t own = nullptr;
  cudaEvent_t in = nullptr, out = nullptr;
};

std::map<int64_t, Slot> g_slots;
std::mutex g_mu;

void release(Slot &s) {
  if (s.A) cudssMatrixDestroy(s.A);
  if (s.X) cudssMatrixDestroy(s.X);
  if (s.B) cudssMatrixDestroy(s.B);
  if (s.data) cudssDataDestroy(s.h, s.data);
  if (s.cfg) cudssConfigDestroy(s.cfg);
  if (s.h) cudssDestroy(s.h);
  if (s.in) cudaEventDestroy(s.in);
  if (s.out) cudaEventDestroy(s.out);
  if (s.own) cudaStreamDestroy(s.own);
  cudaFree(s.ptr); cudaFree(s.idx); cudaFree(s.val); cudaFree(s.tmp); cudaFree(s.xbuf);
  cudaFree(s.r); cudaFree(s.c);
  s = Slot();
}

__global__ void scale_in(const cuDoubleComplex *b, const double *r, cuDoubleComplex *out, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = make_cuDoubleComplex(r[i] * cuCreal(b[i]), r[i] * cuCimag(b[i]));
}

__global__ void scale_out(const cuDoubleComplex *x, const double *c, cuDoubleComplex *out, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = make_cuDoubleComplex(c[i] * cuCreal(x[i]), c[i] * cuCimag(x[i]));
}

}  // namespace

// Returns 0 on success, otherwise a cudaError / cudssStatus (cudss codes offset by 1000).
extern "C" int jno_cudss_factor(int64_t slot, int64_t n, int64_t nnz, const int32_t *ptr, const int32_t *idx,
                                const double *val_interleaved, const double *r, const double *c, int matching) {
  std::lock_guard<std::mutex> lock(g_mu);
  Slot &s = g_slots[slot];
  bool same = s.h && s.n == n && s.nnz == nnz;
  if (same) {  // same sparsity is the caller's promise only if the indices match: check them
    int32_t *hp = new int32_t[n + 1];
    cudaMemcpy(hp, s.ptr, sizeof(int32_t) * (n + 1), cudaMemcpyDeviceToHost);
    same = std::memcmp(hp, ptr, sizeof(int32_t) * (n + 1)) == 0;
    delete[] hp;
    if (same) {
      int32_t *hi = new int32_t[nnz];
      cudaMemcpy(hi, s.idx, sizeof(int32_t) * nnz, cudaMemcpyDeviceToHost);
      same = std::memcmp(hi, idx, sizeof(int32_t) * nnz) == 0;
      delete[] hi;
    }
  }
  cudaError_t ce;
  cudssStatus_t st;
#define CK(x) do { ce = (x); if (ce != cudaSuccess) return (int)ce; } while (0)
#define CS(x) do { st = (x); if (st != CUDSS_STATUS_SUCCESS) return 1000 + (int)st; } while (0)
  if (!same) {
    release(s);
    s.n = n; s.nnz = nnz;
    CK(cudaMalloc(&s.ptr, sizeof(int32_t) * (n + 1)));
    CK(cudaMalloc(&s.idx, sizeof(int32_t) * nnz));
    CK(cudaMalloc(&s.val, sizeof(cuDoubleComplex) * nnz));
    CK(cudaMalloc(&s.tmp, sizeof(cuDoubleComplex) * n));
    CK(cudaMalloc(&s.xbuf, sizeof(cuDoubleComplex) * n));
    CK(cudaMalloc(&s.r, sizeof(double) * n));
    CK(cudaMalloc(&s.c, sizeof(double) * n));
    CK(cudaMemcpy(s.ptr, ptr, sizeof(int32_t) * (n + 1), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s.idx, idx, sizeof(int32_t) * nnz, cudaMemcpyHostToDevice));
    CS(cudssCreate(&s.h));
    CK(cudaStreamCreateWithFlags(&s.own, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&s.in, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&s.out, cudaEventDisableTiming));
    CS(cudssSetStream(s.h, s.own));
    CS(cudssConfigCreate(&s.cfg));
    if (matching) {
      cudssMatchingAlg_t alg = (cudssMatchingAlg_t)matching;  // the caller picks the algorithm
      CS(cudssConfigSet(s.cfg, CUDSS_CONFIG_MATCHING_ALG, &alg, sizeof(alg)));
    }
    CS(cudssDataCreate(s.h, &s.data));
    CK(cudaMemcpy(s.val, val_interleaved, sizeof(cuDoubleComplex) * nnz, cudaMemcpyHostToDevice));
    CS(cudssMatrixCreateCsr(&s.A, n, n, nnz, s.ptr, nullptr, s.idx, s.val, CUDSS_R_32I, CUDSS_R_32I, CUDSS_C_64F,
                            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CS(cudssMatrixCreateDn(&s.B, n, 1, n, s.tmp, CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    CS(cudssMatrixCreateDn(&s.X, n, 1, n, s.xbuf, CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    CS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS, s.cfg, s.data, s.A, s.X, s.B));
  } else {
    CK(cudaMemcpy(s.val, val_interleaved, sizeof(cuDoubleComplex) * nnz, cudaMemcpyHostToDevice));
  }
  CK(cudaMemcpy(s.r, r, sizeof(double) * n, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s.c, c, sizeof(double) * n, cudaMemcpyHostToDevice));
  CS(cudssExecute(s.h, CUDSS_PHASE_FACTORIZATION, s.cfg, s.data, s.A, s.X, s.B));
  CK(cudaStreamSynchronize(s.own));
  return 0;
}

extern "C" void jno_cudss_free(int64_t slot) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_slots.find(slot);
  if (it != g_slots.end()) {
    release(it->second);
    g_slots.erase(it);
  }
}

// The FFI handler: x = c .* B^{-1} (r .* b), on XLA's stream.
static ffi::Error SolveImpl(cudaStream_t stream, ffi::Buffer<ffi::C128> b, ffi::ResultBuffer<ffi::C128> x,
                            int64_t slot) {
  Slot *s;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_slots.find(slot);
    if (it == g_slots.end() || !it->second.h) return ffi::Error::InvalidArgument("jno_cudss_solve: empty slot");
    s = &it->second;
  }
  int64_t n = s->n;
  if ((int64_t)b.element_count() != n) return ffi::Error::InvalidArgument("jno_cudss_solve: size mismatch");
  auto *bp = reinterpret_cast<const cuDoubleComplex *>(b.typed_data());
  auto *xp = reinterpret_cast<cuDoubleComplex *>(x->typed_data());
  int threads = 256, blocks = (int)((n + threads - 1) / threads);
  cudaEventRecord(s->in, stream);          // the input is ready on XLA's stream...
  cudaStreamWaitEvent(s->own, s->in, 0);   // ...before cuDSS's stream touches it
  scale_in<<<blocks, threads, 0, s->own>>>(bp, s->r, s->tmp, n);
  if (cudssExecute(s->h, CUDSS_PHASE_SOLVE, s->cfg, s->data, s->A, s->X, s->B) != CUDSS_STATUS_SUCCESS)
    return ffi::Error::Internal("cudss solve");
  scale_out<<<blocks, threads, 0, s->own>>>(s->xbuf, s->c, xp, n);
  cudaEventRecord(s->out, s->own);
  cudaStreamWaitEvent(stream, s->out, 0);  // XLA's stream resumes after the result is written
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(jno_cudss_solve, SolveImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>()
                                  .Attr<int64_t>("slot"));


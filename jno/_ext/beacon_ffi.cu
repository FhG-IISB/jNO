// Progress beacon: a compiled loop drops a few scalars into a ring buffer in host memory, and a host thread
// reads them while the program runs. Nothing waits: the GPU writes through mapped pinned memory with a
// one-thread kernel and goes on; the CPU handler is a few stores. One ring per platform (each has its own
// emit counter).
//
// Slot protocol (a seqlock with one writer per slot): the payload is written, then a system-wide fence, then
// `seq` = the 1-based emit number. A reader takes slot (n-1) % R for emit n, reads seq, the payload and seq
// again, and keeps the payload only if both seq reads equal n. The emit number comes from a counter that
// lives where the writer runs (device memory on the GPU), so it is right under CUDA graphs too, where the
// host handler is not called once per launch.
//
// Built by jno/utils/beacon.py: nvcc when a GPU is present (both handlers), otherwise the host compiler
// (`-x c++`, the CPU handler only).

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>

#include "xla/ffi/api/ffi.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace ffi = xla::ffi;

constexpr int64_t kValues = 6;  // scalars per emit
constexpr int64_t kSlots = 4096;

struct alignas(64) Slot {
  volatile int64_t seq;
  int64_t tag;
  double v[kValues];
};

static Slot *g_ring = nullptr;      // CPU emits: plain host memory
static Slot *g_ring_gpu = nullptr;  // GPU emits: mapped pinned memory, host address (what the reader maps)
static std::mutex g_mu;
static std::atomic<int64_t> g_cpu_count{0};

#ifdef __CUDACC__
static Slot *g_ring_dev = nullptr;               // the same memory, as the device sees it
static unsigned long long *g_dev_count = nullptr;  // emit counter in device memory
#endif

static ffi::Error ensure_ring(bool gpu) {
  std::lock_guard<std::mutex> lk(g_mu);
#ifdef __CUDACC__
  if (gpu) {
    if (g_ring_dev) return ffi::Error::Success();
    void *p = nullptr;
    if (cudaHostAlloc(&p, sizeof(Slot) * kSlots, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
      return ffi::Error(ffi::ErrorCode::kResourceExhausted, "beacon: cudaHostAlloc failed");
    std::memset(p, 0, sizeof(Slot) * kSlots);
    void *d = nullptr;
    if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess)
      return ffi::Error(ffi::ErrorCode::kInternal, "beacon: cudaHostGetDevicePointer failed");
    if (cudaMalloc(&g_dev_count, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMemset(g_dev_count, 0, sizeof(unsigned long long)) != cudaSuccess)
      return ffi::Error(ffi::ErrorCode::kResourceExhausted, "beacon: device counter");
    g_ring_dev = static_cast<Slot *>(d);
    g_ring_gpu = static_cast<Slot *>(p);
    return ffi::Error::Success();
  }
#endif
  if (g_ring) return ffi::Error::Success();
  void *p = std::aligned_alloc(64, sizeof(Slot) * kSlots);
  if (!p) return ffi::Error(ffi::ErrorCode::kResourceExhausted, "beacon: host ring");
  std::memset(p, 0, sizeof(Slot) * kSlots);
  g_ring = static_cast<Slot *>(p);
  return ffi::Error::Success();
}

// ---- reader API (ctypes) -------------------------------------------------------------------------------
extern "C" void *jno_beacon_ring(int gpu) { return gpu ? g_ring_gpu : g_ring; }
extern "C" int64_t jno_beacon_slots() { return kSlots; }
extern "C" int64_t jno_beacon_values() { return kValues; }
extern "C" int jno_beacon_init_cpu() { return ensure_ring(false).success() ? 0 : 1; }
#ifdef __CUDACC__
extern "C" int jno_beacon_init_gpu() { return ensure_ring(true).success() ? 0 : 1; }
#endif

// ---- CPU handler ----------------------------------------------------------------------------------------
template <ffi::DataType T>
static ffi::Error emit_cpu(ffi::Buffer<T> v, ffi::ResultBuffer<ffi::S32> out, int64_t tag) {
  if (!g_ring) {
    ffi::Error e = ensure_ring(false);
    if (!e.success()) return e;
  }
  const int64_t n = g_cpu_count.fetch_add(1, std::memory_order_relaxed) + 1;
  Slot &s = g_ring[(n - 1) % kSlots];
  s.seq = 0;  // invalidate while the payload changes
  std::atomic_thread_fence(std::memory_order_release);
  const int64_t m = v.element_count() < kValues ? v.element_count() : kValues;
  for (int64_t i = 0; i < m; ++i) s.v[i] = static_cast<double>(v.typed_data()[i]);
  s.tag = tag;
  std::atomic_thread_fence(std::memory_order_release);
  s.seq = n;
  out->typed_data()[0] = 0;
  return ffi::Error::Success();
}
static ffi::Error EmitCpuD(ffi::Buffer<ffi::F64> v, ffi::ResultBuffer<ffi::S32> o, int64_t tag) { return emit_cpu(v, o, tag); }
static ffi::Error EmitCpuS(ffi::Buffer<ffi::F32> v, ffi::ResultBuffer<ffi::S32> o, int64_t tag) { return emit_cpu(v, o, tag); }

#define JNO_BEACON_CPU_BIND(T) ffi::Ffi::Bind().Arg<ffi::Buffer<T>>().Ret<ffi::Buffer<ffi::S32>>().Attr<int64_t>("tag")
XLA_FFI_DEFINE_HANDLER_SYMBOL(jno_beacon_emit_cpu_f64, EmitCpuD, JNO_BEACON_CPU_BIND(ffi::F64));
XLA_FFI_DEFINE_HANDLER_SYMBOL(jno_beacon_emit_cpu_f32, EmitCpuS, JNO_BEACON_CPU_BIND(ffi::F32));

// ---- GPU handler ----------------------------------------------------------------------------------------
#ifdef __CUDACC__
template <typename T>
__global__ void emit_kernel(const T *v, int64_t m, int64_t tag, Slot *ring, unsigned long long *count, int32_t *out) {
  const unsigned long long n = atomicAdd(count, 1ULL) + 1ULL;
  Slot *s = ring + (n - 1) % kSlots;
  s->seq = 0;
  __threadfence_system();
  for (int64_t i = 0; i < m; ++i) s->v[i] = static_cast<double>(v[i]);
  s->tag = tag;
  __threadfence_system();
  s->seq = static_cast<int64_t>(n);
  out[0] = 0;
}

template <ffi::DataType T>
static ffi::Error emit_gpu(cudaStream_t st, ffi::Buffer<T> v, ffi::ResultBuffer<ffi::S32> out, int64_t tag) {
  if (!g_ring_dev) {
    ffi::Error e = ensure_ring(true);
    if (!e.success()) return e;
  }
  const int64_t m = v.element_count() < kValues ? v.element_count() : kValues;
  emit_kernel<<<1, 1, 0, st>>>(v.typed_data(), m, tag, g_ring_dev, g_dev_count, out->typed_data());
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
  return ffi::Error::Success();
}
static ffi::Error EmitGpuD(cudaStream_t st, ffi::Buffer<ffi::F64> v, ffi::ResultBuffer<ffi::S32> o, int64_t tag) {
  return emit_gpu(st, v, o, tag);
}
static ffi::Error EmitGpuS(cudaStream_t st, ffi::Buffer<ffi::F32> v, ffi::ResultBuffer<ffi::S32> o, int64_t tag) {
  return emit_gpu(st, v, o, tag);
}

#define JNO_BEACON_GPU_BIND(T) \
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<T>>().Ret<ffi::Buffer<ffi::S32>>().Attr<int64_t>("tag")
XLA_FFI_DEFINE_HANDLER_SYMBOL(jno_beacon_emit_gpu_f64, EmitGpuD, JNO_BEACON_GPU_BIND(ffi::F64));
XLA_FFI_DEFINE_HANDLER_SYMBOL(jno_beacon_emit_gpu_f32, EmitGpuS, JNO_BEACON_GPU_BIND(ffi::F32));
#endif

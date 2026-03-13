// musa_assign_add_kernel.mu
// MUSA kernel for AssignAdd operation
// Performs in-place addition: ref = ref + value

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>

extern "C" {

// ============================================================================
// Float kernel
// ============================================================================
__global__ void AssignAddKernelFloat(float* ref, const float* value, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    ref[i] = ref[i] + value[i];
  }
}

// ============================================================================
// Double kernel
// ============================================================================
__global__ void AssignAddKernelDouble(double* ref, const double* value, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    ref[i] = ref[i] + value[i];
  }
}

// ============================================================================
// Half (float16) kernel - uses native MUSA half type
// ============================================================================
__global__ void AssignAddKernelHalf(half* ref, const half* value, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    ref[i] = ref[i] + value[i];
  }
}

// ============================================================================
// BFloat16 kernel - uses native MUSA bfloat16 type with float accumulation
// ============================================================================
__global__ void AssignAddKernelBFloat16(__mt_bfloat16* ref, 
                                        const __mt_bfloat16* value, 
                                        int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float ref_val = __bfloat162float(ref[i]);
    float value_val = __bfloat162float(value[i]);
    ref[i] = __float2bfloat16(ref_val + value_val);
  }
}

// ============================================================================
// Int32 kernel
// ============================================================================
__global__ void AssignAddKernelInt32(int* ref, const int* value, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    ref[i] = ref[i] + value[i];
  }
}

// ============================================================================
// Int64 kernel
// ============================================================================
__global__ void AssignAddKernelInt64(int64_t* ref, const int64_t* value, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    ref[i] = ref[i] + value[i];
  }
}

// ============================================================================
// Launcher functions - called from C++ code
// ============================================================================
#define DEFINE_ASSIGN_ADD_LAUNCHER(name, kernel, T) \
  void name(T* ref, const T* value, int64_t n, musaStream_t stream) { \
    if (n <= 0) return; \
    const int threads_per_block = 256; \
    const int blocks = static_cast<int>((n + threads_per_block - 1) / threads_per_block); \
    kernel<<<blocks, threads_per_block, 0, stream>>>(ref, value, n); \
  }

DEFINE_ASSIGN_ADD_LAUNCHER(LaunchAssignAddFloat, AssignAddKernelFloat, float)
DEFINE_ASSIGN_ADD_LAUNCHER(LaunchAssignAddDouble, AssignAddKernelDouble, double)
DEFINE_ASSIGN_ADD_LAUNCHER(LaunchAssignAddHalf, AssignAddKernelHalf, half)
DEFINE_ASSIGN_ADD_LAUNCHER(LaunchAssignAddBFloat16, AssignAddKernelBFloat16, __mt_bfloat16)
DEFINE_ASSIGN_ADD_LAUNCHER(LaunchAssignAddInt32, AssignAddKernelInt32, int)
DEFINE_ASSIGN_ADD_LAUNCHER(LaunchAssignAddInt64, AssignAddKernelInt64, int64_t)

#undef DEFINE_ASSIGN_ADD_LAUNCHER

}  // extern "C"
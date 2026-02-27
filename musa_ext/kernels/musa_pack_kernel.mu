#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>

template <typename T>
__global__ void PackKernel(const T** inputs, T* output, int num_inputs, 
                           int before_size, int after_size, int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < total_elements) {
    int after_idx = idx % after_size;
    int i = (idx / after_size) % num_inputs;
    int b = idx / (after_size * num_inputs);
    
    int in_idx = b * after_size + after_idx;
    output[idx] = inputs[i][in_idx];
  }
}

template <typename T>
__global__ void UnpackKernel(const T* input, T** outputs, int num_outputs, 
                             int before_size, int after_size, int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < total_elements) {
    int after_idx = idx % after_size;
    int i = (idx / after_size) % num_outputs;
    int b = idx / (after_size * num_outputs);
    
    int out_idx = b * after_size + after_idx;
    outputs[i][out_idx] = input[idx];
  }
}

extern "C" {

#define DEFINE_PACK_LAUNCHER(name, T) \
  void name(const T** inputs, T* output, int num_inputs, \
            int before_size, int after_size, int total_elements, musaStream_t stream) { \
    const int threads_per_block = 256; \
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block; \
    PackKernel<T><<<blocks, threads_per_block, 0, stream>>>( \
        inputs, output, num_inputs, before_size, after_size, total_elements); \
  }

#define DEFINE_UNPACK_LAUNCHER(name, T) \
  void name(const T* input, T** outputs, int num_outputs, \
            int before_size, int after_size, int total_elements, musaStream_t stream) { \
    const int threads_per_block = 256; \
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block; \
    UnpackKernel<T><<<blocks, threads_per_block, 0, stream>>>( \
        input, outputs, num_outputs, before_size, after_size, total_elements); \
  }

DEFINE_PACK_LAUNCHER(LaunchPackKernelFloat, float)
DEFINE_PACK_LAUNCHER(LaunchPackKernelDouble, double)
DEFINE_PACK_LAUNCHER(LaunchPackKernelHalf, half)
DEFINE_PACK_LAUNCHER(LaunchPackKernelBFloat16, __mt_bfloat16)
DEFINE_PACK_LAUNCHER(LaunchPackKernelInt32, int)
DEFINE_PACK_LAUNCHER(LaunchPackKernelInt64, long long)

DEFINE_UNPACK_LAUNCHER(LaunchUnpackKernelFloat, float)
DEFINE_UNPACK_LAUNCHER(LaunchUnpackKernelDouble, double)
DEFINE_UNPACK_LAUNCHER(LaunchUnpackKernelHalf, half)
DEFINE_UNPACK_LAUNCHER(LaunchUnpackKernelBFloat16, __mt_bfloat16)
DEFINE_UNPACK_LAUNCHER(LaunchUnpackKernelInt32, int)
DEFINE_UNPACK_LAUNCHER(LaunchUnpackKernelInt64, long long)

#undef DEFINE_PACK_LAUNCHER
#undef DEFINE_UNPACK_LAUNCHER

}  // extern "C"
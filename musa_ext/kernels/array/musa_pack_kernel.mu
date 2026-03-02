//
// MUSA Pack/Unpack Kernels
//
// 5 项优化:
//   1. axis=0 快速路径 (host 端 musaMemcpyAsync DtoD, 不走 kernel)
//   2. after_size=1 专用 kernel (消除 % after_size 除法)
//   3. 2D grid dispatch (大 before_size 时 blockIdx.y = before, 减少除法)
//   4. 向量化 kernel (float4/int4, 4x 带宽)
//   5. grid-stride loop (所有 kernel 均使用)

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <stdint.h>

// ============================================================================
// Pack Kernels
// ============================================================================

// 通用标量 kernel + grid-stride loop (优化5)
template <typename T>
__global__ void PackKernelScalar(const T** __restrict__ inputs,
                                 T* __restrict__ output,
                                 int num_inputs,
                                 int64_t before_size,
                                 int64_t after_size,
                                 int64_t total_elements) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; tid < total_elements; tid += stride) {
    const int64_t a = tid % after_size;
    const int64_t temp = tid / after_size;
    const int i = temp % num_inputs;
    const int64_t b = temp / num_inputs;
    output[tid] = inputs[i][b * after_size + a];
  }
}

// after_size==1 专用 kernel: 消除 % after_size (优化2 + 优化5)
template <typename T>
__global__ void PackKernelAfter1(const T** __restrict__ inputs,
                                 T* __restrict__ output,
                                 int num_inputs,
                                 int64_t total_elements) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; tid < total_elements; tid += stride) {
    const int i = tid % num_inputs;
    const int64_t b = tid / num_inputs;
    output[tid] = inputs[i][b];
  }
}

// 2D grid kernel: blockIdx.y = before 维度 (优化3 + 优化5)
template <typename T>
__global__ void PackKernel2D(const T** __restrict__ inputs,
                             T* __restrict__ output,
                             int num_inputs,
                             int64_t before_size,
                             int64_t after_size,
                             int64_t inner_size) {
  const int64_t b = blockIdx.y;
  if (b >= before_size) return;
  int64_t inner_tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t inner_stride = (int64_t)gridDim.x * blockDim.x;
  for (; inner_tid < inner_size; inner_tid += inner_stride) {
    const int i = inner_tid / after_size;
    const int64_t a = inner_tid % after_size;
    output[b * inner_size + inner_tid] = inputs[i][b * after_size + a];
  }
}

// 向量化 kernel: 用 VecT (float4/int4) 一次读写多个元素 (优化4 + 优化5)
template <typename T, typename VecT, int VecWidth>
__global__ void PackKernelVec(const T** __restrict__ inputs,
                              T* __restrict__ output,
                              int num_inputs,
                              int64_t before_size,
                              int64_t after_size_vec,
                              int64_t total_vec) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  VecT* vec_output = reinterpret_cast<VecT*>(output);
  for (; tid < total_vec; tid += stride) {
    const int64_t a = tid % after_size_vec;
    const int64_t temp = tid / after_size_vec;
    const int i = temp % num_inputs;
    const int64_t b = temp / num_inputs;
    vec_output[tid] =
        reinterpret_cast<const VecT*>(inputs[i])[b * after_size_vec + a];
  }
}

// ============================================================================
// Unpack Kernels (与 Pack 对称)
// ============================================================================

template <typename T>
__global__ void UnpackKernelScalar(const T* __restrict__ input,
                                   T** __restrict__ outputs,
                                   int num_outputs,
                                   int64_t before_size,
                                   int64_t after_size,
                                   int64_t total_elements) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; tid < total_elements; tid += stride) {
    const int64_t a = tid % after_size;
    const int64_t temp = tid / after_size;
    const int i = temp % num_outputs;
    const int64_t b = temp / num_outputs;
    outputs[i][b * after_size + a] = input[tid];
  }
}

template <typename T>
__global__ void UnpackKernelAfter1(const T* __restrict__ input,
                                   T** __restrict__ outputs,
                                   int num_outputs,
                                   int64_t total_elements) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; tid < total_elements; tid += stride) {
    const int i = tid % num_outputs;
    const int64_t b = tid / num_outputs;
    outputs[i][b] = input[tid];
  }
}

template <typename T>
__global__ void UnpackKernel2D(const T* __restrict__ input,
                               T** __restrict__ outputs,
                               int num_outputs,
                               int64_t before_size,
                               int64_t after_size,
                               int64_t inner_size) {
  const int64_t b = blockIdx.y;
  if (b >= before_size) return;
  int64_t inner_tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t inner_stride = (int64_t)gridDim.x * blockDim.x;
  for (; inner_tid < inner_size; inner_tid += inner_stride) {
    const int i = inner_tid / after_size;
    const int64_t a = inner_tid % after_size;
    outputs[i][b * after_size + a] = input[b * inner_size + inner_tid];
  }
}

template <typename T, typename VecT, int VecWidth>
__global__ void UnpackKernelVec(const T* __restrict__ input,
                                T** __restrict__ outputs,
                                int num_outputs,
                                int64_t before_size,
                                int64_t after_size_vec,
                                int64_t total_vec) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  const VecT* vec_input = reinterpret_cast<const VecT*>(input);
  for (; tid < total_vec; tid += stride) {
    const int64_t a = tid % after_size_vec;
    const int64_t temp = tid / after_size_vec;
    const int i = temp % num_outputs;
    const int64_t b = temp / num_outputs;
    reinterpret_cast<VecT*>(outputs[i])[b * after_size_vec + a] =
        vec_input[tid];
  }
}

// ============================================================================
// Launcher 辅助宏
// ============================================================================

extern "C" {

#define THREADS 256
#define MAX_BLOCKS 65535
#define CALC_BLOCKS(n) \
  (int)(min((int64_t)(((n) + THREADS - 1) / THREADS), (int64_t)MAX_BLOCKS))

// before_size 超过此阈值使用 2D grid
#define GRID_2D_THRESH 4

// ============================================================================
// Pack Launcher: 4 字节类型 (float, int32)
// 路径选择: after1 -> vec4 -> 2D -> scalar
// ============================================================================
#define DEFINE_PACK_LAUNCHER_4B(T, Name)                                       \
  void Name(const T** inputs, T* output, int num_inputs, int64_t before_size,  \
            int64_t after_size, int64_t total_elements, musaStream_t stream) {  \
    if (total_elements == 0) return;                                           \
    /* 优化2: after_size==1 专用 */                                              \
    if (after_size == 1) {                                                     \
      PackKernelAfter1<T><<<CALC_BLOCKS(total_elements), THREADS, 0,           \
                            stream>>>(inputs, output, num_inputs,              \
                                      total_elements);                         \
      return;                                                                  \
    }                                                                          \
    /* 优化4: float4 向量化 (after_size 能被 4 整除) */                            \
    if (after_size % 4 == 0) {                                                 \
      int64_t av = after_size / 4, tv = total_elements / 4;                    \
      PackKernelVec<T, float4, 4><<<CALC_BLOCKS(tv), THREADS, 0, stream>>>(   \
          inputs, output, num_inputs, before_size, av, tv);                    \
      return;                                                                  \
    }                                                                          \
    /* 优化3: 2D grid */                                                        \
    if (before_size > GRID_2D_THRESH) {                                        \
      int64_t inner = (int64_t)num_inputs * after_size;                        \
      dim3 grid(CALC_BLOCKS(inner),                                            \
                (int)min(before_size, (int64_t)MAX_BLOCKS));                    \
      PackKernel2D<T><<<grid, THREADS, 0, stream>>>(                           \
          inputs, output, num_inputs, before_size, after_size, inner);         \
      return;                                                                  \
    }                                                                          \
    /* 优化5: 通用标量 + grid-stride */                                           \
    PackKernelScalar<T><<<CALC_BLOCKS(total_elements), THREADS, 0, stream>>>(  \
        inputs, output, num_inputs, before_size, after_size, total_elements);  \
  }

// ============================================================================
// Pack Launcher: 8 字节类型 (double, int64)
// 向量化用 int4 (16B = 2 个 8B 元素)
// ============================================================================
#define DEFINE_PACK_LAUNCHER_8B(T, Name)                                       \
  void Name(const T** inputs, T* output, int num_inputs, int64_t before_size,  \
            int64_t after_size, int64_t total_elements, musaStream_t stream) {  \
    if (total_elements == 0) return;                                           \
    if (after_size == 1) {                                                     \
      PackKernelAfter1<T><<<CALC_BLOCKS(total_elements), THREADS, 0,           \
                            stream>>>(inputs, output, num_inputs,              \
                                      total_elements);                         \
      return;                                                                  \
    }                                                                          \
    if (after_size % 2 == 0) {                                                 \
      int64_t av = after_size / 2, tv = total_elements / 2;                    \
      PackKernelVec<T, int4, 2><<<CALC_BLOCKS(tv), THREADS, 0, stream>>>(     \
          inputs, output, num_inputs, before_size, av, tv);                    \
      return;                                                                  \
    }                                                                          \
    if (before_size > GRID_2D_THRESH) {                                        \
      int64_t inner = (int64_t)num_inputs * after_size;                        \
      dim3 grid(CALC_BLOCKS(inner),                                            \
                (int)min(before_size, (int64_t)MAX_BLOCKS));                    \
      PackKernel2D<T><<<grid, THREADS, 0, stream>>>(                           \
          inputs, output, num_inputs, before_size, after_size, inner);         \
      return;                                                                  \
    }                                                                          \
    PackKernelScalar<T><<<CALC_BLOCKS(total_elements), THREADS, 0, stream>>>(  \
        inputs, output, num_inputs, before_size, after_size, total_elements);  \
  }

// ============================================================================
// Pack Launcher: 2 字节类型 (half, bf16)
// 向量化: %8==0 用 float4 (16B=8×2B), %2==0 用 int (4B=2×2B)
// ============================================================================
#define DEFINE_PACK_LAUNCHER_2B(T, Name)                                       \
  void Name(const void** inputs, void* output, int num_inputs,                \
            int64_t before_size, int64_t after_size, int64_t total_elements,   \
            musaStream_t stream) {                                             \
    if (total_elements == 0) return;                                           \
    const T** ti = reinterpret_cast<const T**>(inputs);                        \
    T* to = reinterpret_cast<T*>(output);                                      \
    if (after_size == 1) {                                                     \
      PackKernelAfter1<T><<<CALC_BLOCKS(total_elements), THREADS, 0,           \
                            stream>>>(ti, to, num_inputs, total_elements);     \
      return;                                                                  \
    }                                                                          \
    if (after_size % 8 == 0) {                                                 \
      int64_t av = after_size / 8, tv = total_elements / 8;                    \
      PackKernelVec<T, float4, 8><<<CALC_BLOCKS(tv), THREADS, 0, stream>>>(   \
          ti, to, num_inputs, before_size, av, tv);                            \
      return;                                                                  \
    }                                                                          \
    if (after_size % 2 == 0) {                                                 \
      int64_t av = after_size / 2, tv = total_elements / 2;                    \
      PackKernelVec<T, int, 2><<<CALC_BLOCKS(tv), THREADS, 0, stream>>>(      \
          ti, to, num_inputs, before_size, av, tv);                            \
      return;                                                                  \
    }                                                                          \
    if (before_size > GRID_2D_THRESH) {                                        \
      int64_t inner = (int64_t)num_inputs * after_size;                        \
      dim3 grid(CALC_BLOCKS(inner),                                            \
                (int)min(before_size, (int64_t)MAX_BLOCKS));                    \
      PackKernel2D<T><<<grid, THREADS, 0, stream>>>(                           \
          ti, to, num_inputs, before_size, after_size, inner);                 \
      return;                                                                  \
    }                                                                          \
    PackKernelScalar<T><<<CALC_BLOCKS(total_elements), THREADS, 0, stream>>>( \
        ti, to, num_inputs, before_size, after_size, total_elements);          \
  }

// ============================================================================
// Unpack Launcher: 4 字节
// ============================================================================
#define DEFINE_UNPACK_LAUNCHER_4B(T, Name)                                     \
  void Name(const T* input, T** outputs, int num_outputs,                      \
            int64_t before_size, int64_t after_size, int64_t total_elements,   \
            musaStream_t stream) {                                             \
    if (total_elements == 0) return;                                           \
    if (after_size == 1) {                                                     \
      UnpackKernelAfter1<T><<<CALC_BLOCKS(total_elements), THREADS, 0,         \
                              stream>>>(input, outputs, num_outputs,            \
                                        total_elements);                       \
      return;                                                                  \
    }                                                                          \
    if (after_size % 4 == 0) {                                                 \
      int64_t av = after_size / 4, tv = total_elements / 4;                    \
      UnpackKernelVec<T, float4, 4><<<CALC_BLOCKS(tv), THREADS, 0,             \
                                      stream>>>(input, outputs, num_outputs,   \
                                                 before_size, av, tv);         \
      return;                                                                  \
    }                                                                          \
    if (before_size > GRID_2D_THRESH) {                                        \
      int64_t inner = (int64_t)num_outputs * after_size;                       \
      dim3 grid(CALC_BLOCKS(inner),                                            \
                (int)min(before_size, (int64_t)MAX_BLOCKS));                    \
      UnpackKernel2D<T><<<grid, THREADS, 0, stream>>>(                         \
          input, outputs, num_outputs, before_size, after_size, inner);        \
      return;                                                                  \
    }                                                                          \
    UnpackKernelScalar<T><<<CALC_BLOCKS(total_elements), THREADS, 0,           \
                            stream>>>(input, outputs, num_outputs, before_size, \
                                      after_size, total_elements);             \
  }

// ============================================================================
// Unpack Launcher: 8 字节
// ============================================================================
#define DEFINE_UNPACK_LAUNCHER_8B(T, Name)                                     \
  void Name(const T* input, T** outputs, int num_outputs,                      \
            int64_t before_size, int64_t after_size, int64_t total_elements,   \
            musaStream_t stream) {                                             \
    if (total_elements == 0) return;                                           \
    if (after_size == 1) {                                                     \
      UnpackKernelAfter1<T><<<CALC_BLOCKS(total_elements), THREADS, 0,         \
                              stream>>>(input, outputs, num_outputs,            \
                                        total_elements);                       \
      return;                                                                  \
    }                                                                          \
    if (after_size % 2 == 0) {                                                 \
      int64_t av = after_size / 2, tv = total_elements / 2;                    \
      UnpackKernelVec<T, int4, 2><<<CALC_BLOCKS(tv), THREADS, 0, stream>>>(   \
          input, outputs, num_outputs, before_size, av, tv);                   \
      return;                                                                  \
    }                                                                          \
    if (before_size > GRID_2D_THRESH) {                                        \
      int64_t inner = (int64_t)num_outputs * after_size;                       \
      dim3 grid(CALC_BLOCKS(inner),                                            \
                (int)min(before_size, (int64_t)MAX_BLOCKS));                    \
      UnpackKernel2D<T><<<grid, THREADS, 0, stream>>>(                         \
          input, outputs, num_outputs, before_size, after_size, inner);        \
      return;                                                                  \
    }                                                                          \
    UnpackKernelScalar<T><<<CALC_BLOCKS(total_elements), THREADS, 0,           \
                            stream>>>(input, outputs, num_outputs, before_size, \
                                      after_size, total_elements);             \
  }

// ============================================================================
// Unpack Launcher: 2 字节
// ============================================================================
#define DEFINE_UNPACK_LAUNCHER_2B(T, Name)                                     \
  void Name(const void* input, void** outputs, int num_outputs,               \
            int64_t before_size, int64_t after_size, int64_t total_elements,   \
            musaStream_t stream) {                                             \
    if (total_elements == 0) return;                                           \
    const T* ti = reinterpret_cast<const T*>(input);                           \
    T** to = reinterpret_cast<T**>(outputs);                                   \
    if (after_size == 1) {                                                     \
      UnpackKernelAfter1<T><<<CALC_BLOCKS(total_elements), THREADS, 0,         \
                              stream>>>(ti, to, num_outputs, total_elements);  \
      return;                                                                  \
    }                                                                          \
    if (after_size % 8 == 0) {                                                 \
      int64_t av = after_size / 8, tv = total_elements / 8;                    \
      UnpackKernelVec<T, float4, 8><<<CALC_BLOCKS(tv), THREADS, 0,             \
                                      stream>>>(ti, to, num_outputs,           \
                                                 before_size, av, tv);         \
      return;                                                                  \
    }                                                                          \
    if (after_size % 2 == 0) {                                                 \
      int64_t av = after_size / 2, tv = total_elements / 2;                    \
      UnpackKernelVec<T, int, 2><<<CALC_BLOCKS(tv), THREADS, 0, stream>>>(    \
          ti, to, num_outputs, before_size, av, tv);                           \
      return;                                                                  \
    }                                                                          \
    if (before_size > GRID_2D_THRESH) {                                        \
      int64_t inner = (int64_t)num_outputs * after_size;                       \
      dim3 grid(CALC_BLOCKS(inner),                                            \
                (int)min(before_size, (int64_t)MAX_BLOCKS));                    \
      UnpackKernel2D<T><<<grid, THREADS, 0, stream>>>(                         \
          ti, to, num_outputs, before_size, after_size, inner);                \
      return;                                                                  \
    }                                                                          \
    UnpackKernelScalar<T><<<CALC_BLOCKS(total_elements), THREADS, 0,           \
                            stream>>>(ti, to, num_outputs, before_size,        \
                                      after_size, total_elements);             \
  }

// ============================================================================
// 实例化
// ============================================================================

DEFINE_PACK_LAUNCHER_4B(float, LaunchPackKernelFloat)
DEFINE_PACK_LAUNCHER_4B(int, LaunchPackKernelInt32)
DEFINE_PACK_LAUNCHER_8B(double, LaunchPackKernelDouble)
DEFINE_PACK_LAUNCHER_8B(long long, LaunchPackKernelInt64)
DEFINE_PACK_LAUNCHER_2B(__half, LaunchPackKernelHalf)
DEFINE_PACK_LAUNCHER_2B(__mt_bfloat16, LaunchPackKernelBFloat16)

DEFINE_UNPACK_LAUNCHER_4B(float, LaunchUnpackKernelFloat)
DEFINE_UNPACK_LAUNCHER_4B(int, LaunchUnpackKernelInt32)
DEFINE_UNPACK_LAUNCHER_8B(double, LaunchUnpackKernelDouble)
DEFINE_UNPACK_LAUNCHER_8B(long long, LaunchUnpackKernelInt64)
DEFINE_UNPACK_LAUNCHER_2B(__half, LaunchUnpackKernelHalf)
DEFINE_UNPACK_LAUNCHER_2B(__mt_bfloat16, LaunchUnpackKernelBFloat16)

#undef DEFINE_PACK_LAUNCHER_4B
#undef DEFINE_PACK_LAUNCHER_8B
#undef DEFINE_PACK_LAUNCHER_2B
#undef DEFINE_UNPACK_LAUNCHER_4B
#undef DEFINE_UNPACK_LAUNCHER_8B
#undef DEFINE_UNPACK_LAUNCHER_2B
#undef THREADS
#undef MAX_BLOCKS
#undef CALC_BLOCKS
#undef GRID_2D_THRESH

}  // extern "C"
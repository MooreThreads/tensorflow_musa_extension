#include <musa_runtime.h>
#include <musa_fp16.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/stream_executor/stream.h"
#include "../utils/musa_guarded_philox_random.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

using random::PhiloxRandom;
using random::TruncatedNormalDistribution;

__device__ __forceinline__ void StoreFloat(float* p, double v) {
  *p = static_cast<float>(v);
}

__device__ __forceinline__ void StoreFloat(double* p, double v) { *p = v; }

__device__ __forceinline__ void StoreFloat(Eigen::half* p, double v) {
  __half h = __double2half(v);
  *reinterpret_cast<__half*>(p) = h;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, double v) {
  union FloatCaster {
    float f;
    uint32_t bits;
  } caster;
  caster.f = static_cast<float>(v);
  uint16_t b_val = static_cast<uint16_t>(caster.bits >> 16);
  *reinterpret_cast<uint16_t*>(p) = b_val;
}

// 每个线程处理 kGroupSize 个元素（TruncatedNormalDistribution 每次生成 4 个 float）
template <typename T, int kBlockSize = 256>
__global__ void __launch_bounds__(kBlockSize)
PhiloxTruncatedNormalKernel(
    const uint64_t num_elements,
    const PhiloxRandom base_gen,
    TruncatedNormalDistribution<PhiloxRandom> dist,
    T* __restrict__ data) {
  using TruncatedDist = TruncatedNormalDistribution<PhiloxRandom>;
  constexpr int kGroupSize = TruncatedDist::kResultElementCount;

  const uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t thread_count = gridDim.x * blockDim.x;
  uint64_t group_index = thread_id;

  while (group_index * kGroupSize < num_elements) {
    PhiloxRandom gen = base_gen;
    gen.Skip(group_index);

    auto samples = dist(&gen);

    for (int i = 0; i < kGroupSize; ++i) {
      const uint64_t index = group_index * kGroupSize + i;
      if (index < num_elements) {
        StoreFloat(&data[index], samples[i]);
      }
    }
    group_index += thread_count;
  }
}

template <typename T>
void LaunchPhiloxTruncatedNormal(
    musaStream_t stream,
    T* data,
    uint64_t num_elements,
    const PhiloxRandom& philox) {
  using TruncatedDist = TruncatedNormalDistribution<PhiloxRandom>;
  TruncatedDist dist;
  constexpr int kBlockSize = 256;
  constexpr int kGroupSize = TruncatedDist::kResultElementCount;
  const uint64_t num_groups = (num_elements + kGroupSize - 1) / kGroupSize;
  const int num_blocks = (num_groups + kBlockSize - 1) / kBlockSize;

  PhiloxTruncatedNormalKernel<T><<<num_blocks, kBlockSize, 0, stream>>>(
      num_elements, philox, dist, data);
}

// 显式实例化
template void LaunchPhiloxTruncatedNormal<float>(
  musaStream_t, float*, uint64_t, const tensorflow::random::PhiloxRandom&);
template void LaunchPhiloxTruncatedNormal<double>(
  musaStream_t, double*, uint64_t, const tensorflow::random::PhiloxRandom&);
template void LaunchPhiloxTruncatedNormal<Eigen::half>(
  musaStream_t, Eigen::half*, uint64_t, const tensorflow::random::PhiloxRandom&);
template void LaunchPhiloxTruncatedNormal<Eigen::bfloat16>(
  musaStream_t, Eigen::bfloat16*, uint64_t, const tensorflow::random::PhiloxRandom&);
}  // namespace musa
}  // namespace tensorflow

#include <musa_fp16.h>
#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"

#include "../utils/musa_guarded_philox_random.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/stream.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

// --------------------- utils ---------------------
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

// ---------------------- Universal Normal Kernel ----------------------
template <typename T, typename DIST_TYPE, int kBlockSize = 256>
__global__ void __launch_bounds__(kBlockSize)
    PhiloxNormalKernel(const uint64_t num_elements,
                       const random::PhiloxRandom base_gen, DIST_TYPE dist,
                       T* __restrict__ data) {
  constexpr int kGroupSize = DIST_TYPE::kResultElementCount;
  const uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t thread_count = gridDim.x * blockDim.x;
  uint64_t group_index = thread_id;

  while (group_index * kGroupSize < num_elements) {
    random::PhiloxRandom gen = base_gen;
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

template <typename T, typename DIST_TYPE>
void LaunchPhiloxNormalKernel(musaStream_t stream, T* data,
                              uint64_t num_elements,
                              const random::PhiloxRandom& philox,
                              const DIST_TYPE& dist) {
  constexpr int kBlockSize = 256;
  constexpr int kGroupSize = DIST_TYPE::kResultElementCount;
  const uint64_t num_groups = (num_elements + kGroupSize - 1) / kGroupSize;
  const int num_blocks = (num_groups + kBlockSize - 1) / kBlockSize;

  PhiloxNormalKernel<T, DIST_TYPE>
      <<<num_blocks, kBlockSize, 0, stream>>>(num_elements, philox, dist, data);
}

// --------------------- Random Normal Kernels ---------------------
template void LaunchPhiloxNormalKernel<float>(
    musaStream_t, float*, uint64_t, const random::PhiloxRandom&,
    const random::NormalDistribution<random::PhiloxRandom>&);
template void LaunchPhiloxNormalKernel<double>(
    musaStream_t, double*, uint64_t, const random::PhiloxRandom&,
    const random::NormalDistribution<random::PhiloxRandom>&);
template void LaunchPhiloxNormalKernel<Eigen::half>(
    musaStream_t, Eigen::half*, uint64_t, const random::PhiloxRandom&,
    const random::NormalDistribution<random::PhiloxRandom>&);
template void LaunchPhiloxNormalKernel<Eigen::bfloat16>(
    musaStream_t, Eigen::bfloat16*, uint64_t, const random::PhiloxRandom&,
    const random::NormalDistribution<random::PhiloxRandom>&);

// --------------------- Truncated Normal Kernels ---------------------
template void LaunchPhiloxNormalKernel<float>(
    musaStream_t, float*, uint64_t, const random::PhiloxRandom&,
    const random::TruncatedNormalDistribution<random::PhiloxRandom>&);
template void LaunchPhiloxNormalKernel<double>(
    musaStream_t, double*, uint64_t, const random::PhiloxRandom&,
    const random::TruncatedNormalDistribution<random::PhiloxRandom>&);
template void LaunchPhiloxNormalKernel<Eigen::half>(
    musaStream_t, Eigen::half*, uint64_t, const random::PhiloxRandom&,
    const random::TruncatedNormalDistribution<random::PhiloxRandom>&);
template void LaunchPhiloxNormalKernel<Eigen::bfloat16>(
    musaStream_t, Eigen::bfloat16*, uint64_t, const random::PhiloxRandom&,
    const random::TruncatedNormalDistribution<random::PhiloxRandom>&);

}  // namespace musa
}  // namespace tensorflow
#include <musa_bf16.h>
#include <musa_runtime.h>

#include <stdint.h>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

__global__ void ResourceApplyGradientDescentFloatKernel(
    float* __restrict__ var, const float* __restrict__ grad, float alpha,
    int64_t n) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    var[i] -= alpha * grad[i];
  }
}

__global__ void ResourceApplyGradientDescentFloatBFloat16Kernel(
    float* __restrict__ var, const __mt_bfloat16* __restrict__ grad,
    float alpha, int64_t n) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    var[i] -= alpha * __bfloat162float(grad[i]);
  }
}

__global__ void GroupedResourceApplyGradientDescentFloatBFloat16Kernel(
    const uint64_t* __restrict__ var_ptrs,
    const uint64_t* __restrict__ grad_ptrs,
    const float* __restrict__ alphas,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ block_offsets, int num_tensors) {
  const int block = blockIdx.x;
  int lo = 0;
  int hi = num_tensors;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) / 2;
    if (block_offsets[mid] <= block) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const int tensor_idx = lo;
  const int64_t n = sizes[tensor_idx];
  const int64_t local_block = block - block_offsets[tensor_idx];
  const int64_t blocks_for_tensor =
      block_offsets[tensor_idx + 1] - block_offsets[tensor_idx];
  float* __restrict__ var =
      reinterpret_cast<float*>(var_ptrs[tensor_idx]);
  const __mt_bfloat16* __restrict__ grad =
      reinterpret_cast<const __mt_bfloat16*>(grad_ptrs[tensor_idx]);
  const float alpha = alphas[tensor_idx];

  for (int64_t i = local_block * blockDim.x + threadIdx.x; i < n;
       i += blocks_for_tensor * blockDim.x) {
    var[i] -= alpha * __bfloat162float(grad[i]);
  }
}

__global__ void GroupedResourceApplyGradientDescentFloatKernel(
    const uint64_t* __restrict__ var_ptrs,
    const uint64_t* __restrict__ grad_ptrs,
    const float* __restrict__ alphas,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ block_offsets, int num_tensors) {
  const int block = blockIdx.x;
  int lo = 0;
  int hi = num_tensors;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) / 2;
    if (block_offsets[mid] <= block) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const int tensor_idx = lo;
  const int64_t n = sizes[tensor_idx];
  const int64_t local_block = block - block_offsets[tensor_idx];
  const int64_t blocks_for_tensor =
      block_offsets[tensor_idx + 1] - block_offsets[tensor_idx];
  float* __restrict__ var =
      reinterpret_cast<float*>(var_ptrs[tensor_idx]);
  const float* __restrict__ grad =
      reinterpret_cast<const float*>(grad_ptrs[tensor_idx]);
  const float alpha = alphas[tensor_idx];

  for (int64_t i = local_block * blockDim.x + threadIdx.x; i < n;
       i += blocks_for_tensor * blockDim.x) {
    var[i] -= alpha * grad[i];
  }
}

int BlocksForNumElements(int64_t n) {
  int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  if (blocks < 1) return 1;
  return blocks > kMaxBlocks ? kMaxBlocks : blocks;
}

}  // namespace

extern "C" {

void LaunchResourceApplyGradientDescentFloat(
    float* var, const float* grad, float alpha, int64_t n,
    musaStream_t stream) {
  if (n <= 0) return;
  ResourceApplyGradientDescentFloatKernel<<<BlocksForNumElements(n), kThreads,
                                            0, stream>>>(var, grad, alpha, n);
}

void LaunchResourceApplyGradientDescentFloatBFloat16(
    float* var, const void* grad, float alpha, int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  ResourceApplyGradientDescentFloatBFloat16Kernel<<<BlocksForNumElements(n),
                                                    kThreads, 0, stream>>>(
      var, reinterpret_cast<const __mt_bfloat16*>(grad), alpha, n);
}

void LaunchGroupedResourceApplyGradientDescentFloatBFloat16(
    const uint64_t* var_ptrs, const uint64_t* grad_ptrs,
    const float* alphas, const int64_t* sizes, const int64_t* block_offsets,
    int num_tensors, int total_blocks, musaStream_t stream) {
  if (num_tensors <= 0 || total_blocks <= 0) return;
  GroupedResourceApplyGradientDescentFloatBFloat16Kernel<<<total_blocks,
                                                           kThreads, 0,
                                                           stream>>>(
      var_ptrs, grad_ptrs, alphas, sizes, block_offsets, num_tensors);
}

void LaunchGroupedResourceApplyGradientDescentFloat(
    const uint64_t* var_ptrs, const uint64_t* grad_ptrs,
    const float* alphas, const int64_t* sizes, const int64_t* block_offsets,
    int num_tensors, int total_blocks, musaStream_t stream) {
  if (num_tensors <= 0 || total_blocks <= 0) return;
  GroupedResourceApplyGradientDescentFloatKernel<<<total_blocks, kThreads, 0,
                                                   stream>>>(
      var_ptrs, grad_ptrs, alphas, sizes, block_offsets, num_tensors);
}

}  // extern "C"

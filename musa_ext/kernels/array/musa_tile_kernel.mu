#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

namespace {

constexpr int kMaxTileDims = 16;
constexpr int kThreads = 256;

struct TileShapeInfo {
  int dims;
  int64_t input_dims[kMaxTileDims];
  int64_t output_dims[kMaxTileDims];
  int64_t input_strides[kMaxTileDims];
};

template <typename T>
__global__ void TileKernel(const T* __restrict__ input, T* __restrict__ output,
                           int64_t total, TileShapeInfo info) {
  int64_t out_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (; out_index < total; out_index += stride) {
    int64_t remaining = out_index;
    int64_t input_index = 0;

    for (int dim = info.dims - 1; dim >= 0; --dim) {
      const int64_t output_dim = info.output_dims[dim];
      const int64_t coord = remaining % output_dim;
      remaining /= output_dim;
      input_index += (coord % info.input_dims[dim]) * info.input_strides[dim];
    }

    output[out_index] = input[input_index];
  }
}

int BlocksFor(int64_t total) {
  const int64_t blocks = (total + kThreads - 1) / kThreads;
  return blocks > 4096 ? 4096 : static_cast<int>(blocks);
}

template <typename T>
void LaunchTileTyped(const T* input, T* output, int64_t total,
                     TileShapeInfo info, musaStream_t stream) {
  if (total <= 0) return;
  TileKernel<T><<<BlocksFor(total), kThreads, 0, stream>>>(input, output, total,
                                                           info);
}

}  // namespace

extern "C" {

void LaunchTileFloat(const float* input, float* output, int64_t total,
                     TileShapeInfo info, musaStream_t stream) {
  LaunchTileTyped(input, output, total, info, stream);
}

void LaunchTileDouble(const double* input, double* output, int64_t total,
                      TileShapeInfo info, musaStream_t stream) {
  LaunchTileTyped(input, output, total, info, stream);
}

void LaunchTileInt32(const int32_t* input, int32_t* output, int64_t total,
                     TileShapeInfo info, musaStream_t stream) {
  LaunchTileTyped(input, output, total, info, stream);
}

void LaunchTileInt64(const int64_t* input, int64_t* output, int64_t total,
                     TileShapeInfo info, musaStream_t stream) {
  LaunchTileTyped(input, output, total, info, stream);
}

void LaunchTileBool(const bool* input, bool* output, int64_t total,
                    TileShapeInfo info, musaStream_t stream) {
  LaunchTileTyped(input, output, total, info, stream);
}

void LaunchTileHalf(const void* input, void* output, int64_t total,
                    TileShapeInfo info, musaStream_t stream) {
  LaunchTileTyped(reinterpret_cast<const half*>(input),
                  reinterpret_cast<half*>(output), total, info, stream);
}

void LaunchTileBFloat16(const void* input, void* output, int64_t total,
                        TileShapeInfo info, musaStream_t stream) {
  LaunchTileTyped(reinterpret_cast<const __mt_bfloat16*>(input),
                  reinterpret_cast<__mt_bfloat16*>(output), total, info,
                  stream);
}

}  // extern "C"

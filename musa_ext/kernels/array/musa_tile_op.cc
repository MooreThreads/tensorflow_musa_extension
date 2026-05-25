#include <mudnn.h>

#include <array>
#include <cstdint>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {
namespace {

constexpr int kMaxTileDims = 16;

struct TileShapeInfo {
  int dims;
  int64_t input_dims[kMaxTileDims];
  int64_t output_dims[kMaxTileDims];
  int64_t input_strides[kMaxTileDims];
};

extern "C" {
void LaunchTileFloat(const float* input, float* output, int64_t total,
                     TileShapeInfo info, musaStream_t stream);
void LaunchTileDouble(const double* input, double* output, int64_t total,
                      TileShapeInfo info, musaStream_t stream);
void LaunchTileInt32(const int32_t* input, int32_t* output, int64_t total,
                     TileShapeInfo info, musaStream_t stream);
void LaunchTileInt64(const int64_t* input, int64_t* output, int64_t total,
                     TileShapeInfo info, musaStream_t stream);
void LaunchTileBool(const bool* input, bool* output, int64_t total,
                    TileShapeInfo info, musaStream_t stream);
void LaunchTileHalf(const void* input, void* output, int64_t total,
                    TileShapeInfo info, musaStream_t stream);
void LaunchTileBFloat16(const void* input, void* output, int64_t total,
                        TileShapeInfo info, musaStream_t stream);
}

template <typename T>
struct TileLauncher;

template <>
struct TileLauncher<float> {
  static void Run(const Tensor& input, Tensor* output, int64_t total,
                  TileShapeInfo info, musaStream_t stream) {
    LaunchTileFloat(input.flat<float>().data(), output->flat<float>().data(),
                    total, info, stream);
  }
};

template <>
struct TileLauncher<double> {
  static void Run(const Tensor& input, Tensor* output, int64_t total,
                  TileShapeInfo info, musaStream_t stream) {
    LaunchTileDouble(input.flat<double>().data(), output->flat<double>().data(),
                     total, info, stream);
  }
};

template <>
struct TileLauncher<int32> {
  static void Run(const Tensor& input, Tensor* output, int64_t total,
                  TileShapeInfo info, musaStream_t stream) {
    LaunchTileInt32(input.flat<int32>().data(), output->flat<int32>().data(),
                    total, info, stream);
  }
};

template <>
struct TileLauncher<int64> {
  static void Run(const Tensor& input, Tensor* output, int64_t total,
                  TileShapeInfo info, musaStream_t stream) {
    LaunchTileInt64(input.flat<int64>().data(), output->flat<int64>().data(),
                    total, info, stream);
  }
};

template <>
struct TileLauncher<bool> {
  static void Run(const Tensor& input, Tensor* output, int64_t total,
                  TileShapeInfo info, musaStream_t stream) {
    LaunchTileBool(input.flat<bool>().data(), output->flat<bool>().data(), total,
                   info, stream);
  }
};

template <>
struct TileLauncher<Eigen::half> {
  static void Run(const Tensor& input, Tensor* output, int64_t total,
                  TileShapeInfo info, musaStream_t stream) {
    LaunchTileHalf(input.tensor_data().data(),
                   const_cast<char*>(output->tensor_data().data()), total, info,
                   stream);
  }
};

template <>
struct TileLauncher<bfloat16> {
  static void Run(const Tensor& input, Tensor* output, int64_t total,
                  TileShapeInfo info, musaStream_t stream) {
    LaunchTileBFloat16(input.tensor_data().data(),
                       const_cast<char*>(output->tensor_data().data()), total,
                       info, stream);
  }
};

template <typename T, typename Tmultiples>
class MusaTileOp : public MusaOpKernel {
 public:
  explicit MusaTileOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& multiples = context->input(1);

    const int input_dims = input.dims();
    OP_REQUIRES(context, TensorShapeUtils::IsVector(multiples.shape()),
                errors::InvalidArgument("multiples must be a 1-D vector"));
    OP_REQUIRES(context, multiples.NumElements() == input_dims,
                errors::InvalidArgument(
                    "Expected multiples argument to be a vector of length ",
                    input_dims, " but got length ", multiples.NumElements()));
    OP_REQUIRES(context, input_dims <= kMaxTileDims,
                errors::InvalidArgument("Tile rank ", input_dims,
                                        " exceeds supported rank ",
                                        kMaxTileDims));

    const Tmultiples* m_data = multiples.flat<Tmultiples>().data();

    TensorShape output_shape;
    bool need_tile = false;
    bool can_use_mudnn_broadcast_tile = true;
    TileShapeInfo info = {};
    info.dims = input_dims;

    int64_t stride = 1;
    for (int i = input_dims - 1; i >= 0; --i) {
      info.input_strides[i] = stride;
      stride *= input.dim_size(i);
    }

    for (int i = 0; i < input_dims; ++i) {
      const Tmultiples multiple = m_data[i];
      OP_REQUIRES(context, multiple >= 0,
                  errors::InvalidArgument("Expected multiples[", i,
                                          "] >= 0, but got ", multiple));
      const int64_t input_dim = input.dim_size(i);
      const int64_t output_dim = input_dim * static_cast<int64_t>(multiple);
      output_shape.AddDim(output_dim);
      info.input_dims[i] = input_dim;
      info.output_dims[i] = output_dim;
      if (multiple != 1) {
        need_tile = true;
        if (input_dim != 1) can_use_mudnn_broadcast_tile = false;
      }
    }

    if (input_dims == 0 || !need_tile) {
      context->set_output(0, input);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    const int64_t total = output->NumElements();
    if (total == 0) return;

    if (can_use_mudnn_broadcast_tile) {
      auto& h = GetHandleByCtx(context);
      auto in_mt = CreateMTensor(input);
      auto out_mt = CreateMTensor(*output);

      std::array<int64_t, kMaxTileDims> b_dims;
      std::array<int64_t, kMaxTileDims> b_strides;
      std::array<int64_t, kMaxTileDims> orig_strides;
      int64_t stride = 1;

      for (int i = input_dims - 1; i >= 0; --i) {
        orig_strides[i] = input.dim_size(i) == 1 ? 0 : stride;
        stride *= input.dim_size(i);
      }
      for (int i = 0; i < input_dims; ++i) {
        b_dims[i] = output_shape.dim_size(i);
        b_strides[i] = input.dim_size(i) == 1 ? 0 : orig_strides[i];
      }

      MTOP_CHECK_OK(in_mt.SetNdInfo(input_dims, b_dims.data(), b_strides.data()),
                    "Tile SetNdInfo", context);
      ::musa::dnn::Permute op;
      MTOP_CHECK_OK_RUN(op.Run(h, out_mt, in_mt), "Permute Run for Tile", context);
      return;
    }

    TileLauncher<T>::Run(input, output, total, info, GetMusaStreamByCtx(context));
  }
};

#define REGISTER_MUSA_TILE_ALL_TYPES(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_MTGPU)                \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          MusaTileOp<type, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_MTGPU)                \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          MusaTileOp<type, int64>);

REGISTER_MUSA_TILE_ALL_TYPES(float);
REGISTER_MUSA_TILE_ALL_TYPES(Eigen::half);
REGISTER_MUSA_TILE_ALL_TYPES(bfloat16);
REGISTER_MUSA_TILE_ALL_TYPES(double);
REGISTER_MUSA_TILE_ALL_TYPES(int32);
REGISTER_MUSA_TILE_ALL_TYPES(int64);
REGISTER_MUSA_TILE_ALL_TYPES(bool);

#undef REGISTER_MUSA_TILE_ALL_TYPES

}  // namespace
}  // namespace musa
}  // namespace tensorflow

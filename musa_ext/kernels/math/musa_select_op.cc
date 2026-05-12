#include <mudnn.h>

#include <cstdint>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace musa {

namespace {

extern "C" {
void LaunchMusaSelectSameShapeFloat(const void* cond, const void* then_t,
                                    const void* else_t, void* output,
                                    int64_t size, musaStream_t stream);
void LaunchMusaSelectScalarCondFloat(const void* cond, const void* then_t,
                                     const void* else_t, void* output,
                                     int64_t size, musaStream_t stream);
void LaunchMusaSelectRank1CondFloat(const void* cond, const void* then_t,
                                    const void* else_t, void* output,
                                    int64_t size, int64_t inner_size,
                                    musaStream_t stream);
void LaunchMusaSelectSameShapeHalf(const void* cond, const void* then_t,
                                   const void* else_t, void* output,
                                   int64_t size, musaStream_t stream);
void LaunchMusaSelectScalarCondHalf(const void* cond, const void* then_t,
                                    const void* else_t, void* output,
                                    int64_t size, musaStream_t stream);
void LaunchMusaSelectRank1CondHalf(const void* cond, const void* then_t,
                                   const void* else_t, void* output,
                                   int64_t size, int64_t inner_size,
                                   musaStream_t stream);
void LaunchMusaSelectSameShapeBFloat16(const void* cond, const void* then_t,
                                       const void* else_t, void* output,
                                       int64_t size, musaStream_t stream);
void LaunchMusaSelectScalarCondBFloat16(const void* cond, const void* then_t,
                                        const void* else_t, void* output,
                                        int64_t size, musaStream_t stream);
void LaunchMusaSelectRank1CondBFloat16(const void* cond, const void* then_t,
                                       const void* else_t, void* output,
                                       int64_t size, int64_t inner_size,
                                       musaStream_t stream);
void LaunchMusaSelectSameShapeInt32(const void* cond, const void* then_t,
                                    const void* else_t, void* output,
                                    int64_t size, musaStream_t stream);
void LaunchMusaSelectScalarCondInt32(const void* cond, const void* then_t,
                                     const void* else_t, void* output,
                                     int64_t size, musaStream_t stream);
void LaunchMusaSelectRank1CondInt32(const void* cond, const void* then_t,
                                    const void* else_t, void* output,
                                    int64_t size, int64_t inner_size,
                                    musaStream_t stream);
void LaunchMusaSelectSameShapeInt64(const void* cond, const void* then_t,
                                    const void* else_t, void* output,
                                    int64_t size, musaStream_t stream);
void LaunchMusaSelectScalarCondInt64(const void* cond, const void* then_t,
                                     const void* else_t, void* output,
                                     int64_t size, musaStream_t stream);
void LaunchMusaSelectRank1CondInt64(const void* cond, const void* then_t,
                                    const void* else_t, void* output,
                                    int64_t size, int64_t inner_size,
                                    musaStream_t stream);
void LaunchMusaSelectSameShapeBool(const void* cond, const void* then_t,
                                   const void* else_t, void* output,
                                   int64_t size, musaStream_t stream);
void LaunchMusaSelectScalarCondBool(const void* cond, const void* then_t,
                                    const void* else_t, void* output,
                                    int64_t size, musaStream_t stream);
void LaunchMusaSelectRank1CondBool(const void* cond, const void* then_t,
                                   const void* else_t, void* output,
                                   int64_t size, int64_t inner_size,
                                   musaStream_t stream);
}

enum class SelectFastPathResult {
  kNotHandled = 0,
  kLaunched,
  kFailed,
};

bool ShouldUseSelectCustomKernelFastPath(const Tensor& cond,
                                         const Tensor& then_t,
                                         const Tensor& else_t,
                                         const TensorShape& output_shape,
                                         bool use_legacy_broadcast);

bool SameShape(const TensorShape& lhs, const TensorShape& rhs) {
  if (lhs.dims() != rhs.dims()) return false;
  for (int i = 0; i < lhs.dims(); ++i) {
    if (lhs.dim_size(i) != rhs.dim_size(i)) return false;
  }
  return true;
}

template <typename T>
struct SelectFastPathLauncher {
  static constexpr bool kSupported = false;
  static void SameShape(const void*, const void*, const void*, void*, int64_t,
                        musaStream_t) {}
  static void ScalarCond(const void*, const void*, const void*, void*, int64_t,
                         musaStream_t) {}
  static void Rank1Cond(const void*, const void*, const void*, void*, int64_t,
                        int64_t, musaStream_t) {}
};

#define DEFINE_SELECT_FAST_PATH_LAUNCHER(T, suffix)                          \
  template <>                                                                \
  struct SelectFastPathLauncher<T> {                                         \
    static constexpr bool kSupported = true;                                 \
    static void SameShape(const void* cond, const void* then_t,              \
                          const void* else_t, void* output, int64_t size,    \
                          musaStream_t stream) {                             \
      LaunchMusaSelectSameShape##suffix(cond, then_t, else_t, output, size,  \
                                        stream);                             \
    }                                                                        \
    static void ScalarCond(const void* cond, const void* then_t,             \
                           const void* else_t, void* output, int64_t size,   \
                           musaStream_t stream) {                            \
      LaunchMusaSelectScalarCond##suffix(cond, then_t, else_t, output, size, \
                                         stream);                            \
    }                                                                        \
    static void Rank1Cond(const void* cond, const void* then_t,              \
                          const void* else_t, void* output, int64_t size,    \
                          int64_t inner_size, musaStream_t stream) {         \
      LaunchMusaSelectRank1Cond##suffix(cond, then_t, else_t, output, size,  \
                                        inner_size, stream);                 \
    }                                                                        \
  };

DEFINE_SELECT_FAST_PATH_LAUNCHER(float, Float)
DEFINE_SELECT_FAST_PATH_LAUNCHER(Eigen::half, Half)
DEFINE_SELECT_FAST_PATH_LAUNCHER(Eigen::bfloat16, BFloat16)
DEFINE_SELECT_FAST_PATH_LAUNCHER(int32, Int32)
DEFINE_SELECT_FAST_PATH_LAUNCHER(int64, Int64)
DEFINE_SELECT_FAST_PATH_LAUNCHER(bool, Bool)

#undef DEFINE_SELECT_FAST_PATH_LAUNCHER

bool ShouldUseSelectCustomKernelFastPath(const Tensor& cond,
                                         const Tensor& then_t,
                                         const Tensor& else_t,
                                         const TensorShape& output_shape,
                                         bool use_legacy_broadcast) {
  if (!SameShape(then_t.shape(), output_shape) ||
      !SameShape(else_t.shape(), output_shape)) {
    return false;
  }
  if (SameShape(cond.shape(), output_shape) || cond.NumElements() == 1) {
    return true;
  }
  return use_legacy_broadcast && cond.dims() == 1 && output_shape.dims() > 1 &&
         cond.dim_size(0) == output_shape.dim_size(0);
}

template <typename T>
SelectFastPathResult TryLaunchSelectFastPath(
    OpKernelContext* ctx, const Tensor& cond, const Tensor& then_t,
    const Tensor& else_t, const TensorShape& output_shape,
    bool use_legacy_broadcast, Tensor* output) {
  if (!SelectFastPathLauncher<T>::kSupported ||
      !ShouldUseSelectCustomKernelFastPath(cond, then_t, else_t, output_shape,
                                           use_legacy_broadcast)) {
    return SelectFastPathResult::kNotHandled;
  }

  const int64_t output_elements = output_shape.num_elements();
  if (output_elements <= 0) return SelectFastPathResult::kNotHandled;
  if (!SameShape(then_t.shape(), output_shape) ||
      !SameShape(else_t.shape(), output_shape)) {
    return SelectFastPathResult::kNotHandled;
  }

  const void* cond_ptr = cond.tensor_data().data();
  const void* then_ptr = then_t.tensor_data().data();
  const void* else_ptr = else_t.tensor_data().data();
  void* out_ptr = const_cast<char*>(output->tensor_data().data());
  musaStream_t stream = GetMusaStreamByCtx(ctx);

  bool launched = false;
  if (SameShape(cond.shape(), output_shape)) {
    SelectFastPathLauncher<T>::SameShape(cond_ptr, then_ptr, else_ptr, out_ptr,
                                         output_elements, stream);
    launched = true;
  } else if (cond.NumElements() == 1) {
    SelectFastPathLauncher<T>::ScalarCond(cond_ptr, then_ptr, else_ptr, out_ptr,
                                          output_elements, stream);
    launched = true;
  } else if (use_legacy_broadcast && cond.dims() == 1 &&
             output_shape.dims() > 1 &&
             cond.dim_size(0) == output_shape.dim_size(0)) {
    const int64_t inner_size = output_elements / cond.dim_size(0);
    SelectFastPathLauncher<T>::Rank1Cond(cond_ptr, then_ptr, else_ptr, out_ptr,
                                         output_elements, inner_size, stream);
    launched = true;
  }

  if (!launched) return SelectFastPathResult::kNotHandled;

  auto launch_status = musaGetLastError();
  if (launch_status != musaSuccess) {
    ctx->CtxFailure(errors::Internal("MUSA Select fast path launch failed: ",
                                     musaGetErrorString(launch_status)));
    return SelectFastPathResult::kFailed;
  }

  return SelectFastPathResult::kLaunched;
}

}  // namespace

template <typename T>
class MusaSelectOp : public MusaOpKernel {
 public:
  explicit MusaSelectOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& cond = ctx->input(0);
    const Tensor& then_t = ctx->input(1);
    const Tensor& else_t = ctx->input(2);

    BCast bcast_te(BCast::FromShape(then_t.shape()),
                   BCast::FromShape(else_t.shape()));
    if (!bcast_te.IsValid()) {
      ctx->SetStatus(
          errors::InvalidArgument("Incompatible shapes: then vs else"));
      return;
    }
    TensorShape te_shape = BCast::ToShape(bcast_te.output_shape());

    bool use_legacy_broadcast = false;
    TensorShape output_shape;

    BCast bcast_final(BCast::FromShape(cond.shape()),
                      BCast::FromShape(te_shape));

    if (bcast_final.IsValid()) {
      output_shape = BCast::ToShape(bcast_final.output_shape());
      use_legacy_broadcast = false;
    } else if (cond.dims() == 1 && te_shape.dims() > 1 &&
               cond.dim_size(0) == te_shape.dim_size(0)) {
      output_shape = te_shape;
      use_legacy_broadcast = true;
    } else {
      ctx->SetStatus(
          errors::InvalidArgument("Incompatible shapes: cond vs (then/else)"));
      return;
    }

    Tensor* output = nullptr;
    const bool fast_path_possible =
        SelectFastPathLauncher<T>::kSupported &&
        ShouldUseSelectCustomKernelFastPath(cond, then_t, else_t, output_shape,
                                            use_legacy_broadcast);
    if (fast_path_possible) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    } else if (then_t.shape() == output_shape) {
      const std::vector<int> forwardable_input_indices = {1};
      OP_REQUIRES_OK(
          ctx, ctx->forward_input_or_allocate_output(forwardable_input_indices,
                                                     0, output_shape, &output));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    }
    if (output->NumElements() == 0) return;

    const auto fast_path_status = TryLaunchSelectFastPath<T>(
        ctx, cond, then_t, else_t, output_shape, use_legacy_broadcast, output);
    if (fast_path_status != SelectFastPathResult::kNotHandled) {
      return;
    }

    MusaDevice* device = reinterpret_cast<MusaDevice*>(ctx->device());
    auto& handle = device->mudnn_handle();

    std::vector<std::vector<int64_t>> shape_storage;
    shape_storage.reserve(10);

    auto CreateMTensor_b = [&](const Tensor& input,
                               bool force_left_align = false) -> mTensor {
      mTensor mt = CreateMTensor(input, mFormat::NCHW);

      int target_rank = output_shape.dims();
      std::vector<int64_t> t_dims(target_rank);
      std::vector<int64_t> i_strides(target_rank, 0);

      for (int i = 0; i < target_rank; ++i)
        t_dims[i] = output_shape.dim_size(i);

      int input_rank = input.dims();
      std::vector<int64_t> dense_strides(input_rank, 1);
      if (input_rank > 0) {
        for (int i = input_rank - 2; i >= 0; --i)
          dense_strides[i] = dense_strides[i + 1] * input.dim_size(i + 1);
      }

      if (force_left_align) {
        if (input_rank == 1) {
          i_strides[0] = dense_strides[0];
          for (int i = 1; i < target_rank; ++i) i_strides[i] = 0;
        }
      } else {
        for (int i = 1; i <= target_rank; ++i) {
          int target_idx = target_rank - i;
          int input_idx = input_rank - i;
          if (input_idx >= 0) {
            if (input.dim_size(input_idx) == t_dims[target_idx]) {
              i_strides[target_idx] = dense_strides[input_idx];
            } else {
              i_strides[target_idx] = 0;
            }
          } else {
            i_strides[target_idx] = 0;
          }
        }
      }

      shape_storage.push_back(t_dims);
      shape_storage.push_back(i_strides);
      mt.SetNdInfo(target_rank, shape_storage[shape_storage.size() - 2].data(),
                   shape_storage[shape_storage.size() - 1].data());
      return mt;
    };

    auto cond_mt = CreateMTensor_b(cond, use_legacy_broadcast);

    auto then_mt = CreateMTensor_b(then_t, false);
    auto else_mt = CreateMTensor_b(else_t, false);

    auto out_mt = CreateMTensor_b(*output, false);

    ::musa::dnn::Ternary op;
    op.SetMode(::musa::dnn::Ternary::Mode::SELECT);

    auto status = op.Run(handle, out_mt, cond_mt, then_mt, else_mt);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Select run failed"));
  }
};

#define REGISTER_SELECT(T)                                          \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Select").Device(DEVICE_MTGPU).TypeConstraint<T>("T"),   \
      MusaSelectOp<T>);                                             \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("SelectV2").Device(DEVICE_MTGPU).TypeConstraint<T>("T"), \
      MusaSelectOp<T>);

REGISTER_SELECT(float);
REGISTER_SELECT(double);
REGISTER_SELECT(int32);
REGISTER_SELECT(int64);
REGISTER_SELECT(bool);
REGISTER_SELECT(Eigen::half);
REGISTER_SELECT(Eigen::bfloat16);

#undef REGISTER_SELECT

}  // namespace musa
}  // namespace tensorflow

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

namespace {

extern "C" {
void LaunchMusaAddContiguousFloat(const float* lhs, const float* rhs,
                                  float* output, int64_t size,
                                  musaStream_t stream);
void LaunchMusaAddScalarFloat(const float* dense, const float* scalar,
                              float* output, int64_t size, musaStream_t stream);
void LaunchMusaAddTailVectorFloat(const float* dense, const float* tail_vector,
                                  float* output, int64_t size, int64_t width,
                                  musaStream_t stream);
}

enum class AddFastPathResult {
  kNotHandled = 0,
  kLaunched,
  kFailed,
};

bool UseAddBroadcastViewOpt() {
  const char* env = std::getenv("MUSA_ADDV2_ENABLE_BCAST_VIEW_OPT");
  if (env == nullptr || std::string(env).empty()) {
    return true;
  }
  const std::string value(env);
  return !(value == "0" || value == "false" || value == "FALSE" ||
           value == "off" || value == "OFF" || value == "no" || value == "NO");
}

bool SameShape(const TensorShape& lhs, const TensorShape& rhs) {
  if (lhs.dims() != rhs.dims()) return false;
  for (int i = 0; i < lhs.dims(); ++i) {
    if (lhs.dim_size(i) != rhs.dim_size(i)) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> MakeDenseStrides(const Tensor& tensor) {
  std::vector<int64_t> strides(tensor.dims(), 1);
  for (int i = tensor.dims() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * tensor.dim_size(i + 1);
  }
  return strides;
}

bool SameShape(const Tensor& tensor, const TensorShape& output_shape) {
  if (tensor.dims() != output_shape.dims()) return false;
  for (int i = 0; i < tensor.dims(); ++i) {
    if (tensor.dim_size(i) != output_shape.dim_size(i)) {
      return false;
    }
  }
  return true;
}

bool IsSmallRepeatedBroadcast(const Tensor& tensor,
                              const TensorShape& output_shape) {
  // The broadcast-view path only helps when a relatively small input is reused
  // many times in the output. Without this guard, descriptor setup overhead can
  // dominate and turn into a regression on common same-shape AddV2 cases.
  if (output_shape.dims() == 0 || tensor.NumElements() <= 0 ||
      output_shape.num_elements() <= 0) {
    return false;
  }
  if (SameShape(tensor, output_shape)) {
    return false;
  }

  const int64_t input_elements = tensor.NumElements();
  const int64_t output_elements = output_shape.num_elements();
  if (input_elements <= 0 || output_elements <= input_elements) {
    return false;
  }

  const int64_t reuse_factor = output_elements / input_elements;
  if (reuse_factor < 4) {
    return false;
  }

  if (input_elements == 1) {
    return true;
  }

  if (input_elements > 4096) {
    return false;
  }

  if (tensor.dims() < output_shape.dims()) {
    return true;
  }

  return tensor.dims() > 0 && tensor.dim_size(0) == 1;
}

bool UseAddCustomKernelFastPath() {
  const char* env = std::getenv("MUSA_ADDV2_ENABLE_CUSTOM_KERNEL");
  if (env == nullptr || std::string(env).empty()) {
    return true;
  }
  const std::string value(env);
  return !(value == "0" || value == "false" || value == "FALSE" ||
           value == "off" || value == "OFF" || value == "no" || value == "NO");
}

bool IsTailVectorBroadcast(const Tensor& tensor,
                           const TensorShape& output_shape, int64_t* width) {
  if (output_shape.dims() <= 0) {
    return false;
  }
  const int64_t last_dim = output_shape.dim_size(output_shape.dims() - 1);
  if (last_dim <= 0 || tensor.NumElements() != last_dim || tensor.dims() == 0) {
    return false;
  }

  // Accept [C], [1, C], [1, 1, C], ... as tail-vector broadcasts.
  for (int i = tensor.dims() - 1; i >= 0; --i) {
    const int64_t dim = tensor.dim_size(i);
    if (i == tensor.dims() - 1) {
      if (dim != last_dim) {
        return false;
      }
      continue;
    }
    if (dim != 1) {
      return false;
    }
  }

  *width = last_dim;
  return true;
}

template <typename T>
AddFastPathResult TryLaunchAddFastPath(OpKernelContext* ctx, const Tensor& in0,
                                       const Tensor& in1,
                                       const TensorShape& output_shape,
                                       bool same_shape, Tensor* out) {
  return AddFastPathResult::kNotHandled;
}

template <>
AddFastPathResult TryLaunchAddFastPath<float>(OpKernelContext* ctx,
                                              const Tensor& in0,
                                              const Tensor& in1,
                                              const TensorShape& output_shape,
                                              bool same_shape, Tensor* out) {
  if (!UseAddCustomKernelFastPath()) {
    return AddFastPathResult::kNotHandled;
  }

  const int64_t output_elements = output_shape.num_elements();
  if (output_elements <= 0) {
    return AddFastPathResult::kNotHandled;
  }

  const float* in0_ptr = in0.flat<float>().data();
  const float* in1_ptr = in1.flat<float>().data();
  float* out_ptr = out->flat<float>().data();
  musaStream_t stream = GetMusaStreamByCtx(ctx);
  if (stream == nullptr) {
    return AddFastPathResult::kNotHandled;
  }

  bool launched = false;
  if (same_shape) {
    LaunchMusaAddContiguousFloat(in0_ptr, in1_ptr, out_ptr, output_elements,
                                 stream);
    launched = true;
  } else if (in0.NumElements() == output_elements && in1.NumElements() == 1) {
    LaunchMusaAddScalarFloat(in0_ptr, in1_ptr, out_ptr, output_elements,
                             stream);
    launched = true;
  } else if (in1.NumElements() == output_elements && in0.NumElements() == 1) {
    LaunchMusaAddScalarFloat(in1_ptr, in0_ptr, out_ptr, output_elements,
                             stream);
    launched = true;
  } else if (in0.NumElements() == output_elements) {
    int64_t width = 0;
    if (IsTailVectorBroadcast(in1, output_shape, &width)) {
      LaunchMusaAddTailVectorFloat(in0_ptr, in1_ptr, out_ptr, output_elements,
                                   width, stream);
      launched = true;
    }
  } else if (in1.NumElements() == output_elements) {
    int64_t width = 0;
    if (IsTailVectorBroadcast(in0, output_shape, &width)) {
      LaunchMusaAddTailVectorFloat(in1_ptr, in0_ptr, out_ptr, output_elements,
                                   width, stream);
      launched = true;
    }
  }

  if (!launched) {
    return AddFastPathResult::kNotHandled;
  }

  auto launch_status = musaGetLastError();
  if (launch_status != musaSuccess) {
    ctx->CtxFailure(errors::Internal("MUSA Add fast path launch failed: ",
                                     musaGetErrorString(launch_status)));
    return AddFastPathResult::kFailed;
  }

  return AddFastPathResult::kLaunched;
}

Status ConfigureBroadcastView(const Tensor& tensor,
                              const TensorShape& output_shape, mTensor* mt,
                              std::vector<int64_t>* dims,
                              std::vector<int64_t>* strides) {
  // Express TensorFlow-style broadcast as a muDNN tensor view by keeping the
  // output dims and setting broadcasted axes to stride 0.
  if (SameShape(tensor, output_shape) || output_shape.dims() == 0) {
    return Status::OK();
  }

  const int input_rank = tensor.dims();
  const int target_rank = output_shape.dims();
  const auto dense_strides = MakeDenseStrides(tensor);

  dims->assign(target_rank, 1);
  strides->assign(target_rank, 0);

  for (int i = 0; i < target_rank; ++i) {
    (*dims)[i] = output_shape.dim_size(i);
  }

  for (int i = 1; i <= target_rank; ++i) {
    const int target_idx = target_rank - i;
    const int input_idx = input_rank - i;
    if (input_idx < 0) continue;

    const int64_t in_dim = tensor.dim_size(input_idx);
    const int64_t out_dim = output_shape.dim_size(target_idx);
    if (in_dim == out_dim) {
      (*strides)[target_idx] = dense_strides[input_idx];
    } else if (in_dim != 1) {
      return errors::InvalidArgument(
          "Invalid Add broadcast view: input shape=",
          tensor.shape().DebugString(),
          ", output shape=", output_shape.DebugString());
    }
  }

  auto status = mt->SetNdInfo(target_rank, dims->data(), strides->data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal("MUSA Add SetNdInfo failed. Status: ",
                            static_cast<int>(status));
  }
  return Status::OK();
}

}  // namespace

template <typename T>
class MusaAddOp : public MusaOpKernel {
 public:
  explicit MusaAddOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  // Add is element-wise and computationally lightweight
  // Mark as inexpensive to enable inline scheduling
  // Expected improvement: Reduced scheduling overhead
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    TensorShape output_shape;
    const bool same_shape = SameShape(in0.shape(), in1.shape());
    if (same_shape) {
      output_shape = in0.shape();
    } else {
      BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                  BCast::Vec(in1.shape().dim_sizes()));
      OP_REQUIRES(ctx, bcast.IsValid(),
                  errors::InvalidArgument("Incompatible shapes for Add: ",
                                          in0.shape().DebugString(), " and ",
                                          in1.shape().DebugString()));
      output_shape = BCast::ToShape(bcast.output_shape());
    }

    // Reuse the left input buffer when TensorFlow determines it is safe.
    // This particularly helps the common [N, C] + [C] broadcast pattern where
    // the output shape matches input 0.
    Tensor* out = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output({0}, 0, output_shape, &out));

    if (in0.NumElements() == 0 || in1.NumElements() == 0 ||
        output_shape.num_elements() == 0) {
      return;
    }

    const auto fast_path_status =
        TryLaunchAddFastPath<T>(ctx, in0, in1, output_shape, same_shape, out);
    if (fast_path_status != AddFastPathResult::kNotHandled) {
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    mTensor t0 = CreateMTensor(in0, format_);
    mTensor t1 = CreateMTensor(in1, format_);
    mTensor t_out = CreateMTensor(*out, format_);
    std::vector<int64_t> t0_dims, t0_strides;
    std::vector<int64_t> t1_dims, t1_strides;
    if (UseAddBroadcastViewOpt()) {
      if (IsSmallRepeatedBroadcast(in0, output_shape)) {
        OP_REQUIRES_OK(ctx, ConfigureBroadcastView(in0, output_shape, &t0,
                                                   &t0_dims, &t0_strides));
      }
      if (IsSmallRepeatedBroadcast(in1, output_shape)) {
        OP_REQUIRES_OK(ctx, ConfigureBroadcastView(in1, output_shape, &t1,
                                                   &t1_dims, &t1_strides));
      }
    }

    ::musa::dnn::Binary op;
    op.SetMode(::musa::dnn::Binary::Mode::ADD);

    auto status = op.Run(handle, t_out, t0, t1);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Add execution failed."));
  }
};

#define REGISTER_MUSA_ADD(TYPE)                               \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("AddV2").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaAddOp<TYPE>);                                       \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Add").Device("MUSA").TypeConstraint<TYPE>("T"), MusaAddOp<TYPE>);

REGISTER_MUSA_ADD(float);
REGISTER_MUSA_ADD(int32);
REGISTER_MUSA_ADD(int64);
REGISTER_MUSA_ADD(Eigen::half);
REGISTER_MUSA_ADD(bfloat16);
REGISTER_MUSA_ADD(double);
REGISTER_MUSA_ADD(uint8);
REGISTER_MUSA_ADD(bool);

}  // namespace musa
}  // namespace tensorflow

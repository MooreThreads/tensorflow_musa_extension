#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "../utils/musa_guarded_philox_random.h"
#include "mu/device/musa_memcpy.h"
#include "utils_op.h"

#define CDIV(a, b) (((a) + (b) - 1) / (b))

namespace tensorflow {
namespace musa {

template <typename T, typename DIST_TYPE>
void LaunchPhiloxNormalKernel(musaStream_t stream, T* data,
                              uint64_t num_elements,
                              const random::PhiloxRandom& philox,
                              const DIST_TYPE& dist);

template <typename T>
class MusaNormalOp : public MusaOpKernel {
 public:
  explicit MusaNormalOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2_));
  }

  void Compute(OpKernelContext* ctx) override {
    using PhiloxRandom = random::PhiloxRandom;
    using NormalDist = random::NormalDistribution<PhiloxRandom>;
    using TruncatedDist = random::TruncatedNormalDistribution<PhiloxRandom>;

    // Parse the input
    const Tensor& shape_tensor = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &shape));

    // allocate the output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    int64_t total_elements = shape.num_elements();
    if (total_elements == 0) return;

    // Initialize Philox with seed management
    GuardedPhiloxRandom generator;
    generator.Init(seed_, seed2_);

    // ready to launch MUSA kernel with pre-reserved Philox samples
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    auto activated_mode = name();
    if (activated_mode == "RandomStandardNormal") {
      // NormalDistribution generates 4 values per call, consuming 4 uint32s
      // For N outputs, need ⌈N/4⌉ calls, each consuming 4 uint32s
      uint64_t samples_needed = CDIV(output->NumElements(), 4) * 4;
      auto philox = generator.ReserveSamples32(samples_needed);
      NormalDist dist;  // Stack object, no heap allocation
      LaunchPhiloxNormalKernel<T, NormalDist>(stream, output->flat<T>().data(),
                                              output->NumElements(), philox,
                                              dist);
    } else if (activated_mode == "TruncatedNormal") {
      // TruncatedNormal uses rejection sampling, need ~4x oversampling
      // For N outputs, reserve 4N elements worth of samples
      uint64_t samples_needed = total_elements * 4;
      auto philox = generator.ReserveSamples32(samples_needed);
      TruncatedDist dist;  // Stack object, no heap allocation
      LaunchPhiloxNormalKernel<T, TruncatedDist>(
          stream, output->flat<T>().data(), output->NumElements(), philox,
          dist);
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported op name: ", name()));
    }
  }

 private:
  tensorflow::int64 seed_;
  tensorflow::int64 seed2_;
};

#define REGISTER_MUSA_NORMAL_KERNEL(TYPE)                     \
  REGISTER_KERNEL_BUILDER(Name("RandomStandardNormal")        \
                              .Device("MUSA")                 \
                              .HostMemory("shape")            \
                              .TypeConstraint<TYPE>("dtype"), \
                          MusaNormalOp<TYPE>);                \
  REGISTER_KERNEL_BUILDER(Name("TruncatedNormal")             \
                              .Device("MUSA")                 \
                              .HostMemory("shape")            \
                              .TypeConstraint<TYPE>("dtype"), \
                          MusaNormalOp<TYPE>);

// 执行批量注册
REGISTER_MUSA_NORMAL_KERNEL(float);
REGISTER_MUSA_NORMAL_KERNEL(double);
REGISTER_MUSA_NORMAL_KERNEL(Eigen::half)
REGISTER_MUSA_NORMAL_KERNEL(Eigen::bfloat16);

#undef REGISTER_MUSA_NORMAL_KERNEL

}  // namespace musa
}  // namespace tensorflow
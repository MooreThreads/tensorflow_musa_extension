// Optimized MUSA Gather Op Implementation
// Uses custom kernels for maximum performance
//
// Performance optimizations:
// 1. Custom MUSA kernels with vectorized memory access
// 2. GPU-side bounds checking (no D2H memcpy overhead)
// 3. Coalesced memory access patterns
// 4. Direct launcher calls without muDNN overhead

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

// ============================================================================
// Custom Kernel Launcher Declarations
// ============================================================================

extern "C" {
void LaunchGatherV2FloatInt32(const float* params, const int* indices, float* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, int limit,
                              musaStream_t stream);
void LaunchGatherV2FloatInt64(const float* params, const long long* indices, float* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, long long limit,
                              musaStream_t stream);
void LaunchGatherV2DoubleInt32(const double* params, const int* indices, double* output,
                               int64_t batch_size, int64_t axis_size, int64_t inner_size,
                               int64_t indices_size, int64_t params_stride, int limit,
                               musaStream_t stream);
void LaunchGatherV2DoubleInt64(const double* params, const long long* indices, double* output,
                               int64_t batch_size, int64_t axis_size, int64_t inner_size,
                               int64_t indices_size, int64_t params_stride, long long limit,
                               musaStream_t stream);
void LaunchGatherV2Int32Int32(const int* params, const int* indices, int* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, int limit,
                              musaStream_t stream);
void LaunchGatherV2Int32Int64(const int* params, const long long* indices, int* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, long long limit,
                              musaStream_t stream);
void LaunchGatherV2Int64Int32(const long long* params, const int* indices, long long* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, int limit,
                              musaStream_t stream);
void LaunchGatherV2Int64Int64(const long long* params, const long long* indices, long long* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, long long limit,
                              musaStream_t stream);
void LaunchGatherV2BoolInt32(const bool* params, const int* indices, bool* output,
                             int64_t batch_size, int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride, int limit,
                             musaStream_t stream);
void LaunchGatherV2BoolInt64(const bool* params, const long long* indices, bool* output,
                             int64_t batch_size, int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride, long long limit,
                             musaStream_t stream);
void LaunchGatherV2HalfInt32(const void* params, const int* indices, void* output,
                             int64_t batch_size, int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride, int limit,
                             musaStream_t stream);
void LaunchGatherV2HalfInt64(const void* params, const long long* indices, void* output,
                             int64_t batch_size, int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride, long long limit,
                             musaStream_t stream);
void LaunchGatherV2BFloat16Int32(const void* params, const int* indices, void* output,
                                 int64_t batch_size, int64_t axis_size, int64_t inner_size,
                                 int64_t indices_size, int64_t params_stride, int limit,
                                 musaStream_t stream);
void LaunchGatherV2BFloat16Int64(const void* params, const long long* indices, void* output,
                                 int64_t batch_size, int64_t axis_size, int64_t inner_size,
                                 int64_t indices_size, int64_t params_stride, long long limit,
                                 musaStream_t stream);
}

namespace tensorflow {
namespace musa {

// ============================================================================
// Optimized Gather Op Implementation
// ============================================================================

template <typename T, typename IndexT>
class MusaGatherOp : public MusaOpKernel {
 public:
  explicit MusaGatherOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    axis_ = 0;
    has_axis_input_ = false;
  }

  // Gather is computationally intensive due to irregular memory access
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& params = ctx->input(0);
    const Tensor& indices = ctx->input(1);

    int64_t axis = axis_;
    if (ctx->num_inputs() >= 3) {
      const Tensor& axis_tensor = ctx->input(2);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                  errors::InvalidArgument("axis must be a scalar"));

      if (axis_tensor.dtype() == DT_INT32) {
        axis = static_cast<int64_t>(axis_tensor.scalar<int32>()());
      } else if (axis_tensor.dtype() == DT_INT64) {
        axis = axis_tensor.scalar<int64>()();
      } else {
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument("axis must be int32 or int64"));
      }
      has_axis_input_ = true;
    }

    const int64_t params_dims = params.dims();
    if (axis < 0) {
      axis += params_dims;
    }

    OP_REQUIRES(
        ctx, axis >= 0 && axis < params_dims,
        errors::InvalidArgument("Expected axis in the range [", -params_dims,
                                ", ", params_dims, "), but got ", axis));

    OP_REQUIRES(ctx, indices.dtype() == DT_INT32 || indices.dtype() == DT_INT64,
                errors::InvalidArgument("indices must be int32 or int64"));

    // Build output shape
    TensorShape output_shape;
    for (int64_t i = 0; i < axis; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }
    for (int64_t i = 0; i < indices.dims(); ++i) {
      output_shape.AddDim(indices.dim_size(i));
    }
    for (int64_t i = axis + 1; i < params_dims; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    // Compute dimensions for kernel launch
    const int64_t limit = params.dim_size(axis);
    
    int64_t batch_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
      batch_size *= params.dim_size(i);
    }
    
    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < params_dims; ++i) {
      inner_size *= params.dim_size(i);
    }
    
    const int64_t indices_size = indices.NumElements();
    const int64_t params_stride = limit * inner_size;

    // Get stream for async execution
    musaStream_t stream = GetMusaStreamByCtx(ctx);

    // Launch optimized custom kernel
    LaunchKernel(
        params.flat<T>().data(),
        indices.flat<IndexT>().data(),
        output->flat<T>().data(),
        batch_size,
        limit,
        inner_size,
        indices_size,
        params_stride,
        static_cast<IndexT>(limit),
        stream);
  }

 private:
  int64_t axis_;
  bool has_axis_input_;

  // Type-specific kernel launcher
  void LaunchKernel(const T* params, const IndexT* indices, T* output,
                    int64_t batch_size, int64_t axis_size, int64_t inner_size,
                    int64_t indices_size, int64_t params_stride, IndexT limit,
                    musaStream_t stream);
};

// ============================================================================
// Launcher Specializations
// ============================================================================

#define DEFINE_GATHER_LAUNCHER(T, IndexT, launcher_func) \
  template <> \
  void MusaGatherOp<T, IndexT>::LaunchKernel( \
      const T* params, const IndexT* indices, T* output, \
      int64_t batch_size, int64_t axis_size, int64_t inner_size, \
      int64_t indices_size, int64_t params_stride, IndexT limit, \
      musaStream_t stream) { \
    launcher_func(params, indices, output, batch_size, axis_size, inner_size, \
                  indices_size, params_stride, limit, stream); \
  }

DEFINE_GATHER_LAUNCHER(float, int32, LaunchGatherV2FloatInt32)
DEFINE_GATHER_LAUNCHER(float, int64, LaunchGatherV2FloatInt64)
DEFINE_GATHER_LAUNCHER(double, int32, LaunchGatherV2DoubleInt32)
DEFINE_GATHER_LAUNCHER(double, int64, LaunchGatherV2DoubleInt64)
DEFINE_GATHER_LAUNCHER(int32, int32, LaunchGatherV2Int32Int32)
DEFINE_GATHER_LAUNCHER(int32, int64, LaunchGatherV2Int32Int64)
DEFINE_GATHER_LAUNCHER(int64, int32, LaunchGatherV2Int64Int32)
DEFINE_GATHER_LAUNCHER(int64, int64, LaunchGatherV2Int64Int64)
DEFINE_GATHER_LAUNCHER(bool, int32, LaunchGatherV2BoolInt32)
DEFINE_GATHER_LAUNCHER(bool, int64, LaunchGatherV2BoolInt64)

// Half specialization
#define DEFINE_GATHER_LAUNCHER_HALF(IndexT, launcher_func) \
  template <> \
  void MusaGatherOp<Eigen::half, IndexT>::LaunchKernel( \
      const Eigen::half* params, const IndexT* indices, Eigen::half* output, \
      int64_t batch_size, int64_t axis_size, int64_t inner_size, \
      int64_t indices_size, int64_t params_stride, IndexT limit, \
      musaStream_t stream) { \
    launcher_func(reinterpret_cast<const void*>(params), indices, \
                  reinterpret_cast<void*>(output), batch_size, axis_size, inner_size, \
                  indices_size, params_stride, limit, stream); \
  }

DEFINE_GATHER_LAUNCHER_HALF(int32, LaunchGatherV2HalfInt32)
DEFINE_GATHER_LAUNCHER_HALF(int64, LaunchGatherV2HalfInt64)

// BFloat16 specialization
#define DEFINE_GATHER_LAUNCHER_BF16(IndexT, launcher_func) \
  template <> \
  void MusaGatherOp<bfloat16, IndexT>::LaunchKernel( \
      const bfloat16* params, const IndexT* indices, bfloat16* output, \
      int64_t batch_size, int64_t axis_size, int64_t inner_size, \
      int64_t indices_size, int64_t params_stride, IndexT limit, \
      musaStream_t stream) { \
    launcher_func(reinterpret_cast<const void*>(params), indices, \
                  reinterpret_cast<void*>(output), batch_size, axis_size, inner_size, \
                  indices_size, params_stride, limit, stream); \
  }

DEFINE_GATHER_LAUNCHER_BF16(int32, LaunchGatherV2BFloat16Int32)
DEFINE_GATHER_LAUNCHER_BF16(int64, LaunchGatherV2BFloat16Int64)

#undef DEFINE_GATHER_LAUNCHER
#undef DEFINE_GATHER_LAUNCHER_HALF
#undef DEFINE_GATHER_LAUNCHER_BF16

// ============================================================================
// Kernel Registration
// ============================================================================

#define REGISTER_GATHER_V2_FULL(T)                               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int32>("Tindices") \
                              .TypeConstraint<int32>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int32>);               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int64>("Tindices") \
                              .TypeConstraint<int64>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int64>);               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int32>("Tindices") \
                              .TypeConstraint<int64>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int32>);               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int64>("Tindices") \
                              .TypeConstraint<int32>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int64>);

REGISTER_GATHER_V2_FULL(float);
REGISTER_GATHER_V2_FULL(double);
REGISTER_GATHER_V2_FULL(int32);
REGISTER_GATHER_V2_FULL(int64);
REGISTER_GATHER_V2_FULL(bool);
REGISTER_GATHER_V2_FULL(Eigen::half);
REGISTER_GATHER_V2_FULL(bfloat16);

#undef REGISTER_GATHER_V2_FULL

#define REGISTER_GATHER_V1(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("Gather")                          \
                              .Device("MUSA")                     \
                              .TypeConstraint<T>("Tparams")       \
                              .TypeConstraint<int32>("Tindices"), \
                          MusaGatherOp<T, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("Gather")                          \
                              .Device("MUSA")                     \
                              .TypeConstraint<T>("Tparams")       \
                              .TypeConstraint<int64>("Tindices"), \
                          MusaGatherOp<T, int64>);

REGISTER_GATHER_V1(float);
REGISTER_GATHER_V1(double);
REGISTER_GATHER_V1(int32);
REGISTER_GATHER_V1(int64);
REGISTER_GATHER_V1(bool);
REGISTER_GATHER_V1(Eigen::half);
REGISTER_GATHER_V1(bfloat16);

#undef REGISTER_GATHER_V1

}  // namespace musa
}  // namespace tensorflow

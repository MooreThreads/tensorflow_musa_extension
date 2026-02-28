// Optimized MUSA GatherND Op Implementation
// Uses custom kernels for maximum performance
//
// Performance optimizations:
// 1. Custom MUSA kernels with coalesced memory access
// 2. Precomputed strides on GPU
// 3. No muDNN overhead
// 4. Direct launcher calls with optimal grid/block configuration

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

// ============================================================================
// Custom Kernel Launcher Declarations
// ============================================================================

extern "C" {
void LaunchGatherNDFloatInt32(const float* params, const int* indices, float* output,
                              int index_depth, int64_t indices_nd_size, int64_t slice_size,
                              const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDFloatInt64(const float* params, const long long* indices, float* output,
                              int index_depth, int64_t indices_nd_size, int64_t slice_size,
                              const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDDoubleInt32(const double* params, const int* indices, double* output,
                               int index_depth, int64_t indices_nd_size, int64_t slice_size,
                               const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDDoubleInt64(const double* params, const long long* indices, double* output,
                               int index_depth, int64_t indices_nd_size, int64_t slice_size,
                               const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDInt32Int32(const int* params, const int* indices, int* output,
                              int index_depth, int64_t indices_nd_size, int64_t slice_size,
                              const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDInt32Int64(const int* params, const long long* indices, int* output,
                              int index_depth, int64_t indices_nd_size, int64_t slice_size,
                              const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDInt64Int32(const long long* params, const int* indices, long long* output,
                              int index_depth, int64_t indices_nd_size, int64_t slice_size,
                              const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDInt64Int64(const long long* params, const long long* indices, long long* output,
                              int index_depth, int64_t indices_nd_size, int64_t slice_size,
                              const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDBoolInt32(const bool* params, const int* indices, bool* output,
                             int index_depth, int64_t indices_nd_size, int64_t slice_size,
                             const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDBoolInt64(const bool* params, const long long* indices, bool* output,
                             int index_depth, int64_t indices_nd_size, int64_t slice_size,
                             const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDHalfInt32(const void* params, const int* indices, void* output,
                             int index_depth, int64_t indices_nd_size, int64_t slice_size,
                             const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDHalfInt64(const void* params, const long long* indices, void* output,
                             int index_depth, int64_t indices_nd_size, int64_t slice_size,
                             const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDBFloat16Int32(const void* params, const int* indices, void* output,
                                 int index_depth, int64_t indices_nd_size, int64_t slice_size,
                                 const int64_t* params_strides, musaStream_t stream);
void LaunchGatherNDBFloat16Int64(const void* params, const long long* indices, void* output,
                                 int index_depth, int64_t indices_nd_size, int64_t slice_size,
                                 const int64_t* params_strides, musaStream_t stream);
}

namespace tensorflow {
namespace musa {

// ============================================================================
// Optimized GatherND Op Implementation
// ============================================================================

template <typename T, typename IndexT>
class MusaGatherNdOp : public MusaOpKernel {
 public:
  explicit MusaGatherNdOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  // GatherNd is memory-intensive - irregular memory access
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& params = ctx->input(0);
    const Tensor& indices = ctx->input(1);

    const int64_t params_dims = params.dims();
    const int64_t indices_dims = indices.dims();
    const int64_t index_depth = indices.dim_size(indices_dims - 1);

    OP_REQUIRES(ctx, index_depth <= params_dims,
                errors::InvalidArgument("index_depth (", index_depth,
                                        ") must be <= params_dims (",
                                        params_dims, ")"));

    // Build output shape
    TensorShape output_shape;
    for (int i = 0; i < indices_dims - 1; ++i) {
      output_shape.AddDim(indices.dim_size(i));
    }
    for (int i = index_depth; i < params_dims; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    // Compute dimensions for kernel launch
    const int64_t indices_nd_size = indices.NumElements() / index_depth;
    
    int64_t slice_size = 1;
    for (int i = index_depth; i < params_dims; ++i) {
      slice_size *= params.dim_size(i);
    }

    // Precompute params strides for each index dimension
    // Strides are computed from innermost to outermost
    std::vector<int64_t> params_strides(index_depth);
    for (int d = index_depth - 1; d >= 0; --d) {
      if (d == index_depth - 1) {
        params_strides[d] = slice_size;
      } else {
        params_strides[d] = params_strides[d + 1] * params.dim_size(d + 1);
      }
    }

    // Get stream
    musaStream_t stream = GetMusaStreamByCtx(ctx);

    // Copy strides to device
    int64_t* d_params_strides = nullptr;
    const size_t strides_bytes = index_depth * sizeof(int64_t);
    musaMalloc(reinterpret_cast<void**>(&d_params_strides), strides_bytes);
    musaMemcpyAsync(d_params_strides, params_strides.data(), strides_bytes,
                    musaMemcpyHostToDevice, stream);

    // Launch optimized kernel
    LaunchKernel(
        params.flat<T>().data(),
        indices.flat<IndexT>().data(),
        output->flat<T>().data(),
        static_cast<int>(index_depth),
        indices_nd_size,
        slice_size,
        d_params_strides,
        stream);

    // Free device memory
    musaFree(d_params_strides);
  }

 private:
  // Type-specific kernel launcher
  void LaunchKernel(const T* params, const IndexT* indices, T* output,
                    int index_depth, int64_t indices_nd_size, int64_t slice_size,
                    const int64_t* d_params_strides, musaStream_t stream);
};

// ============================================================================
// Launcher Specializations
// ============================================================================

#define DEFINE_GATHER_ND_LAUNCHER(T, IndexT, launcher_func) \
  template <> \
  void MusaGatherNdOp<T, IndexT>::LaunchKernel( \
      const T* params, const IndexT* indices, T* output, \
      int index_depth, int64_t indices_nd_size, int64_t slice_size, \
      const int64_t* d_params_strides, musaStream_t stream) { \
    launcher_func(params, indices, output, index_depth, indices_nd_size, \
                  slice_size, d_params_strides, stream); \
  }

DEFINE_GATHER_ND_LAUNCHER(float, int32, LaunchGatherNDFloatInt32)
DEFINE_GATHER_ND_LAUNCHER(float, int64, LaunchGatherNDFloatInt64)
DEFINE_GATHER_ND_LAUNCHER(double, int32, LaunchGatherNDDoubleInt32)
DEFINE_GATHER_ND_LAUNCHER(double, int64, LaunchGatherNDDoubleInt64)
DEFINE_GATHER_ND_LAUNCHER(int32, int32, LaunchGatherNDInt32Int32)
DEFINE_GATHER_ND_LAUNCHER(int32, int64, LaunchGatherNDInt32Int64)
DEFINE_GATHER_ND_LAUNCHER(int64, int32, LaunchGatherNDInt64Int32)
DEFINE_GATHER_ND_LAUNCHER(int64, int64, LaunchGatherNDInt64Int64)
DEFINE_GATHER_ND_LAUNCHER(bool, int32, LaunchGatherNDBoolInt32)
DEFINE_GATHER_ND_LAUNCHER(bool, int64, LaunchGatherNDBoolInt64)

// Half specialization
#define DEFINE_GATHER_ND_LAUNCHER_HALF(IndexT, launcher_func) \
  template <> \
  void MusaGatherNdOp<Eigen::half, IndexT>::LaunchKernel( \
      const Eigen::half* params, const IndexT* indices, Eigen::half* output, \
      int index_depth, int64_t indices_nd_size, int64_t slice_size, \
      const int64_t* d_params_strides, musaStream_t stream) { \
    launcher_func(reinterpret_cast<const void*>(params), indices, \
                  reinterpret_cast<void*>(output), index_depth, indices_nd_size, \
                  slice_size, d_params_strides, stream); \
  }

DEFINE_GATHER_ND_LAUNCHER_HALF(int32, LaunchGatherNDHalfInt32)
DEFINE_GATHER_ND_LAUNCHER_HALF(int64, LaunchGatherNDHalfInt64)

// BFloat16 specialization
#define DEFINE_GATHER_ND_LAUNCHER_BF16(IndexT, launcher_func) \
  template <> \
  void MusaGatherNdOp<bfloat16, IndexT>::LaunchKernel( \
      const bfloat16* params, const IndexT* indices, bfloat16* output, \
      int index_depth, int64_t indices_nd_size, int64_t slice_size, \
      const int64_t* d_params_strides, musaStream_t stream) { \
    launcher_func(reinterpret_cast<const void*>(params), indices, \
                  reinterpret_cast<void*>(output), index_depth, indices_nd_size, \
                  slice_size, d_params_strides, stream); \
  }

DEFINE_GATHER_ND_LAUNCHER_BF16(int32, LaunchGatherNDBFloat16Int32)
DEFINE_GATHER_ND_LAUNCHER_BF16(int64, LaunchGatherNDBFloat16Int64)

#undef DEFINE_GATHER_ND_LAUNCHER
#undef DEFINE_GATHER_ND_LAUNCHER_HALF
#undef DEFINE_GATHER_ND_LAUNCHER_BF16

// ============================================================================
// Kernel Registration
// ============================================================================

#define REGISTER_MUSA_GATHER_ND(type, itype)                      \
  REGISTER_KERNEL_BUILDER(Name("GatherNd")                        \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<type>("Tparams")    \
                              .TypeConstraint<itype>("Tindices"), \
                          MusaGatherNdOp<type, itype>);

REGISTER_MUSA_GATHER_ND(float, int32);
REGISTER_MUSA_GATHER_ND(float, int64);

REGISTER_MUSA_GATHER_ND(Eigen::half, int32);
REGISTER_MUSA_GATHER_ND(Eigen::half, int64);
REGISTER_MUSA_GATHER_ND(bfloat16, int32);
REGISTER_MUSA_GATHER_ND(bfloat16, int64);

REGISTER_MUSA_GATHER_ND(int32, int32);
REGISTER_MUSA_GATHER_ND(int32, int64);
REGISTER_MUSA_GATHER_ND(int64, int32);
REGISTER_MUSA_GATHER_ND(int64, int64);

#undef REGISTER_MUSA_GATHER_ND

}  // namespace musa
}  // namespace tensorflow

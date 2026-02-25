#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

constexpr int kMaxBroadcastDims = 8;

template <typename T>
void MusaBitwiseAndKernelLauncher(const T* input_a, const T* input_b,
                                  T* output, int64_t size);

template <typename T>
void MusaBitwiseAndBroadcastKernelLauncher(
    const T* input_a, const T* input_b, T* output,
    const int64_t* output_shape, const int64_t* a_strides,
    const int64_t* b_strides, int ndims, int64_t size);

template <typename T>
class MusaBitwiseAndOp : public MusaOpKernel {
 public:
  explicit MusaBitwiseAndOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_a = ctx->input(0);
    const Tensor& input_b = ctx->input(1);

    BCast bcast(BCast::FromShape(input_a.shape()),
                BCast::FromShape(input_b.shape()));
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes for BitwiseAnd: ",
                    input_a.shape().DebugString(), " vs. ",
                    input_b.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    const int64_t size = output->NumElements();
    if (size == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = (musaStream_t)handle.GetStream();

    if (input_a.shape() == input_b.shape()) {
      MusaBitwiseAndKernelLauncher<T>(input_a.flat<T>().data(),
                                      input_b.flat<T>().data(),
                                      output->flat<T>().data(),
                                      size);
    } else {
      const auto& out_shape_vec = bcast.output_shape();
      const int ndims = out_shape_vec.size();

      OP_REQUIRES(ctx, ndims <= kMaxBroadcastDims,
                  errors::InvalidArgument(
                      "BitwiseAnd broadcast supports at most ",
                      kMaxBroadcastDims, " dims, got ", ndims));

      auto pad_shape = [ndims](const TensorShape& shape) {
        std::vector<int64_t> padded(ndims, 1);
        int offset = ndims - shape.dims();
        for (int i = 0; i < shape.dims(); ++i) {
          padded[offset + i] = shape.dim_size(i);
        }
        return padded;
      };

      std::vector<int64_t> a_shape = pad_shape(input_a.shape());
      std::vector<int64_t> b_shape = pad_shape(input_b.shape());

      int64_t h_output_shape[kMaxBroadcastDims];
      int64_t h_a_strides[kMaxBroadcastDims];
      int64_t h_b_strides[kMaxBroadcastDims];

      for (int i = 0; i < ndims; ++i) {
        h_output_shape[i] = out_shape_vec[i];
      }
      auto compute_strides = [ndims](const std::vector<int64_t>& shape,
                                     int64_t* strides) {
        int64_t stride = 1;
        for (int i = ndims - 1; i >= 0; --i) {
          strides[i] = (shape[i] == 1) ? 0 : stride;
          stride *= shape[i];
        }
      };

      compute_strides(a_shape, h_a_strides);
      compute_strides(b_shape, h_b_strides);

      Tensor d_output_shape_t, d_a_strides_t, d_b_strides_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape({ndims}),
                                             &d_output_shape_t));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape({ndims}),
                                             &d_a_strides_t));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape({ndims}),
                                             &d_b_strides_t));

      auto* d_output_shape = d_output_shape_t.flat<int64_t>().data();
      auto* d_a_strides = d_a_strides_t.flat<int64_t>().data();
      auto* d_b_strides = d_b_strides_t.flat<int64_t>().data();

      musaMemcpyAsync(d_output_shape, h_output_shape,
                      ndims * sizeof(int64_t), musaMemcpyHostToDevice, stream);
      musaMemcpyAsync(d_a_strides, h_a_strides,
                      ndims * sizeof(int64_t), musaMemcpyHostToDevice, stream);
      musaMemcpyAsync(d_b_strides, h_b_strides,
                      ndims * sizeof(int64_t), musaMemcpyHostToDevice, stream);

      MusaBitwiseAndBroadcastKernelLauncher<T>(
          input_a.flat<T>().data(),
          input_b.flat<T>().data(),
          output->flat<T>().data(),
          d_output_shape, d_a_strides, d_b_strides,
          ndims, size);
    }

    auto kernel_status = musaGetLastError();
    OP_REQUIRES(ctx, kernel_status == musaSuccess,
                errors::Internal("MUSA BitwiseAnd kernel failed: ",
                                 musaGetErrorString(kernel_status)));
  }
};

#define REGISTER_MUSA_BITWISE_AND(TYPE)                              \
  REGISTER_KERNEL_BUILDER(Name("BitwiseAnd")                         \
                              .Device("MUSA")                        \
                              .TypeConstraint<TYPE>("T"),             \
                          MusaBitwiseAndOp<TYPE>)

REGISTER_MUSA_BITWISE_AND(int8);
REGISTER_MUSA_BITWISE_AND(int16);
REGISTER_MUSA_BITWISE_AND(int32);
REGISTER_MUSA_BITWISE_AND(long long);
REGISTER_MUSA_BITWISE_AND(uint8);
REGISTER_MUSA_BITWISE_AND(uint16);
REGISTER_MUSA_BITWISE_AND(uint32);
REGISTER_MUSA_BITWISE_AND(unsigned long long);

#undef REGISTER_MUSA_BITWISE_AND

}  // namespace musa
}  // namespace tensorflow
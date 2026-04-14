#include <mudnn.h>
#include <mudnn_xmma.h>
#include <musa_runtime.h>

#include <functional>
#include <memory>

#include "../utils_op.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

// The fused op for MusaLinearRelu, which computes MatMul + BiasAdd + Relu.
// The 2D path uses the MatMul+BiasAdd epilogue directly, then applies Relu
// in-place. The batch path writes MatMul output directly to the final tensor,
// then applies BiasAdd and Relu in-place.

template <typename T>
class MusaLinearReluOp : public MusaOpKernel {
 public:
  explicit MusaLinearReluOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &trans_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &trans_b_));
  }

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD(ctx);
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    const Tensor& bias_input = ctx->input(2);

    // 1. MatMul
    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", in0.shape().DebugString(), " vs ",
                    in1.shape().DebugString()));

    int64 d0 = in0.dim_size(in0.dims() - 2);
    int64 d1 = in0.dim_size(in0.dims() - 1);
    int64 d2 = in1.dim_size(in1.dims() - 2);
    int64 d3 = in1.dim_size(in1.dims() - 1);

    int64 m = trans_a_ ? d1 : d0;
    int64 k = trans_a_ ? d0 : d1;
    int64 n = trans_b_ ? d2 : d3;
    int64 k_check = trans_b_ ? d3 : d2;

    OP_REQUIRES(ctx, k == k_check,
                errors::InvalidArgument(
                    "Matrix size-incompatible: In[0] mismatch In[1]"));

    TensorShape mm_out_shape = bcast.output_batch_shape();
    mm_out_shape.AddDim(m);
    mm_out_shape.AddDim(n);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mm_out_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);
    mTensor mt_a = CreateMTensor(in0);
    mTensor mt_b = CreateMTensor(in1);
    mTensor mt_out = CreateMTensor(*output);

    ::musa::dnn::Status status;

    if (in0.dims() == 2 && in1.dims() == 2) {
      mMatMul op;
      op.SetTranspose(trans_a_, trans_b_);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
      mTensor mt_bias = CreateMTensor(bias_input);

      tensorflow::Allocator* tf_allocator =
          ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
      auto alloc_func =
          [tf_allocator](
              size_t size)
              -> std::unique_ptr<void, std::function<void(void*)>> {
        void* ptr = tf_allocator->AllocateRaw(256, size);
        auto deleter = [tf_allocator](void* p) {
          if (p) {
            tf_allocator->DeallocateRaw(p);
          }
        };
        return std::unique_ptr<void, std::function<void(void*)>>(ptr, deleter);
      };
      ::musa::dnn::MemoryMaintainer mm(alloc_func);
      status = op.RunWithBiasAdd(handle, mt_out, mt_a, mt_b, mt_bias, mm);
    } else {
      mBatchMatMul op;
      op.SetTranspose(trans_a_, trans_b_);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
      int64_t out_batch = bcast.output_batch_shape().num_elements();

      auto ReshapeTo3D = [out_batch](mTensor& mt, const Tensor& t) {
        int64_t dims = t.dims();
        int64_t rows = t.dim_size(dims - 2);
        int64_t cols = t.dim_size(dims - 1);
        int64_t batch = t.NumElements() / (rows * cols);
        if (dims != 3 || (batch == 1 && out_batch > 1)) {
          mt.SetNdInfo(
              {batch == 1 && out_batch > 1 ? out_batch : batch, rows, cols},
              {batch == 1 && out_batch > 1 ? 0 : rows * cols, cols, 1});
        }
      };
      ReshapeTo3D(mt_a, in0);
      ReshapeTo3D(mt_b, in1);
      mt_out.SetNdInfo({out_batch, m, n}, {m * n, n, 1});
      status = op.Run(handle, mt_out, mt_a, mt_b);
    }

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "MUSA MatMul/BatchMatMul execution failed in LinearRelu."));

    MUSA_KERNEL_TRACE_START("ApplyBiasAddRelu");
    ApplyBiasAddRelu(ctx, bias_input, mm_out_shape, mt_out,
                     in0.dims() == 2 && in1.dims() == 2);
    MUSA_KERNEL_TRACE_END("ApplyBiasAddRelu");
  }

  bool IsExpensive() override { return true; }

 private:
  bool trans_a_ = false;
  bool trans_b_ = false;
  bool tf32_enabled_ = false;  // TF32 acceleration enabled by default

  void ApplyBiasAddRelu(OpKernelContext* ctx, const Tensor& bias_input,
                        const TensorShape& mm_out_shape, mTensor& mt_out,
                        bool bias_already_applied) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_bias = CreateMTensor(bias_input);

    int channel_dim = mm_out_shape.dims() - 1;
    OP_REQUIRES(
        ctx, bias_input.dim_size(0) == mm_out_shape.dim_size(channel_dim),
        errors::InvalidArgument("Dimension mismatch in BiasAdd of LinearRelu"));

    int dims_cnt = mm_out_shape.dims();
    std::vector<int64_t> b_dims(dims_cnt, 1);
    std::vector<int64_t> b_strides(dims_cnt, 0);
    b_dims[channel_dim] = bias_input.dim_size(0);
    b_strides[channel_dim] = 1;

    mt_bias.SetNdInfo(dims_cnt, b_dims.data(), b_strides.data());

    if (!bias_already_applied) {
      mBinary bias_op;
      bias_op.SetMode(::musa::dnn::Binary::Mode::ADD);
      mStatus status = bias_op.Run(handle, mt_out, mt_out, mt_bias);

      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("MUSA BiasAdd failed in LinearRelu."));
    }

    // 3. Relu (In-place on current output)
    mUnary relu_op;
    relu_op.SetMode(::musa::dnn::Unary::Mode::RELU);
    auto status = relu_op.Run(handle, mt_out, mt_out);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Relu failed in LinearRelu."));
  }
};

#define REGISTER_MUSA_LINEAR_RELU(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("MusaLinearRelu").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaLinearReluOp<TYPE>);

REGISTER_MUSA_LINEAR_RELU(float);
REGISTER_MUSA_LINEAR_RELU(Eigen::half);
REGISTER_MUSA_LINEAR_RELU(bfloat16);
REGISTER_MUSA_LINEAR_RELU(double);

#undef REGISTER_MUSA_LINEAR_RELU
}  // namespace musa

REGISTER_OP("MusaLinearRelu")
    .Input("a: T")
    .Input("b: T")
    .Input("bias: T")
    .Output("product: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(::tensorflow::shape_inference::MatMulShape);

}  // namespace tensorflow

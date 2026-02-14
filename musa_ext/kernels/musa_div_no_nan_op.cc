#include <musa_runtime.h>

#include <cstdio>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace musa {

struct DivNoNanStrides {
  int s0, s1, s2, s3;
};
struct DivNoNanDims {
  int d0, d1, d2, d3;
};

template <typename T>
void LaunchDivNoNan(const T* in0, const T* in1, T* out, DivNoNanStrides s_in0,
                    DivNoNanStrides s_in1, DivNoNanDims dims,
                    int total_elements, musaStream_t stream);

template <typename T>
class MusaDivNoNanOp : public OpKernel {
 public:
  explicit MusaDivNoNanOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_0 = ctx->input(0);
    const Tensor& input_1 = ctx->input(1);

    BCast bcast(BCast::FromShape(input_0.shape()),
                BCast::FromShape(input_1.shape()));

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", input_0.shape().DebugString(),
                    " vs. ", input_1.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    musaStream_t stream = 0;

    auto pad_shape_to_4d = [](const TensorShape& s) -> TensorShape {
      TensorShape new_s = s;
      while (new_s.dims() < 4) new_s.InsertDim(0, 1);
      return new_s;
    };

    TensorShape s_out = pad_shape_to_4d(output_shape);
    TensorShape s_in0 = pad_shape_to_4d(input_0.shape());
    TensorShape s_in1 = pad_shape_to_4d(input_1.shape());

    DivNoNanDims k_dims;
    k_dims.d0 = s_out.dim_size(0);
    k_dims.d1 = s_out.dim_size(1);
    k_dims.d2 = s_out.dim_size(2);
    k_dims.d3 = s_out.dim_size(3);

    auto calc_strides = [&](const TensorShape& shape,
                            const TensorShape& out_s) -> DivNoNanStrides {
      int64_t raw_st[4];
      int64_t acc = 1;
      for (int i = 3; i >= 0; --i) {
        raw_st[i] = acc;
        acc *= shape.dim_size(i);
      }
      DivNoNanStrides st;
      int* st_arr = (int*)&st;
      for (int i = 0; i < 4; ++i) {
        if (shape.dim_size(i) == 1 && out_s.dim_size(i) > 1) {
          st_arr[i] = 0;
        } else {
          st_arr[i] = (int)raw_st[i];
        }
      }
      return st;
    };

    DivNoNanStrides k_in0_st = calc_strides(s_in0, s_out);
    DivNoNanStrides k_in1_st = calc_strides(s_in1, s_out);

    LaunchDivNoNan<T>(input_0.flat<T>().data(), input_1.flat<T>().data(),
                      output->flat<T>().data(), k_in0_st, k_in1_st, k_dims,
                      output->NumElements(), stream);
  }
};

#define REGISTER_MUSA_DIV_NO_NAN(TYPE)                           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DivNoNan").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaDivNoNanOp<TYPE>);

REGISTER_MUSA_DIV_NO_NAN(float);
REGISTER_MUSA_DIV_NO_NAN(double);

#undef REGISTER_MUSA_DIV_NO_NAN

}  // namespace musa
}  // namespace tensorflow
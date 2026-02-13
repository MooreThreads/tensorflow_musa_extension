#include <algorithm>
#include <numeric>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename Tlen>
class MusaSplitVOp : public OpKernel {
 public:
  explicit MusaSplitVOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    const int32 num_split = context->num_outputs();
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    const Tensor& split_tensor = context->input(1);
    const Tensor& split_dim_tensor = context->input(2);

    OP_REQUIRES(context, split_dim_tensor.NumElements() == 1,
                errors::InvalidArgument(
                    "split_dim_tensor must have exactly one element."));

    const int32 split_dim_orig = split_dim_tensor.flat<int32>()(0);
    const int32 split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;

    OP_REQUIRES(
        context, 0 <= split_dim && split_dim < input.dims(),
        errors::InvalidArgument("Split dim ", split_dim_orig, " out of range"));

    OP_REQUIRES(
        context,
        split_tensor.dims() == 1 && split_tensor.NumElements() == num_split,
        errors::InvalidArgument(
            "size of the split_tensor must be 1-D and match num_outputs"));

    auto split_sizes_d = split_tensor.vec<Tlen>();
    std::vector<Tlen> split_sizes_vec(num_split);

    for (int i = 0; i < num_split; ++i) split_sizes_vec[i] = split_sizes_d(i);

    Tlen input_size_split_dim =
        static_cast<Tlen>(input_shape.dim_size(split_dim));

    int neg_one_dim = -1;
    Tlen determined_size = 0;
    for (int d = 0; d < num_split; ++d) {
      Tlen size = split_sizes_vec[d];
      if (size == -1) {
        OP_REQUIRES(
            context, neg_one_dim == -1,
            errors::InvalidArgument("There can only be one -1 in the input."));
        neg_one_dim = d;
      } else {
        determined_size += size;
      }
    }

    if (neg_one_dim >= 0) {
      split_sizes_vec[neg_one_dim] = input_size_split_dim - determined_size;
    }

    Tlen total_len = 0;
    for (auto s : split_sizes_vec) {
      OP_REQUIRES(context, s >= 0,
                  errors::InvalidArgument("Split size must be >= 0"));
      total_len += s;
    }
    OP_REQUIRES(context, total_len == input_size_split_dim,
                errors::InvalidArgument(
                    "Sum of split sizes must match input dim size"));

    if (num_split == 1) {
      context->set_output(0, input);
      return;
    }

    auto& h = GetHandleByCtx(context);
    ::musa::dnn::Permute op;

    std::vector<int64_t> starts_mt(input.dims(), 0);
    std::vector<int64_t> strides_mt(input.dims(), 1);

    TensorShape out_shape_base = input_shape;

    for (int i = 0; i < num_split; ++i) {
      Tlen len = split_sizes_vec[i];

      out_shape_base.set_dim(split_dim, len);

      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, out_shape_base, &output));

      if (len == 0) continue;

      auto in_mt = CreateMTensor(input);
      auto out_mt = CreateMTensor(*output);

      MTOP_CHECK_OK(op.ConfigDimStrideForSlice(out_mt, in_mt, starts_mt.data(),
                                               strides_mt.data()),
                    "ConfigDimStrideForSlice", context);

      MTOP_CHECK_OK_RUN(op.Run(h, out_mt, in_mt), "Permute Run", context);

      starts_mt[split_dim] += len;
    }
  }
};

#define REGISTER_SPLIT_V(type, len_type)                        \
  REGISTER_KERNEL_BUILDER(Name("SplitV")                        \
                              .Device("MUSA")                   \
                              .TypeConstraint<type>("T")        \
                              .TypeConstraint<len_type>("Tlen") \
                              .HostMemory("size_splits")        \
                              .HostMemory("split_dim"),         \
                          MusaSplitVOp<len_type>)

#define REGISTER_SPLIT_V_ALL_LEN(type) \
  REGISTER_SPLIT_V(type, int32);       \
  REGISTER_SPLIT_V(type, int64);

REGISTER_SPLIT_V_ALL_LEN(float);
REGISTER_SPLIT_V_ALL_LEN(double);
REGISTER_SPLIT_V_ALL_LEN(Eigen::half);
REGISTER_SPLIT_V_ALL_LEN(Eigen::bfloat16);
REGISTER_SPLIT_V_ALL_LEN(int32);
REGISTER_SPLIT_V_ALL_LEN(int64);
REGISTER_SPLIT_V_ALL_LEN(bool);
REGISTER_SPLIT_V_ALL_LEN(uint8);

#undef REGISTER_SPLIT_V_ALL_LEN
#undef REGISTER_SPLIT_V

}  // namespace musa
}  // namespace tensorflow

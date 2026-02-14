#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace musa {

template <typename T, typename Tindex>
class UnsortedSegmentSumOp : public OpKernel {
 public:
  explicit UnsortedSegmentSumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data = ctx->input(0);
    const Tensor& segment_ids = ctx->input(1);
    const Tensor& num_segments_t = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_segments_t.shape()),
                errors::InvalidArgument("num_segments should be a scalar."));

    const Tindex num_segments = num_segments_t.scalar<Tindex>()();

    OP_REQUIRES(ctx, num_segments > 0,
                errors::InvalidArgument("num_segments should be positive."));

    const int64 N = segment_ids.NumElements();
    const int64 data_elements = data.NumElements();

    TensorShape output_shape;
    output_shape.AddDim(num_segments);
    TensorShape data_shape = data.shape();
    for (int i = segment_ids.dims(); i < data_shape.dims(); ++i) {
      output_shape.AddDim(data_shape.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    auto output_flat = output->flat_outer_dims<T>();
    output_flat.setZero();

    if (N == 0) {
      return;
    }

    OP_REQUIRES(ctx, data_elements % N == 0,
                errors::InvalidArgument(
                    "data size must be a multiple of segment_ids size."));

    const int64 M = data_elements / N;

    const auto data_flat = data.shaped<T, 2>({N, M});
    const auto segment_vec = segment_ids.flat<Tindex>();

    for (int64 i = 0; i < N; ++i) {
      Tindex idx = segment_vec(i);
      if (idx < 0 || idx >= num_segments) {
        continue;
      }
      for (int64 j = 0; j < M; ++j) {
        output_flat(idx, j) += data_flat(i, j);
      }
    }
  }
};

#define REGISTER_MUSA_SEGMENT_SUM(type, index_type)                   \
  REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentSum")                  \
                              .Device("MUSA")                         \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<index_type>("Tindices") \
                              .HostMemory("data")                     \
                              .HostMemory("segment_ids")              \
                              .HostMemory("num_segments")             \
                              .HostMemory("output"),                  \
                          UnsortedSegmentSumOp<type, index_type>)

#define REGISTER_MUSA_SEGMENT_SUM_ALL(type) \
  REGISTER_MUSA_SEGMENT_SUM(type, int32);   \
  REGISTER_MUSA_SEGMENT_SUM(type, int64);

REGISTER_MUSA_SEGMENT_SUM_ALL(float);
REGISTER_MUSA_SEGMENT_SUM_ALL(double);
REGISTER_MUSA_SEGMENT_SUM_ALL(int32);
REGISTER_MUSA_SEGMENT_SUM_ALL(int64);

#undef REGISTER_MUSA_SEGMENT_SUM_ALL
#undef REGISTER_MUSA_SEGMENT_SUM

}  // namespace musa
}  // namespace tensorflow

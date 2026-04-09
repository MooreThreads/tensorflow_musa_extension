#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/bcast.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaClipOp : public MusaOpKernel {
 public:
  explicit MusaClipOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    const Tensor& input_x = ctx->input(0);
    const Tensor& input_lo = ctx->input(1);
    const Tensor& input_hi = ctx->input(2);

    BCast bcast_x_lo(BCast::Vec(input_x.shape().dim_sizes()),
                     BCast::Vec(input_lo.shape().dim_sizes()));
    OP_REQUIRES(ctx, bcast_x_lo.IsValid(),
                errors::InvalidArgument("MusaClip: incompatible shapes between "
                                        "x and lo: ",
                                        input_x.shape().DebugString(), " vs ",
                                        input_lo.shape().DebugString()));

    TensorShape x_lo_shape = BCast::ToShape(bcast_x_lo.output_shape());

    BCast bcast_x_lo_hi(BCast::Vec(x_lo_shape.dim_sizes()),
                        BCast::Vec(input_hi.shape().dim_sizes()));
    OP_REQUIRES(
        ctx, bcast_x_lo_hi.IsValid(),
        errors::InvalidArgument(
            "MusaClip: incompatible shapes between broadcast(x, lo) and hi: ",
            x_lo_shape.DebugString(), " vs ", input_hi.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast_x_lo_hi.output_shape());

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor mt_x = CreateMTensor(input_x, format_);
    mTensor mt_lo = CreateMTensor(input_lo, format_);
    mTensor mt_hi = CreateMTensor(input_hi, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    mTernary ternary_op;
    auto mode_status = ternary_op.SetMode(::musa::dnn::Ternary::Mode::CLAMP);
    OP_REQUIRES(ctx, mode_status == mStatus::SUCCESS,
                errors::Internal("MUSA muDNN Ternary CLAMP SetMode failed. "
                                 "Status: ",
                                 static_cast<int>(mode_status)));

    auto run_status = ternary_op.Run(handle, mt_out, mt_x, mt_lo, mt_hi);
    OP_REQUIRES(ctx, run_status == mStatus::SUCCESS,
                errors::Internal("MUSA muDNN Ternary CLAMP execution failed. "
                                 "Status: ",
                                 static_cast<int>(run_status)));
  }
};

}  // namespace musa

REGISTER_OP("MusaClip")
    .Input("x: T")
    .Input("lo: T")
    .Input("hi: T")
    .Output("y: T")
    .Attr("T: {float, half, bfloat16, double, int32, int64}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using ::tensorflow::shape_inference::ShapeHandle;

      ShapeHandle x_shape = c->input(0);
      ShapeHandle lo_shape = c->input(1);
      ShapeHandle hi_shape = c->input(2);

      ShapeHandle x_lo_shape;
      ShapeHandle out_shape;

      if (!c->RankKnown(x_shape) || !c->RankKnown(lo_shape) ||
          !c->RankKnown(hi_shape)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      auto BroadcastTwoShapes =
          [&](ShapeHandle a, ShapeHandle b, ShapeHandle* out) -> Status {
        const int rank_a = c->Rank(a);
        const int rank_b = c->Rank(b);
        const int out_rank = std::max(rank_a, rank_b);

        std::vector<::tensorflow::shape_inference::DimensionHandle> dims;
        dims.reserve(out_rank);

        for (int i = 0; i < out_rank; ++i) {
          const int ia = rank_a - 1 - i;
          const int ib = rank_b - 1 - i;

          auto dim_a = (ia >= 0) ? c->Dim(a, ia) : c->MakeDim(1);
          auto dim_b = (ib >= 0) ? c->Dim(b, ib) : c->MakeDim(1);

          if (c->ValueKnown(dim_a) && c->Value(dim_a) == 1) {
            dims.push_back(dim_b);
            continue;
          }
          if (c->ValueKnown(dim_b) && c->Value(dim_b) == 1) {
            dims.push_back(dim_a);
            continue;
          }

          if (c->ValueKnown(dim_a) && c->ValueKnown(dim_b)) {
            if (c->Value(dim_a) != c->Value(dim_b)) {
              return errors::InvalidArgument(
                  "MusaClip shape inference failed: dimensions not "
                  "broadcastable.");
            }
            dims.push_back(dim_a);
            continue;
          }

          ::tensorflow::shape_inference::DimensionHandle merged;
          TF_RETURN_IF_ERROR(c->Merge(dim_a, dim_b, &merged));
          dims.push_back(merged);
        }

        std::reverse(dims.begin(), dims.end());
        *out = c->MakeShape(dims);
        return Status::OK();
      };

      TF_RETURN_IF_ERROR(BroadcastTwoShapes(x_shape, lo_shape, &x_lo_shape));
      TF_RETURN_IF_ERROR(BroadcastTwoShapes(x_lo_shape, hi_shape, &out_shape));

      c->set_output(0, out_shape);
      return Status::OK();
    });

}  // namespace tensorflow

#define REGISTER_MUSA_CLIP(TYPE)                                             \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("MusaClip").Device(DEVICE_MTGPU).TypeConstraint<TYPE>("T"),       \
      ::tensorflow::musa::MusaClipOp<TYPE>)

REGISTER_MUSA_CLIP(float);
REGISTER_MUSA_CLIP(double);
REGISTER_MUSA_CLIP(::tensorflow::int32);
REGISTER_MUSA_CLIP(::tensorflow::int64);
REGISTER_MUSA_CLIP(Eigen::half);
REGISTER_MUSA_CLIP(::tensorflow::bfloat16);

#undef REGISTER_MUSA_CLIP

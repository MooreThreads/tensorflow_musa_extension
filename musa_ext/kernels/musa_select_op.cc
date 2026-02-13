#include <mudnn.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

namespace {
mType GetMusaTypeLocal(DataType t) {
  switch (t) {
    case DataType::DT_FLOAT:
      return mType::FLOAT;
    case DataType::DT_HALF:
      return mType::HALF;
    case DataType::DT_BFLOAT16:
      return mType::BFLOAT16;
    case DataType::DT_INT32:
      return mType::INT32;
    case DataType::DT_INT64:
      return mType::INT64;
    case DataType::DT_DOUBLE:
      return mType::DOUBLE;
    case DataType::DT_BOOL:
      return mType::BOOL;
    case DataType::DT_INT8:
      return mType::INT8;
    case DataType::DT_UINT8:
      return mType::UINT8;
    default:
      return mType::FLOAT;
  }
}
}  // namespace

template <typename T>
class MusaSelectOp : public MusaOpKernel {
 public:
  explicit MusaSelectOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& cond = ctx->input(0);
    const Tensor& then_t = ctx->input(1);
    const Tensor& else_t = ctx->input(2);

    BCast bcast_te(BCast::FromShape(then_t.shape()),
                   BCast::FromShape(else_t.shape()));
    if (!bcast_te.IsValid()) {
      ctx->SetStatus(
          errors::InvalidArgument("Incompatible shapes: then vs else"));
      return;
    }
    TensorShape te_shape = BCast::ToShape(bcast_te.output_shape());

    bool use_legacy_broadcast = false;
    TensorShape output_shape;

    BCast bcast_final(BCast::FromShape(cond.shape()),
                      BCast::FromShape(te_shape));

    if (bcast_final.IsValid()) {
      output_shape = BCast::ToShape(bcast_final.output_shape());
      use_legacy_broadcast = false;
    } else if (cond.dims() == 1 && te_shape.dims() > 1 &&
               cond.dim_size(0) == te_shape.dim_size(0)) {
      output_shape = te_shape;
      use_legacy_broadcast = true;
    } else {
      ctx->SetStatus(
          errors::InvalidArgument("Incompatible shapes: cond vs (then/else)"));
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    MusaDevice* device = reinterpret_cast<MusaDevice*>(ctx->device());
    auto& handle = device->mudnn_handle();

    std::vector<std::vector<int64_t>> shape_storage;
    shape_storage.reserve(10);

    auto CreateMTensor = [&](const Tensor& input,
                             bool force_left_align = false) -> mTensor {
      mTensor mt;
      mt.SetAddr(const_cast<void*>(
          static_cast<const void*>(input.tensor_data().data())));
      mt.SetType(GetMusaTypeLocal(input.dtype()));
      mt.SetFormat(mFormat::NCHW);

      int target_rank = output_shape.dims();
      std::vector<int64_t> t_dims(target_rank);
      std::vector<int64_t> i_strides(target_rank, 0);

      for (int i = 0; i < target_rank; ++i)
        t_dims[i] = output_shape.dim_size(i);

      int input_rank = input.dims();
      std::vector<int64_t> dense_strides(input_rank, 1);
      if (input_rank > 0) {
        for (int i = input_rank - 2; i >= 0; --i)
          dense_strides[i] = dense_strides[i + 1] * input.dim_size(i + 1);
      }

      if (force_left_align) {
        if (input_rank == 1) {
          i_strides[0] = dense_strides[0];
          for (int i = 1; i < target_rank; ++i) i_strides[i] = 0;
        }
      } else {
        for (int i = 1; i <= target_rank; ++i) {
          int target_idx = target_rank - i;
          int input_idx = input_rank - i;
          if (input_idx >= 0) {
            if (input.dim_size(input_idx) == t_dims[target_idx]) {
              i_strides[target_idx] = dense_strides[input_idx];
            } else {
              i_strides[target_idx] = 0;
            }
          } else {
            i_strides[target_idx] = 0;
          }
        }
      }

      shape_storage.push_back(t_dims);
      shape_storage.push_back(i_strides);
      mt.SetNdInfo(target_rank, shape_storage[shape_storage.size() - 2].data(),
                   shape_storage[shape_storage.size() - 1].data());
      return mt;
    };

    auto cond_mt = CreateMTensor(cond, use_legacy_broadcast);

    auto then_mt = CreateMTensor(then_t, false);
    auto else_mt = CreateMTensor(else_t, false);

    auto out_mt = CreateMTensor(*output, false);

    ::musa::dnn::Ternary op;
    op.SetMode(::musa::dnn::Ternary::Mode::SELECT);

    auto status = op.Run(handle, out_mt, cond_mt, then_mt, else_mt);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Select run failed"));
  }
};

#define REGISTER_SELECT(T)                                          \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Select").Device(DEVICE_MTGPU).TypeConstraint<T>("T"),   \
      MusaSelectOp<T>);                                             \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("SelectV2").Device(DEVICE_MTGPU).TypeConstraint<T>("T"), \
      MusaSelectOp<T>);

REGISTER_SELECT(float);
REGISTER_SELECT(double);
REGISTER_SELECT(int32);
REGISTER_SELECT(int64);
REGISTER_SELECT(bool);
REGISTER_SELECT(Eigen::half);
REGISTER_SELECT(Eigen::bfloat16);

#undef REGISTER_SELECT

}  // namespace musa
}  // namespace tensorflow
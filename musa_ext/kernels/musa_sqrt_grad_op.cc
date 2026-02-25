#include <musa_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSqrtGradOp : public OpKernel {
 public:
  explicit MusaSqrtGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& y = ctx->input(0);
    const Tensor& dy = ctx->input(1);

    BCast bcast(BCast::FromShape(y.shape()), BCast::FromShape(dy.shape()),
                false);

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes for SqrtGrad: ",
                                        "y: ", y.shape().DebugString(),
                                        ", dy: ", dy.shape().DebugString()));

    TensorShape output_shape = y.shape();

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &dx));

    if (output_shape.num_elements() == 0) return;

    size_t y_bytes = y.NumElements() * sizeof(T);
    size_t dy_bytes = dy.NumElements() * sizeof(T);
    size_t dx_bytes = dx->NumElements() * sizeof(T);

    std::vector<T> h_y(y.NumElements());
    std::vector<T> h_dy(dy.NumElements());
    std::vector<T> h_dx(dx->NumElements(), static_cast<T>(0));

    // Use async memcpy with stream for D2H copies
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    mStatus status_y = MusaMemcpyAsyncD2H(h_y.data(), y.flat<T>().data(),
                                          y_bytes, stream);
    mStatus status_dy = MusaMemcpyAsyncD2H(h_dy.data(), dy.flat<T>().data(),
                                           dy_bytes, stream);

    OP_REQUIRES(ctx, status_y == mStatus::SUCCESS && status_dy == mStatus::SUCCESS,
                errors::Internal("MUSA SqrtGrad: Memcpy DeviceToHost failed"));

    // Synchronize stream before CPU computation
    musaStreamSynchronize(stream);

    const double kSafeThreshold = 1e-9;

    auto compute_safe_grad = [&](double val_y, double val_dy) -> double {
      double res = 0.0;

      if (std::abs(val_y) < kSafeThreshold || std::isnan(val_y) ||
          std::isinf(val_y)) {
        return 0.0;
      }

      if (std::isnan(val_dy) || std::isinf(val_dy)) {
        return 0.0;
      }

      res = 0.5 * val_dy / val_y;

      if (std::isnan(res) || std::isinf(res)) {
        return 0.0;
      }

      return res;
    };

    if (y.shape() == dy.shape()) {
      size_t n = h_y.size();
      for (size_t i = 0; i < n; ++i) {
        double val_y = static_cast<double>(h_y[i]);
        double val_dy = static_cast<double>(h_dy[i]);
        h_dx[i] = static_cast<T>(compute_safe_grad(val_y, val_dy));
      }
    } else {
      TensorShape bcast_shape = BCast::ToShape(bcast.result_shape());

      auto pad_shape_to_nd = [](const TensorShape& s,
                                int n_dims) -> TensorShape {
        TensorShape new_s = s;
        while (new_s.dims() < n_dims) new_s.InsertDim(0, 1);
        return new_s;
      };

      int rank = bcast_shape.dims();
      std::vector<int64_t> y_strides(rank);
      std::vector<int64_t> dy_strides(rank);
      std::vector<int64_t> dx_strides(rank);

      auto compute_strides_for_broadcast = [&](const TensorShape& shape,
                                               std::vector<int64_t>& strides,
                                               const TensorShape& result_s) {
        TensorShape padded = pad_shape_to_nd(shape, rank);
        std::vector<int64_t> dense_strides(rank, 0);
        int64_t acc = 1;
        for (int i = rank - 1; i >= 0; --i) {
          dense_strides[i] = acc;
          acc *= padded.dim_size(i);
        }
        for (int i = 0; i < rank; ++i) {
          if (padded.dim_size(i) == 1 && result_s.dim_size(i) > 1) {
            strides[i] = 0;
          } else {
            strides[i] = dense_strides[i];
          }
        }
      };

      compute_strides_for_broadcast(y.shape(), y_strides, bcast_shape);
      compute_strides_for_broadcast(dy.shape(), dy_strides, bcast_shape);
      compute_strides_for_broadcast(output_shape, dx_strides, bcast_shape);

      int64_t total_elements = bcast_shape.num_elements();

      std::fill(h_dx.begin(), h_dx.end(), static_cast<T>(0));

      for (int64_t i = 0; i < total_elements; ++i) {
        int64_t temp = i;
        int64_t idx_y = 0;
        int64_t idx_dy = 0;
        int64_t idx_dx = 0;

        for (int d = rank - 1; d >= 0; --d) {
          int64_t dim_size = bcast_shape.dim_size(d);
          int64_t coord = temp % dim_size;
          temp /= dim_size;
          idx_y += coord * y_strides[d];
          idx_dy += coord * dy_strides[d];
          idx_dx += coord * dx_strides[d];
        }

        double val_y = static_cast<double>(h_y[idx_y]);
        double val_dy = static_cast<double>(h_dy[idx_dy]);

        double grad_contribution = compute_safe_grad(val_y, val_dy);
        h_dx[idx_dx] += static_cast<T>(grad_contribution);
      }
    }

    // Use async memcpy with stream for H2D copy
    mStatus status_dx = MusaMemcpyAsyncH2D(dx->flat<T>().data(), h_dx.data(),
                                           dx_bytes, stream);
    OP_REQUIRES(ctx, status_dx == mStatus::SUCCESS,
                errors::Internal("MUSA SqrtGrad: Memcpy HostToDevice failed"));

    // Synchronize stream to ensure copy completes
    musaStreamSynchronize(stream);
  }
};

#define REGISTER_MUSA_SQRT_GRAD(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SqrtGrad").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaSqrtGradOp<TYPE>)

REGISTER_MUSA_SQRT_GRAD(float);
REGISTER_MUSA_SQRT_GRAD(double);
REGISTER_MUSA_SQRT_GRAD(Eigen::half);
REGISTER_MUSA_SQRT_GRAD(bfloat16);

#undef REGISTER_MUSA_SQRT_GRAD

}  // namespace musa
}  // namespace tensorflow
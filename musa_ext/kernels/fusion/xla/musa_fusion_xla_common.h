#pragma once

#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/bcast.h"
#include "tsl/platform/tensor_float_32_utils.h"
#include "xla/client/lib/arithmetic.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/math.h"
#include "xla/client/lib/matrix.h"
#include "xla/client/xla_builder.h"

namespace tensorflow {

inline std::vector<int64_t> DimSizes(const TensorShape& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.dims());
  for (int i = 0; i < shape.dims(); ++i) {
    dims.push_back(shape.dim_size(i));
  }
  return dims;
}

inline std::vector<int64_t> IotaDims(int64_t n) {
  std::vector<int64_t> dims(n);
  std::iota(dims.begin(), dims.end(), 0);
  return dims;
}

inline void BroadcastToShape(XlaOpKernelContext* ctx, xla::XlaOp* op,
                             const TensorShape& shape) {
  auto xla_shape_or = op->builder()->GetShape(*op);
  OP_REQUIRES_OK(ctx, xla_shape_or.status());
  if (xla_shape_or.value().dimensions() == DimSizes(shape)) {
    return;
  }
  auto broadcast = BroadcastTo(*op, shape.dim_sizes());
  OP_REQUIRES_OK(ctx, broadcast.status());
  *op = broadcast.value();
}

inline void BroadcastLeftAligned1DToShape(XlaOpKernelContext* ctx,
                                          xla::XlaOp* op,
                                          const TensorShape& input_shape,
                                          const TensorShape& output_shape) {
  OP_REQUIRES(ctx, input_shape.dims() == 1 && output_shape.dims() >= 1 &&
                       input_shape.dim_size(0) == output_shape.dim_size(0),
              errors::InvalidArgument("left-aligned broadcast requires a 1D "
                                      "input matching output dim 0"));
  *op = xla::BroadcastInDim(*op, DimSizes(output_shape), {0});
}

inline void BroadcastMaskToShape(XlaOpKernelContext* ctx, xla::XlaOp* op,
                                 const TensorShape& input_shape,
                                 const TensorShape& output_shape) {
  auto xla_shape_or = op->builder()->GetShape(*op);
  OP_REQUIRES_OK(ctx, xla_shape_or.status());
  if (xla_shape_or.value().dimensions() == DimSizes(output_shape)) {
    return;
  }
  BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(output_shape));
  if (bcast.IsValid() && BCast::ToShape(bcast.output_shape()) == output_shape) {
    BroadcastToShape(ctx, op, output_shape);
    return;
  }
  BroadcastLeftAligned1DToShape(ctx, op, input_shape, output_shape);
}

inline bool BroadcastShapeOrLeftAlignedMask(const TensorShape& base,
                                            const TensorShape& candidate,
                                            TensorShape* output,
                                            bool* left_aligned) {
  BCast bcast(BCast::FromShape(base), BCast::FromShape(candidate));
  if (bcast.IsValid()) {
    *output = BCast::ToShape(bcast.output_shape());
    *left_aligned = false;
    return true;
  }
  if (candidate.dims() == 1 && base.dims() >= 2 &&
      candidate.dim_size(0) == base.dim_size(0)) {
    *output = base;
    *left_aligned = true;
    return true;
  }
  return false;
}

inline void ShapeOfXlaOp(XlaOpKernelContext* ctx, xla::XlaOp op,
                         TensorShape* shape) {
  auto shape_or = op.builder()->GetShape(op);
  OP_REQUIRES_OK(ctx, shape_or.status());
  OP_REQUIRES_OK(
      ctx, TensorShape::BuildTensorShape(shape_or.value().dimensions(), shape));
}

inline xla::XlaOp ConvertTo(xla::XlaOp op, DataType dtype) {
  return XlaHelpers::ConvertElementType(op, dtype);
}

inline void TableRow(XlaOpKernelContext* ctx, xla::XlaOp table,
                     const TensorShape& table_shape, int row,
                     xla::XlaOp* output) {
  OP_REQUIRES(ctx, table_shape.dims() == 2,
              errors::InvalidArgument("table input must be rank 2"));
  OP_REQUIRES(ctx, row >= 0 && row < table_shape.dim_size(0),
              errors::InvalidArgument("table row index out of range"));
  *output = xla::Reshape(xla::Slice(table, {row, 0},
                                    {row + 1, table_shape.dim_size(1)},
                                    {1, 1}),
                         {table_shape.dim_size(1)});
}

inline xla::XlaOp BatchMatMul(xla::XlaOp a, bool transpose_a, xla::XlaOp b,
                              bool transpose_b) {
  xla::PrecisionConfig::Precision precision =
      tsl::tensor_float_32_execution_enabled() ? xla::PrecisionConfig::DEFAULT
                                               : xla::PrecisionConfig::HIGHEST;
  return xla::BatchDot(a, transpose_a, b, transpose_b, precision,
                       /*preferred_element_type=*/std::nullopt);
}

inline xla::XlaOp BroadcastRows(xla::XlaOp op, int64_t num_rows,
                                int64_t row_size) {
  return xla::BroadcastInDim(op, {num_rows, row_size}, {0, 1});
}

inline xla::XlaOp BroadcastCols(xla::XlaOp op, int64_t num_rows,
                                int64_t row_size) {
  return xla::BroadcastInDim(op, {num_rows, row_size}, {0, 1});
}

inline xla::XlaOp OnesColumn(xla::XlaBuilder* b, DataType dtype,
                             int64_t rows) {
  return xla::Broadcast(XlaHelpers::FloatLiteral(b, dtype, 1.0), {rows, 1});
}

inline xla::XlaOp OnesRow(xla::XlaBuilder* b, DataType dtype, int64_t cols) {
  return xla::Broadcast(XlaHelpers::FloatLiteral(b, dtype, 1.0), {1, cols});
}

inline xla::XlaOp RowWiseSum2D(XlaOpKernelContext* ctx, xla::XlaOp matrix,
                               DataType dtype, int64_t num_rows,
                               int64_t row_size) {
  xla::DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);
  xla::PrecisionConfig precision_config;
  precision_config.add_operand_precision(
      tsl::tensor_float_32_execution_enabled() ? xla::PrecisionConfig::DEFAULT
                                               : xla::PrecisionConfig::HIGHEST);
  precision_config.add_operand_precision(
      tsl::tensor_float_32_execution_enabled() ? xla::PrecisionConfig::DEFAULT
                                               : xla::PrecisionConfig::HIGHEST);
  return xla::DotGeneral(matrix, OnesColumn(ctx->builder(), dtype, row_size),
                         dnums, &precision_config,
                         /*preferred_element_type=*/std::nullopt);
}

inline xla::XlaOp SumAcrossRows2D(XlaOpKernelContext* ctx, xla::XlaOp matrix,
                                  DataType dtype, int64_t num_rows,
                                  int64_t row_size) {
  xla::DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);
  xla::PrecisionConfig precision_config;
  precision_config.add_operand_precision(
      tsl::tensor_float_32_execution_enabled() ? xla::PrecisionConfig::DEFAULT
                                               : xla::PrecisionConfig::HIGHEST);
  precision_config.add_operand_precision(
      tsl::tensor_float_32_execution_enabled() ? xla::PrecisionConfig::DEFAULT
                                               : xla::PrecisionConfig::HIGHEST);
  return xla::DotGeneral(OnesRow(ctx->builder(), dtype, num_rows), matrix,
                         dnums, &precision_config,
                         /*preferred_element_type=*/std::nullopt);
}

inline void CompileNormalize(XlaOpKernelContext* ctx, bool use_affine,
                             float epsilon, float max_std,
                             bool layernorm_eps_inside_sqrt);

struct TensorDotDims {
  int64_t a_batch_size = 1;
  int64_t a_contract_size = 1;
  int64_t b_contract_size = 1;
  int64_t b_batch_size = 1;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> a_perm;
  std::vector<int64_t> b_perm;
};

inline Status ComputeTensorDotDims(const TensorShape& a_shape,
                                   const TensorShape& b_shape,
                                   const std::vector<int>& axes_a,
                                   const std::vector<int>& axes_b,
                                   TensorDotDims* dims) {
  const int a_rank = a_shape.dims();
  const int b_rank = b_shape.dims();
  if (axes_a.size() != axes_b.size()) {
    return errors::InvalidArgument("axes_a and axes_b must have same size");
  }

  std::vector<int> norm_axes_a(axes_a.size());
  std::vector<int> norm_axes_b(axes_b.size());
  for (size_t i = 0; i < axes_a.size(); ++i) {
    norm_axes_a[i] = axes_a[i] < 0 ? axes_a[i] + a_rank : axes_a[i];
    norm_axes_b[i] = axes_b[i] < 0 ? axes_b[i] + b_rank : axes_b[i];
    if (norm_axes_a[i] < 0 || norm_axes_a[i] >= a_rank ||
        norm_axes_b[i] < 0 || norm_axes_b[i] >= b_rank) {
      return errors::InvalidArgument("tensordot contraction axis out of range");
    }
    if (a_shape.dim_size(norm_axes_a[i]) != b_shape.dim_size(norm_axes_b[i])) {
      return errors::InvalidArgument("tensordot contraction dimensions differ");
    }
  }

  std::unordered_set<int> a_contract(norm_axes_a.begin(), norm_axes_a.end());
  std::unordered_set<int> b_contract(norm_axes_b.begin(), norm_axes_b.end());

  dims->a_perm.clear();
  dims->b_perm.clear();
  dims->output_dims.clear();
  dims->a_batch_size = 1;
  dims->a_contract_size = 1;
  dims->b_contract_size = 1;
  dims->b_batch_size = 1;

  for (int i = 0; i < a_rank; ++i) {
    if (a_contract.find(i) == a_contract.end()) {
      dims->a_perm.push_back(i);
      dims->output_dims.push_back(a_shape.dim_size(i));
      dims->a_batch_size *= a_shape.dim_size(i);
    }
  }
  for (int axis : norm_axes_a) {
    dims->a_perm.push_back(axis);
    dims->a_contract_size *= a_shape.dim_size(axis);
  }

  for (int axis : norm_axes_b) {
    dims->b_perm.push_back(axis);
    dims->b_contract_size *= b_shape.dim_size(axis);
  }
  for (int i = 0; i < b_rank; ++i) {
    if (b_contract.find(i) == b_contract.end()) {
      dims->b_perm.push_back(i);
      dims->output_dims.push_back(b_shape.dim_size(i));
      dims->b_batch_size *= b_shape.dim_size(i);
    }
  }

  return OkStatus();
}

inline void TensorDot(XlaOpKernelContext* ctx, xla::XlaOp a,
                      const TensorShape& a_shape, xla::XlaOp b,
                      const TensorShape& b_shape,
                      const std::vector<int>& axes_a,
                      const std::vector<int>& axes_b, TensorDotDims* dims,
                      xla::XlaOp* output) {
  OP_REQUIRES_OK(ctx,
                 ComputeTensorDotDims(a_shape, b_shape, axes_a, axes_b, dims));
  xla::XlaOp a_perm = xla::Transpose(a, dims->a_perm);
  xla::XlaOp b_perm = xla::Transpose(b, dims->b_perm);
  xla::XlaOp a_2d =
      xla::Reshape(a_perm, {dims->a_batch_size, dims->a_contract_size});
  xla::XlaOp b_2d =
      xla::Reshape(b_perm, {dims->b_contract_size, dims->b_batch_size});
  xla::XlaOp y_2d =
      BatchMatMul(a_2d, /*transpose_a=*/false, b_2d, /*transpose_b=*/false);
  *output = xla::Reshape(y_2d, dims->output_dims);
}

inline void CompileNormalize(XlaOpKernelContext* ctx, bool use_affine,
                             float epsilon, float max_std,
                             bool layernorm_eps_inside_sqrt) {
  const TensorShape x_shape = ctx->InputShape(0);
  OP_REQUIRES(ctx, x_shape.dims() >= 1,
              errors::InvalidArgument("normalization input rank must be >= 1"));
  const int64_t rank = x_shape.dims();
  const int64_t last_dim = x_shape.dim_size(rank - 1);
  OP_REQUIRES(ctx, last_dim > 0,
              errors::InvalidArgument("normalization last dimension must be "
                                      "positive"));

  const DataType out_type = ctx->input_type(0);
  const DataType compute_type = out_type == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;
  xla::XlaBuilder* b = ctx->builder();
  xla::XlaOp x = ConvertTo(ctx->Input(0), compute_type);
  const int64_t num_rows = x_shape.num_elements() / last_dim;
  xla::XlaOp x_2d = xla::Reshape(x, {num_rows, last_dim});
  xla::XlaOp divisor = XlaHelpers::FloatLiteral(b, compute_type, last_dim);

  xla::XlaOp mean_col =
      RowWiseSum2D(ctx, x_2d, compute_type, num_rows, last_dim) / divisor;
  xla::XlaOp centered_2d = x_2d - BroadcastCols(mean_col, num_rows, last_dim);
  xla::XlaOp variance_col =
      RowWiseSum2D(ctx, centered_2d * centered_2d, compute_type, num_rows,
                   last_dim) /
      divisor;

  xla::XlaOp denom_col;
  if (layernorm_eps_inside_sqrt) {
    denom_col =
        xla::Sqrt(variance_col + XlaHelpers::FloatLiteral(b, compute_type,
                                                          epsilon));
  } else {
    xla::XlaOp std = xla::Sqrt(xla::Max(
        variance_col, XlaHelpers::FloatLiteral(b, compute_type, 0.0)));
    xla::XlaOp lo = XlaHelpers::FloatLiteral(b, compute_type, epsilon);
    xla::XlaOp hi = XlaHelpers::FloatLiteral(b, compute_type, max_std);
    denom_col = xla::Clamp(lo, std, hi);
  }
  xla::XlaOp y_2d =
      centered_2d / BroadcastCols(denom_col, num_rows, last_dim);

  if (use_affine) {
    xla::XlaOp gamma = ConvertTo(ctx->Input(1), compute_type);
    xla::XlaOp beta = ConvertTo(ctx->Input(2), compute_type);
    gamma = xla::BroadcastInDim(gamma, {num_rows, last_dim}, {1});
    beta = xla::BroadcastInDim(beta, {num_rows, last_dim}, {1});
    y_2d = y_2d * gamma + beta;
  }

  ctx->SetOutput(
      0, ConvertTo(xla::Reshape(y_2d, DimSizes(x_shape)), out_type));
}

}  // namespace tensorflow

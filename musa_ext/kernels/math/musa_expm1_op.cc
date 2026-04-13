/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * @file musa_expm1_op.cc
 * @brief MUSA implementation of TensorFlow Expm1 operator.
 *
 * The Expm1 operator computes exp(x) - 1 with improved numerical precision
 * for values of x near zero. This is the inverse operation of Log1p.
 *
 * Mathematical definition: expm1(x) = e^x - 1
 *
 * Use cases:
 * - Financial calculations (continuous compounding with small rates)
 * - Scientific computing requiring high precision
 * - Machine learning activation functions
 * - Statistical computations
 */

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

// Forward declarations of kernel launchers

template <typename T>
void LaunchExpm1(const T* src, T* dst, int n, musaStream_t stream);

template <typename T>
void LaunchExpm1Int(const T* src, T* dst, int n, musaStream_t stream);

/**
 * @brief MUSA implementation of Expm1 operator for floating point types.
 *
 * Computes exp(x) - 1 element-wise with improved numerical stability
 * for small values of x. This is a lightweight element-wise operation.
 *
 * @tparam T Data type (float, double, half, bfloat16)
 */
template <typename T>
class MusaExpm1Op : public MusaOpKernel {
 public:
  explicit MusaExpm1Op(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  /**
   * @brief Returns false as this is a lightweight element-wise operation.
   */
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    const int64_t size = input.NumElements();
    if (size == 0) return;

    musaStream_t stream = GetMusaStreamByCtx(ctx);

    LaunchExpm1<T>(input.flat<T>().data(), output->flat<T>().data(),
                   static_cast<int>(size), stream);

    auto kernel_status = musaGetLastError();
    OP_REQUIRES(ctx, kernel_status == musaSuccess,
                errors::Internal("MUSA Expm1 kernel failed: ",
                                 musaGetErrorString(kernel_status)));
  }
};

/**
 * @brief MUSA implementation of Expm1 operator for integer types.
 *
 * Integer inputs are converted to float for computation, then rounded back.
 *
 * @tparam T Integer data type (int32, int64)
 */
template <typename T>
class MusaExpm1IntOp : public MusaOpKernel {
 public:
  explicit MusaExpm1IntOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    const int64_t size = input.NumElements();
    if (size == 0) return;

    musaStream_t stream = GetMusaStreamByCtx(ctx);

    LaunchExpm1Int<T>(input.flat<T>().data(), output->flat<T>().data(),
                      static_cast<int>(size), stream);

    auto kernel_status = musaGetLastError();
    OP_REQUIRES(ctx, kernel_status == musaSuccess,
                errors::Internal("MUSA Expm1 kernel failed: ",
                                 musaGetErrorString(kernel_status)));
  }
};

// Register kernels for floating point types
#define REGISTER_MUSA_EXPM1(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Expm1").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaExpm1Op<TYPE>);

REGISTER_MUSA_EXPM1(float);
REGISTER_MUSA_EXPM1(double);
REGISTER_MUSA_EXPM1(Eigen::half);
REGISTER_MUSA_EXPM1(bfloat16);

#undef REGISTER_MUSA_EXPM1

// Register kernels for integer types
#define REGISTER_MUSA_EXPM1_INT(TYPE)                         \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Expm1").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaExpm1IntOp<TYPE>);

REGISTER_MUSA_EXPM1_INT(int32);
REGISTER_MUSA_EXPM1_INT(int64);

#undef REGISTER_MUSA_EXPM1_INT

}  // namespace musa
}  // namespace tensorflow
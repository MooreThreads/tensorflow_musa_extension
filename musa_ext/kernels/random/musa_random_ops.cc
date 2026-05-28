// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// MUSA Random Ops - Unified implementation for random number generation
// Supports: RandomUniform, RandomUniformInt, RandomStandardNormal,
// TruncatedNormal
//
// This implementation uses Philox random number generator (same as CUDA)
// to ensure bit-wise identical results with CUDA when possible.
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// MUSA Random Ops - Unified implementation for random number generation
// Supports: RandomUniform, RandomUniformInt, RandomStandardNormal,
// TruncatedNormal
//
// This implementation uses Philox random number generator (same as CUDA)
// to ensure bit-wise identical results with CUDA when possible.

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/util/guarded_philox_random.h"

struct MusaPhiloxState {
  uint32_t counter[4];
  uint32_t key[2];
};

extern "C" {
void LaunchRandomUniform_float(void* stream, int64_t n, int num_blocks,
                               int block_size, MusaPhiloxState state,
                               float* output);
void LaunchRandomUniform_double(void* stream, int64_t n, int num_blocks,
                                int block_size, MusaPhiloxState state,
                                double* output);
void LaunchRandomUniform_half(void* stream, int64_t n, int num_blocks,
                              int block_size, MusaPhiloxState state,
                              Eigen::half* output);
void LaunchRandomUniform_bfloat16(void* stream, int64_t n, int num_blocks,
                                  int block_size, MusaPhiloxState state,
                                  tensorflow::bfloat16* output);
void LaunchRandomUniformGreaterEqual_bool(void* stream, int64_t n,
                                          int num_blocks, int block_size,
                                          MusaPhiloxState state,
                                          uint32_t threshold_bits,
                                          bool* output);
void LaunchRandomUniformInt_int(void* stream, int64_t n, int num_blocks,
                                int block_size, MusaPhiloxState state,
                                int minval, int maxval, int* output);
void LaunchRandomUniformInt_int64_t(void* stream, int64_t n, int num_blocks,
                                    int block_size, MusaPhiloxState state,
                                    int64_t minval, int64_t maxval,
                                    int64_t* output);
void LaunchRandomStandardNormal_float(void* stream, int64_t n, int num_blocks,
                                      int block_size, MusaPhiloxState state,
                                      float* output);
void LaunchRandomStandardNormal_double(void* stream, int64_t n, int num_blocks,
                                       int block_size, MusaPhiloxState state,
                                       double* output);
void LaunchRandomStandardNormal_half(void* stream, int64_t n, int num_blocks,
                                     int block_size, MusaPhiloxState state,
                                     Eigen::half* output);
void LaunchRandomStandardNormal_bfloat16(void* stream, int64_t n,
                                         int num_blocks, int block_size,
                                         MusaPhiloxState state,
                                         tensorflow::bfloat16* output);
void LaunchTruncatedNormal_float(void* stream, int64_t n, int num_blocks,
                                 int block_size, MusaPhiloxState state,
                                 float* output);
void LaunchTruncatedNormal_double(void* stream, int64_t n, int num_blocks,
                                  int block_size, MusaPhiloxState state,
                                  double* output);
void LaunchTruncatedNormal_half(void* stream, int64_t n, int num_blocks,
                                int block_size, MusaPhiloxState state,
                                Eigen::half* output);
void LaunchTruncatedNormal_bfloat16(void* stream, int64_t n, int num_blocks,
                                    int block_size, MusaPhiloxState state,
                                    tensorflow::bfloat16* output);
}

namespace tensorflow {
namespace musa {

template <typename T>
void* GetStream(OpKernelContext* ctx) {
  auto* device = GetDeviceByCtx(ctx);
  return device ? device->GetStream() : nullptr;
}

// ==========================================
// RandomUniform Op
// ==========================================
template <typename T>
class MusaRandomUniformOp : public MusaOpKernel {
 public:
  explicit MusaRandomUniformOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_tensor = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    int64_t n = output->NumElements();
    if (n == 0) return;

    auto philox = generator_.ReserveSamples32(n);
    MusaPhiloxState state;
    std::memcpy(&state, &philox, sizeof(MusaPhiloxState));

    const int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    if (num_blocks > 4096) num_blocks = 4096;

    void* stream = GetStream<T>(ctx);
    if (std::is_same<T, float>::value) {
      LaunchRandomUniform_float(stream, n, num_blocks, block_size, state,
                                output->flat<float>().data());
    } else if (std::is_same<T, Eigen::half>::value) {
      LaunchRandomUniform_half(stream, n, num_blocks, block_size, state,
                               output->flat<Eigen::half>().data());
    } else if (std::is_same<T, bfloat16>::value) {
      LaunchRandomUniform_bfloat16(stream, n, num_blocks, block_size, state,
                                   output->flat<bfloat16>().data());
    } else {
      LaunchRandomUniform_double(stream, n, num_blocks, block_size, state,
                                 output->flat<double>().data());
    }
  }

 private:
  GuardedPhiloxRandom generator_;
};

class MusaRandomUniformGreaterEqualOp : public MusaOpKernel {
 public:
  explicit MusaRandomUniformGreaterEqualOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &threshold_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_tensor = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    const int64_t n = output->NumElements();
    if (n == 0) return;

    auto philox = generator_.ReserveSamples32(n);
    MusaPhiloxState state;
    std::memcpy(&state, &philox, sizeof(MusaPhiloxState));

    const int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    if (num_blocks > 4096) num_blocks = 4096;

    const float clamped_threshold = std::min(1.0f, std::max(0.0f, threshold_));
    const uint32_t threshold_bits =
        static_cast<uint32_t>(std::ceil(clamped_threshold * 8388608.0f));

    LaunchRandomUniformGreaterEqual_bool(GetStream<bool>(ctx), n, num_blocks,
                                         block_size, state, threshold_bits,
                                         output->flat<bool>().data());
  }

 private:
  GuardedPhiloxRandom generator_;
  float threshold_ = 0.0f;
};

// ==========================================
// RandomUniformInt Op
// ==========================================
template <typename T>
class MusaRandomUniformIntOp : public MusaOpKernel {
 public:
  explicit MusaRandomUniformIntOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_tensor = ctx->input(0);
    const Tensor& minval_tensor = ctx->input(1);
    const Tensor& maxval_tensor = ctx->input(2);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    int64_t n = output->NumElements();
    if (n == 0) return;

    T minval = minval_tensor.scalar<T>()();
    T maxval = maxval_tensor.scalar<T>()();
    OP_REQUIRES(ctx, minval < maxval,
                errors::InvalidArgument("Need minval < maxval"));

    auto philox = generator_.ReserveSamples32(n);
    MusaPhiloxState state;
    std::memcpy(&state, &philox, sizeof(MusaPhiloxState));

    const int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    if (num_blocks > 1024) num_blocks = 1024;

    void* stream = GetStream<T>(ctx);
    if (std::is_same<T, int32>::value) {
      LaunchRandomUniformInt_int(
          stream, n, num_blocks, block_size, state, static_cast<int>(minval),
          static_cast<int>(maxval),
          reinterpret_cast<int*>(output->flat<T>().data()));
    } else {
      LaunchRandomUniformInt_int64_t(
          stream, n, num_blocks, block_size, state,
          static_cast<int64_t>(minval), static_cast<int64_t>(maxval),
          reinterpret_cast<int64_t*>(output->flat<T>().data()));
    }
  }

 private:
  GuardedPhiloxRandom generator_;
};

// ==========================================
// RandomStandardNormal Op
// ==========================================
template <typename T>
class MusaNormalOp : public MusaOpKernel {
 public:
  explicit MusaNormalOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_tensor = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    int64_t n = output->NumElements();
    if (n == 0) return;

    auto op_name = name();
    if (op_name == "RandomStandardNormal") {
      auto philox = generator_.ReserveSamples32(n);
      MusaPhiloxState state;
      std::memcpy(&state, &philox, sizeof(MusaPhiloxState));

      const int block_size = 256;
      int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
      if (num_blocks > 1024) num_blocks = 1024;

      void* stream = GetStream<T>(ctx);
      if (std::is_same<T, float>::value) {
        LaunchRandomStandardNormal_float(stream, n, num_blocks, block_size,
                                         state, output->flat<float>().data());
      } else if (std::is_same<T, Eigen::half>::value) {
        LaunchRandomStandardNormal_half(stream, n, num_blocks, block_size,
                                        state,
                                        output->flat<Eigen::half>().data());
      } else if (std::is_same<T, bfloat16>::value) {
        LaunchRandomStandardNormal_bfloat16(stream, n, num_blocks, block_size,
                                            state,
                                            output->flat<bfloat16>().data());
      } else {
        LaunchRandomStandardNormal_double(stream, n, num_blocks, block_size,
                                          state, output->flat<double>().data());
      }
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported normal op currently: ",
                                          op_name));
    }
  }

 private:
  GuardedPhiloxRandom generator_;
};

#define REGISTER_MUSA_UNIFORM(TYPE)                           \
  REGISTER_KERNEL_BUILDER(Name("RandomUniform")               \
                              .Device("MUSA")                 \
                              .HostMemory("shape")            \
                              .TypeConstraint<int32>("T")     \
                              .TypeConstraint<TYPE>("dtype"), \
                          MusaRandomUniformOp<TYPE>)
REGISTER_MUSA_UNIFORM(float);
REGISTER_MUSA_UNIFORM(double);
REGISTER_MUSA_UNIFORM(Eigen::half);
REGISTER_MUSA_UNIFORM(bfloat16);

REGISTER_OP("MusaRandomUniformGreaterEqual")
    .Input("shape: T")
    .Output("output: bool")
    .Attr("T: {int32}")
    .Attr("threshold: float")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      c->set_output(0, out);
      return ::tensorflow::OkStatus();
    });

REGISTER_KERNEL_BUILDER(Name("MusaRandomUniformGreaterEqual")
                            .Device("MUSA")
                            .HostMemory("shape")
                            .TypeConstraint<int32>("T"),
                        MusaRandomUniformGreaterEqualOp);

#define REGISTER_MUSA_UNIFORM_INT(TYPE)                      \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")           \
                              .Device("MUSA")                \
                              .HostMemory("shape")           \
                              .HostMemory("minval")          \
                              .HostMemory("maxval")          \
                              .TypeConstraint<int32>("T")    \
                              .TypeConstraint<TYPE>("Tout"), \
                          MusaRandomUniformIntOp<TYPE>)
REGISTER_MUSA_UNIFORM_INT(int32);
REGISTER_MUSA_UNIFORM_INT(int64);

#define REGISTER_MUSA_NORMAL_KERNEL(TYPE)                     \
  REGISTER_KERNEL_BUILDER(Name("RandomStandardNormal")        \
                              .Device("MUSA")                 \
                              .HostMemory("shape")            \
                              .TypeConstraint<int32>("T")     \
                              .TypeConstraint<TYPE>("dtype"), \
                          MusaNormalOp<TYPE>)
REGISTER_MUSA_NORMAL_KERNEL(float);
REGISTER_MUSA_NORMAL_KERNEL(double);
REGISTER_MUSA_NORMAL_KERNEL(Eigen::half);
REGISTER_MUSA_NORMAL_KERNEL(bfloat16);

// ==========================================
// TruncatedNormal Op
// ==========================================
template <typename T>
class MusaTruncatedNormalOp : public MusaOpKernel {
 public:
  explicit MusaTruncatedNormalOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_tensor = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    int64_t n = output->NumElements();
    if (n == 0) return;

    auto philox = generator_.ReserveSamples32(n);
    MusaPhiloxState state;
    std::memcpy(&state, &philox, sizeof(MusaPhiloxState));

    const int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    if (num_blocks > 1024) num_blocks = 1024;

    void* stream = GetStream<T>(ctx);
    if (std::is_same<T, float>::value) {
      LaunchTruncatedNormal_float(stream, n, num_blocks, block_size, state,
                                  output->flat<float>().data());
    } else if (std::is_same<T, Eigen::half>::value) {
      LaunchTruncatedNormal_half(stream, n, num_blocks, block_size, state,
                                 output->flat<Eigen::half>().data());
    } else if (std::is_same<T, bfloat16>::value) {
      LaunchTruncatedNormal_bfloat16(stream, n, num_blocks, block_size, state,
                                     output->flat<bfloat16>().data());
    } else {
      LaunchTruncatedNormal_double(stream, n, num_blocks, block_size, state,
                                   output->flat<double>().data());
    }
  }

 private:
  GuardedPhiloxRandom generator_;
};

#define REGISTER_MUSA_TRUNCATED_NORMAL(TYPE)                  \
  REGISTER_KERNEL_BUILDER(Name("TruncatedNormal")             \
                              .Device("MUSA")                 \
                              .HostMemory("shape")            \
                              .TypeConstraint<int32>("T")     \
                              .TypeConstraint<TYPE>("dtype"), \
                          MusaTruncatedNormalOp<TYPE>)
REGISTER_MUSA_TRUNCATED_NORMAL(float);
REGISTER_MUSA_TRUNCATED_NORMAL(double);
REGISTER_MUSA_TRUNCATED_NORMAL(Eigen::half);
REGISTER_MUSA_TRUNCATED_NORMAL(bfloat16);

#undef REGISTER_MUSA_UNIFORM
#undef REGISTER_MUSA_UNIFORM_INT
#undef REGISTER_MUSA_NORMAL_KERNEL
#undef REGISTER_MUSA_TRUNCATED_NORMAL

}  // namespace musa
}  // namespace tensorflow

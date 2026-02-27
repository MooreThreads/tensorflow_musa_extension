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

#include "../utils/musa_guarded_philox_random.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "utils_op.h"

// External kernel launchers from .mu files (defined in global namespace)
// These are C++ functions with C++ linkage (name mangled)
extern void LaunchRandomUniform_float(void* stream, int64_t n, uint64_t seed,
                                      float* output);
extern void LaunchRandomUniform_double(void* stream, int64_t n, uint64_t seed,
                                       double* output);
extern void LaunchRandomUniformInt_int(void* stream, int64_t n, uint64_t seed,
                                       int minval, int maxval, int* output);
extern void LaunchRandomUniformInt_int64_t(void* stream, int64_t n,
                                           uint64_t seed, int64_t minval,
                                           int64_t maxval, int64_t* output);

namespace tensorflow {
namespace musa {

// Launchers from musa_normal_kernel.mu
template <typename T, typename DIST_TYPE>
void LaunchPhiloxNormalKernel(musaStream_t stream, T* data,
                              uint64_t num_elements,
                              const random::PhiloxRandom& philox,
                              const DIST_TYPE& dist);

// Trait for dispatching to correct uniform launcher
template <typename T>
struct UniformLauncherTrait;

template <>
struct UniformLauncherTrait<float> {
  static void Launch(void* stream, int64_t n, uint64_t seed, float* output) {
    LaunchRandomUniform_float(stream, n, seed, output);
  }
};

template <>
struct UniformLauncherTrait<double> {
  static void Launch(void* stream, int64_t n, uint64_t seed, double* output) {
    LaunchRandomUniform_double(stream, n, seed, output);
  }
};

// Helper to get MUSA stream
template <typename T>
void* GetStream(OpKernelContext* ctx) {
  if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
    // For uniform ops using legacy launchers
    return nullptr;
  }
  // For normal ops using MusaOpKernel
  auto* device = GetDeviceByCtx(ctx);
  return device ? device->GetStream() : nullptr;
}

// ============================================================================
// RandomUniform Op
// ============================================================================
template <typename T>
class MusaRandomUniformOp : public OpKernel {
 public:
  explicit MusaRandomUniformOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_tensor = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));

    int64_t num_elements = output->NumElements();
    if (num_elements == 0) return;

    // Initialize Philox generator
    GuardedPhiloxRandom generator;
    generator.Init(seed_, seed2_);

    // Reserve samples and get Philox instance
    uint64_t samples_needed = num_elements;
    auto philox = generator.ReserveSamples32(samples_needed);
    uint32_t samples[4];
    philox.Next(samples);
    uint64_t seed = (static_cast<uint64_t>(samples[0]) << 32) | samples[1];

    // Launch kernel
    void* stream = GetStream<T>(ctx);
    T* data = output->flat<T>().data();
    UniformLauncherTrait<T>::Launch(stream, num_elements, seed, data);
  }

 private:
  tensorflow::int64 seed_;
  tensorflow::int64 seed2_;
};

// ============================================================================
// RandomUniformInt Op
// ============================================================================
template <typename T>
class MusaRandomUniformIntOp : public OpKernel {
 public:
  explicit MusaRandomUniformIntOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_tensor = ctx->input(0);
    const Tensor& minval_tensor = ctx->input(1);
    const Tensor& maxval_tensor = ctx->input(2);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));

    int64_t num_elements = output->NumElements();
    if (num_elements == 0) return;

    T minval = minval_tensor.scalar<T>()();
    T maxval = maxval_tensor.scalar<T>()();
    OP_REQUIRES(ctx, minval < maxval,
                errors::InvalidArgument("Need minval < maxval, got ", minval,
                                        " >= ", maxval));

    // Initialize Philox generator
    GuardedPhiloxRandom generator;
    generator.Init(seed_, seed2_);

    // Reserve samples and get Philox instance
    uint64_t samples_needed = num_elements;
    auto philox = generator.ReserveSamples32(samples_needed);
    uint32_t samples[4];
    philox.Next(samples);
    uint64_t seed = (static_cast<uint64_t>(samples[0]) << 32) | samples[1];

    // Launch kernel
    void* stream = GetStream<T>(ctx);
    T* data = output->flat<T>().data();

    if (std::is_same<T, int32>::value) {
      LaunchRandomUniformInt_int(
          stream, num_elements, seed, static_cast<int>(minval),
          static_cast<int>(maxval), reinterpret_cast<int*>(data));
    } else {
      LaunchRandomUniformInt_int64_t(
          stream, num_elements, seed, static_cast<int64_t>(minval),
          static_cast<int64_t>(maxval), reinterpret_cast<int64_t*>(data));
    }
  }

 private:
  tensorflow::int64 seed_;
  tensorflow::int64 seed2_;
};

// ============================================================================
// RandomStandardNormal / TruncatedNormal Ops
// ============================================================================
template <typename T>
class MusaNormalOp : public MusaOpKernel {
 public:
  explicit MusaNormalOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2_));
  }

  void Compute(OpKernelContext* ctx) override {
    using PhiloxRandom = random::PhiloxRandom;
    using NormalDist = random::NormalDistribution<PhiloxRandom>;
    using TruncatedDist = random::TruncatedNormalDistribution<PhiloxRandom>;

    const Tensor& shape_tensor = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    int64_t total_elements = shape.num_elements();
    if (total_elements == 0) return;

    // Initialize Philox with seed management
    GuardedPhiloxRandom generator;
    generator.Init(seed_, seed2_);

    // Get MUSA stream
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    // Dispatch based on op type
    auto op_name = name();
    if (op_name == "RandomStandardNormal") {
      // NormalDistribution generates 4 values per call
      uint64_t samples_needed = ((output->NumElements() + 3) / 4) * 4;
      auto philox = generator.ReserveSamples32(samples_needed);
      NormalDist dist;
      LaunchPhiloxNormalKernel<T, NormalDist>(stream, output->flat<T>().data(),
                                              output->NumElements(), philox,
                                              dist);
    } else if (op_name == "TruncatedNormal") {
      // TruncatedNormal uses rejection sampling, need ~4x oversampling
      uint64_t samples_needed = total_elements * 4;
      auto philox = generator.ReserveSamples32(samples_needed);
      TruncatedDist dist;
      LaunchPhiloxNormalKernel<T, TruncatedDist>(
          stream, output->flat<T>().data(), output->NumElements(), philox,
          dist);
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported op name: ", op_name));
    }
  }

 private:
  tensorflow::int64 seed_;
  tensorflow::int64 seed2_;
};

// ============================================================================
// Kernel Registration
// ============================================================================

// RandomUniform - float and double
#define REGISTER_MUSA_UNIFORM(TYPE)                           \
  REGISTER_KERNEL_BUILDER(Name("RandomUniform")               \
                              .Device("MUSA")                 \
                              .HostMemory("shape")            \
                              .TypeConstraint<int32>("T")     \
                              .TypeConstraint<TYPE>("dtype"), \
                          MusaRandomUniformOp<TYPE>)

REGISTER_MUSA_UNIFORM(float);
REGISTER_MUSA_UNIFORM(double);

// RandomUniformInt - int32 and int64
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

// RandomStandardNormal and TruncatedNormal
#define REGISTER_MUSA_NORMAL_KERNEL(TYPE)                     \
  REGISTER_KERNEL_BUILDER(Name("RandomStandardNormal")        \
                              .Device("MUSA")                 \
                              .HostMemory("shape")            \
                              .TypeConstraint<int32>("T")     \
                              .TypeConstraint<TYPE>("dtype"), \
                          MusaNormalOp<TYPE>);                \
  REGISTER_KERNEL_BUILDER(Name("TruncatedNormal")             \
                              .Device("MUSA")                 \
                              .HostMemory("shape")            \
                              .TypeConstraint<int32>("T")     \
                              .TypeConstraint<TYPE>("dtype"), \
                          MusaNormalOp<TYPE>)

REGISTER_MUSA_NORMAL_KERNEL(float);
REGISTER_MUSA_NORMAL_KERNEL(double);
REGISTER_MUSA_NORMAL_KERNEL(Eigen::half);
REGISTER_MUSA_NORMAL_KERNEL(Eigen::bfloat16);

#undef REGISTER_MUSA_UNIFORM
#undef REGISTER_MUSA_UNIFORM_INT
#undef REGISTER_MUSA_NORMAL_KERNEL

}  // namespace musa
}  // namespace tensorflow

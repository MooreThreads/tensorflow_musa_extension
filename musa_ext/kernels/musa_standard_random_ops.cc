
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "mu/device/musa_executor.h" 
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>

namespace tensorflow {
namespace musa {

using stream_executor::musa::FromMusaStatus;



namespace {

// ==========================================
// 辅助函数区域
// ==========================================

// 将 uint32 转换为 [0, 1) 的 float/double
template <typename T>
T Uint32ToFloatOfficial(uint32 x) {
  const uint32 man = x & 0x7fffffu;
  const uint32 exp = 127u << 23;
  const uint32 val = exp | man;
  float result;
  std::memcpy(&result, &val, sizeof(val));
  return static_cast<T>(result - 1.0f);
}

// Box-Muller 变换：将两个均匀分布转换为两个正态分布
template <typename T>
void BoxMullerTransform(T u1, T u2, T* z1, T* z2) {
    const T kTwoPi = static_cast<T>(2.0 * M_PI);
    T log_u1 = std::log(u1 + static_cast<T>(1e-20)); 
    T multiplier = std::sqrt(static_cast<T>(-2.0) * log_u1);
    
    *z1 = multiplier * std::cos(kTwoPi * u2);
    *z2 = multiplier * std::sin(kTwoPi * u2);
}

} // namespace

// ==========================================
// 1. RandomUniform Op (均匀分布)
// ==========================================
template <typename T>
class MusaRandomUniformOp : public MusaOpKernel {
 public:
  explicit MusaRandomUniformOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, guarded_philox_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensorflow::tensor::MakeShape(shape_t, &shape));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    
    int64 num_elements = output->NumElements();
    if (num_elements == 0) return;

    Tensor tmp_host;
    AllocatorAttributes host_attr; host_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(output->dtype(), shape, &tmp_host, host_attr));
    T* cpu_ptr = tmp_host.flat<T>().data();

    auto local_gen = guarded_philox_.ReserveSamples32(num_elements);

    for (int64 i = 0; i < num_elements; i += 4) {
        auto samples = local_gen();
        for (int j = 0; j < 4 && (i + j) < num_elements; ++j) {
            cpu_ptr[i + j] = Uint32ToFloatOfficial<T>(samples[j]);
        }
    }

    mStatus s = tensorflow::musa::MusaMemcpyH2D(output->data(), tmp_host.data(), num_elements * sizeof(T));
    OP_REQUIRES_OK(ctx, FromMusaStatus(s));
    musaDeviceSynchronize();
  }

 private:

  tensorflow::GuardedPhiloxRandom guarded_philox_;
};

// ==========================================
// 2. RandomStandardNormal Op (标准正态分布)
// ==========================================
template <typename T>
class MusaRandomStandardNormalOp : public MusaOpKernel {
 public:
  explicit MusaRandomStandardNormalOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, guarded_philox_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensorflow::tensor::MakeShape(shape_t, &shape));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));

    int64 num_elements = output->NumElements();
    if (num_elements == 0) return;

    Tensor tmp_host;
    AllocatorAttributes host_attr; host_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(output->dtype(), shape, &tmp_host, host_attr));
    T* cpu_ptr = tmp_host.flat<T>().data();

   
    auto local_gen = guarded_philox_.ReserveSamples32(num_elements);

    for (int64 i = 0; i < num_elements; i += 4) {
        auto samples = local_gen();
        T u[4];
        for(int j=0; j<4; ++j) u[j] = Uint32ToFloatOfficial<T>(samples[j]);

        // Box-Muller 变换
        if (i < num_elements) {
            T z0, z1;
            BoxMullerTransform(u[0], u[1], &z0, &z1);
            cpu_ptr[i] = z0;
            if (i + 1 < num_elements) cpu_ptr[i+1] = z1;
        }
        if (i + 2 < num_elements) {
            T z2, z3;
            BoxMullerTransform(u[2], u[3], &z2, &z3);
            cpu_ptr[i+2] = z2;
            if (i + 3 < num_elements) cpu_ptr[i+3] = z3;
        }
    }

    mStatus s = tensorflow::musa::MusaMemcpyH2D(output->data(), tmp_host.data(), num_elements * sizeof(T));
    OP_REQUIRES_OK(ctx, FromMusaStatus(s));
    musaDeviceSynchronize();
  }

 private:

  tensorflow::GuardedPhiloxRandom guarded_philox_;
};


template <typename T>
class MusaTruncatedNormalOp : public MusaOpKernel {
 public:
  explicit MusaTruncatedNormalOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, guarded_philox_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensorflow::tensor::MakeShape(shape_t, &shape));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));

    int64 num_elements = output->NumElements();
    if (num_elements == 0) return;

    Tensor tmp_host;
    AllocatorAttributes host_attr; host_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(output->dtype(), shape, &tmp_host, host_attr));
    T* cpu_ptr = tmp_host.flat<T>().data();

    
    auto local_gen = guarded_philox_.ReserveSamples32(num_elements * 2);
    
    
    tensorflow::random::PhiloxRandom::ResultType samples;
    int sample_idx = 4; 

    for (int64 i = 0; i < num_elements; ++i) {
        T candidate;
        while (true) {
            if (sample_idx >= 4) {
                samples = local_gen(); // 获取新的一组随机数
                sample_idx = 0;
            }
            
            uint32 s1 = samples[sample_idx++];
            if (sample_idx >= 4) { samples = local_gen(); sample_idx = 0; }
            uint32 s2 = samples[sample_idx++];

            T u1 = Uint32ToFloatOfficial<T>(s1);
            T u2 = Uint32ToFloatOfficial<T>(s2);
            
            T z1, z2;
            BoxMullerTransform(u1, u2, &z1, &z2);
            
            // 检查截断条件: abs(x) <= 2.0
            if (std::abs(z1) <= static_cast<T>(2.0)) {
                candidate = z1;
                break;
            }
            if (std::abs(z2) <= static_cast<T>(2.0)) {
                candidate = z2;
                break;
            }
        }
        cpu_ptr[i] = candidate;
    }

    mStatus s = tensorflow::musa::MusaMemcpyH2D(output->data(), tmp_host.data(), num_elements * sizeof(T));
    OP_REQUIRES_OK(ctx, FromMusaStatus(s));
    musaDeviceSynchronize();
  }

 private:

  tensorflow::GuardedPhiloxRandom guarded_philox_;
};


template <typename IntType>
class MusaRandomUniformIntOp : public MusaOpKernel {
 public:
  explicit MusaRandomUniformIntOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, guarded_philox_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);
    const Tensor& minval = ctx->input(1);
    const Tensor& maxval = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument("minval must be 0-D"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument("maxval must be 0-D"));

    Tensor* output = nullptr;
    TensorShape tensor_shape;
    OP_REQUIRES_OK(ctx, tensorflow::tensor::MakeShape(shape, &tensor_shape));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tensor_shape, &output));

    if (output->NumElements() == 0) return;

    IntType lo = minval.scalar<IntType>()();
    IntType hi = maxval.scalar<IntType>()();
    OP_REQUIRES(ctx, lo < hi,
                errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

    
    typedef typename std::make_unsigned<IntType>::type UIntType;
    UIntType range = static_cast<UIntType>(hi - lo);
    
    int64 num_elements = output->NumElements();

    Tensor tmp_host;
    AllocatorAttributes host_attr; host_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(output->dtype(), tensor_shape, &tmp_host, host_attr));
    IntType* cpu_ptr = tmp_host.flat<IntType>().data();

    bool is_64bit = (sizeof(IntType) == 8);
    auto local_gen = guarded_philox_.ReserveSamples32(num_elements * (is_64bit ? 2 : 1));

    int sample_idx = 0;
    auto samples = local_gen(); 

    for (int64 i = 0; i < num_elements; ++i) {
        
        UIntType rand_val;
        
        if (is_64bit) {
            uint32 low = samples[sample_idx++];
            if (sample_idx >= 4) { samples = local_gen(); sample_idx = 0; }
            uint32 high = samples[sample_idx++];
            if (sample_idx >= 4) { samples = local_gen(); sample_idx = 0; }
            
            // 组合成 64 位无符号整数
            rand_val = (static_cast<uint64>(high) << 32) | low;
        } else {
            uint32 val = samples[sample_idx++];
            if (sample_idx >= 4) { samples = local_gen(); sample_idx = 0; }
            // 转换为对应的无符号类型
            rand_val = static_cast<UIntType>(val);
        }

      
        cpu_ptr[i] = lo + static_cast<IntType>(rand_val % range);
    }

    mStatus s = tensorflow::musa::MusaMemcpyH2D(output->data(), tmp_host.data(), num_elements * sizeof(IntType));
    OP_REQUIRES_OK(ctx, FromMusaStatus(s));
    musaDeviceSynchronize();
  }

 private:
  tensorflow::GuardedPhiloxRandom guarded_philox_;
};


// ==========================================
// 注册 Kernels
// ==========================================

// 1. Register RandomUniform
#define REGISTER_MUSA_UNIFORM(TYPE)                          \
  REGISTER_KERNEL_BUILDER(Name("RandomUniform")              \
                              .Device("MUSA")                \
                              .HostMemory("shape")           \
                              .TypeConstraint<int32>("T")    \
                              .TypeConstraint<TYPE>("dtype"), \
                          MusaRandomUniformOp<TYPE>)

REGISTER_MUSA_UNIFORM(float);
REGISTER_MUSA_UNIFORM(double);

// 2. Register RandomStandardNormal
#define REGISTER_MUSA_STANDARD_NORMAL(TYPE)                          \
  REGISTER_KERNEL_BUILDER(Name("RandomStandardNormal")               \
                              .Device("MUSA")                        \
                              .HostMemory("shape")                   \
                              .TypeConstraint<int32>("T")            \
                              .TypeConstraint<TYPE>("dtype"),        \
                          MusaRandomStandardNormalOp<TYPE>)

REGISTER_MUSA_STANDARD_NORMAL(float);
REGISTER_MUSA_STANDARD_NORMAL(double);

// 3. Register TruncatedNormal
#define REGISTER_MUSA_TRUNCATED_NORMAL(TYPE)                         \
  REGISTER_KERNEL_BUILDER(Name("TruncatedNormal")                    \
                              .Device("MUSA")                        \
                              .HostMemory("shape")                   \
                              .TypeConstraint<int32>("T")            \
                              .TypeConstraint<TYPE>("dtype"),        \
                          MusaTruncatedNormalOp<TYPE>)

REGISTER_MUSA_TRUNCATED_NORMAL(float);
REGISTER_MUSA_TRUNCATED_NORMAL(double);

// 4. Register RandomUniformInt
#define REGISTER_MUSA_UNIFORM_INT(IntType)                           \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")                   \
                              .Device("MUSA")                        \
                              .HostMemory("shape")                   \
                              .HostMemory("minval")                  \
                              .HostMemory("maxval")                  \
                              .TypeConstraint<int32>("T")            \
                              .TypeConstraint<IntType>("Tout"),      \
                          MusaRandomUniformIntOp<IntType>)

REGISTER_MUSA_UNIFORM_INT(int32);
REGISTER_MUSA_UNIFORM_INT(int64);

} // namespace musa
} // namespace tensorflow


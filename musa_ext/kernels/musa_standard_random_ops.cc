#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/stream_executor/stream.h"
#include <iostream>

extern void LaunchRandomUniform_float(void* stream, int64_t n, uint64_t seed, float* output);
extern void LaunchRandomUniform_double(void* stream, int64_t n, uint64_t seed, double* output);

extern void LaunchRandomStandardNormal_float(void* stream, int64_t n, uint64_t seed, float* output);
extern void LaunchRandomStandardNormal_double(void* stream, int64_t n, uint64_t seed, double* output);

extern void LaunchTruncatedNormal_float(void* stream, int64_t n, uint64_t seed, float* output);
extern void LaunchTruncatedNormal_double(void* stream, int64_t n, uint64_t seed, double* output);

extern void LaunchRandomUniformInt_int(void* stream, int64_t n, uint64_t seed, int minval, int maxval, int* output);
extern void LaunchRandomUniformInt_int64_t(void* stream, int64_t n, uint64_t seed, int64_t minval, int64_t maxval, int64_t* output);

extern "C" {
    typedef int musaError_t;
    musaError_t musaStreamSynchronize(void* stream);
    musaError_t musaGetLastError();
    const char* musaGetErrorString(musaError_t error);
}

namespace tensorflow {
namespace musa {

template <typename T> struct LauncherTrait;

template <> struct LauncherTrait<float> {
    static void Uniform(void* s, int64 n, uint64 seed, float* o) { ::LaunchRandomUniform_float(s, n, seed, o); }
    static void Normal(void* s, int64 n, uint64 seed, float* o) { ::LaunchRandomStandardNormal_float(s, n, seed, o); }
    static void Truncated(void* s, int64 n, uint64 seed, float* o) { ::LaunchTruncatedNormal_float(s, n, seed, o); }
};
template <> struct LauncherTrait<double> {
    static void Uniform(void* s, int64 n, uint64 seed, double* o) { ::LaunchRandomUniform_double(s, n, seed, o); }
    static void Normal(void* s, int64 n, uint64 seed, double* o) { ::LaunchRandomStandardNormal_double(s, n, seed, o); }
    static void Truncated(void* s, int64 n, uint64 seed, double* o) { ::LaunchTruncatedNormal_double(s, n, seed, o); }
};

uint64 GetPhiloxSeed(GuardedPhiloxRandom& guarded_philox, int64 num_elements) {
    auto local_gen = guarded_philox.ReserveSamples32(num_elements);
    auto samples = local_gen(); 
    return (static_cast<uint64>(samples[0]) << 32) | samples[1];
}

void* GetMusaStreamHandle(OpKernelContext* ctx) {
    return nullptr; 
}

template <typename Func, typename... Args>
void LaunchAndCheck(OpKernelContext* ctx, const char* name, Func func, void* stream, Args... args) {
    func(stream, args...);
    
    musaError_t err = musaGetLastError();
    if (err != 0) {
        std::cerr << "KERNEL LAUNCH ERROR (Pre-Sync): " << name << " : " << musaGetErrorString(err) << std::endl;
        OP_REQUIRES(ctx, err == 0, errors::Internal("Kernel Launch Failed (Pre-Sync): ", name, ": ", musaGetErrorString(err)));
    }

    musaError_t sync_err = musaStreamSynchronize(stream);
    
    if (sync_err != 0) {
        std::cerr << "KERNEL EXECUTION FAILED: " << name << " Error: " << musaGetErrorString(sync_err) << std::endl;
        OP_REQUIRES(ctx, sync_err == 0, errors::Internal("Kernel Execution Failed: ", name, " Msg: ", musaGetErrorString(sync_err)));
    }
}

template <typename T>
class MusaRandomUniformOp : public OpKernel {
 public:
  explicit MusaRandomUniformOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, guarded_philox_.Init(ctx));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_t, &shape));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    int64 num_elements = output->NumElements();
    if (num_elements == 0) return;

    void* stream = GetMusaStreamHandle(ctx);
    uint64 seed = GetPhiloxSeed(guarded_philox_, num_elements);
    
    LaunchAndCheck(ctx, "RandomUniform", LauncherTrait<T>::Uniform, stream, num_elements, seed, output->flat<T>().data());
  }
 private:
  GuardedPhiloxRandom guarded_philox_;
};

template <typename T>
class MusaRandomStandardNormalOp : public OpKernel {
 public:
  explicit MusaRandomStandardNormalOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, guarded_philox_.Init(ctx));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_t, &shape));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    int64 num_elements = output->NumElements();
    if (num_elements == 0) return;

    void* stream = GetMusaStreamHandle(ctx);
    uint64 seed = GetPhiloxSeed(guarded_philox_, num_elements);

    LaunchAndCheck(ctx, "RandomStandardNormal", LauncherTrait<T>::Normal, stream, num_elements, seed, output->flat<T>().data());
  }
 private:
  GuardedPhiloxRandom guarded_philox_;
};

template <typename T>
class MusaTruncatedNormalOp : public OpKernel {
 public:
  explicit MusaTruncatedNormalOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, guarded_philox_.Init(ctx));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_t, &shape));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    int64 num_elements = output->NumElements();
    if (num_elements == 0) return;

    void* stream = GetMusaStreamHandle(ctx);
    uint64 seed = GetPhiloxSeed(guarded_philox_, num_elements * 2);

    LaunchAndCheck(ctx, "TruncatedNormal", LauncherTrait<T>::Truncated, stream, num_elements, seed, output->flat<T>().data());
  }
 private:
  GuardedPhiloxRandom guarded_philox_;
};

template <typename T>
class MusaRandomUniformIntOp : public OpKernel {
 public:
  explicit MusaRandomUniformIntOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, guarded_philox_.Init(ctx));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);
    const Tensor& minval = ctx->input(1);
    const Tensor& maxval = ctx->input(2);

    Tensor* output = nullptr;
    TensorShape tensor_shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape, &tensor_shape));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tensor_shape, &output));
    if (output->NumElements() == 0) return;

    T lo = minval.scalar<T>()();
    T hi = maxval.scalar<T>()();
    OP_REQUIRES(ctx, lo < hi, errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

    int64 num_elements = output->NumElements();
    void* stream = GetMusaStreamHandle(ctx);
    uint64 seed = GetPhiloxSeed(guarded_philox_, num_elements);

    if (std::is_same<T, int>::value) {
        LaunchAndCheck(ctx, "RandomUniformInt_int", ::LaunchRandomUniformInt_int, stream, num_elements, seed, (int)lo, (int)hi, (int*)output->flat<T>().data());
    } else {
        LaunchAndCheck(ctx, "RandomUniformInt_int64", ::LaunchRandomUniformInt_int64_t, stream, num_elements, seed, (int64_t)lo, (int64_t)hi, (int64_t*)output->flat<T>().data());
    }
  }
 private:
  GuardedPhiloxRandom guarded_philox_;
};

#define REGISTER_MUSA_UNIFORM(TYPE) \
  REGISTER_KERNEL_BUILDER(Name("RandomUniform").Device("MUSA").HostMemory("shape") \
                              .TypeConstraint<int32>("T").TypeConstraint<TYPE>("dtype"), \
                          MusaRandomUniformOp<TYPE>)
REGISTER_MUSA_UNIFORM(float);
REGISTER_MUSA_UNIFORM(double);

#define REGISTER_MUSA_NORMAL(TYPE) \
  REGISTER_KERNEL_BUILDER(Name("RandomStandardNormal").Device("MUSA").HostMemory("shape") \
                              .TypeConstraint<int32>("T").TypeConstraint<TYPE>("dtype"), \
                          MusaRandomStandardNormalOp<TYPE>)
REGISTER_MUSA_NORMAL(float);
REGISTER_MUSA_NORMAL(double);

#define REGISTER_MUSA_TRUNCATED(TYPE) \
  REGISTER_KERNEL_BUILDER(Name("TruncatedNormal").Device("MUSA").HostMemory("shape") \
                              .TypeConstraint<int32>("T").TypeConstraint<TYPE>("dtype"), \
                          MusaTruncatedNormalOp<TYPE>)
REGISTER_MUSA_TRUNCATED(float);
REGISTER_MUSA_TRUNCATED(double);

#define REGISTER_MUSA_UNIFORM_INT(TYPE) \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt").Device("MUSA").HostMemory("shape") \
                              .HostMemory("minval").HostMemory("maxval") \
                              .TypeConstraint<int32>("T").TypeConstraint<TYPE>("Tout"), \
                          MusaRandomUniformIntOp<TYPE>)
REGISTER_MUSA_UNIFORM_INT(int32);
REGISTER_MUSA_UNIFORM_INT(int64);

} // namespace musa
} // namespace tensorflow

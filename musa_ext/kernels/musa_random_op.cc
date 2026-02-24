#include "mu/device/musa_executor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
namespace musa {

using random::PhiloxRandom;
using stream_executor::musa::FromMusaStatus;

namespace {

template <typename T>
PHILOX_DEVICE_INLINE T Uint32ToFloatOfficial(uint32 x) {
  const uint32 man = x & 0x7fffffu;
  const uint32 exp = 127u << 23;
  const uint32 val = exp | man;
  float result;
  std::memcpy(&result, &val, sizeof(val));
  return static_cast<T>(result - 1.0f);
}

Status InternalGenerateKey(const Tensor& seed, PhiloxRandom::Key* out_key,
                           PhiloxRandom::ResultType* out_counter) {
  uint64 seed0;
  uint64 seed1;

  if (seed.dtype() == DT_INT32) {
    const auto seed_vals = seed.flat<int32>();
    seed0 = static_cast<uint64>(seed_vals(0));
    seed1 = static_cast<uint64>(seed_vals(1));
  } else if (seed.dtype() == DT_INT64) {
    const auto seed_vals = seed.flat<int64>();
    seed0 = static_cast<uint64>(seed_vals(0));
    seed1 = static_cast<uint64>(seed_vals(1));
  } else {
    return errors::InvalidArgument("Invalid seed type");
  }

  (*out_key)[0] = 0x3ec8f720;
  (*out_key)[1] = 0x02461e29;
  (*out_counter)[0] = static_cast<uint32>(seed0);
  (*out_counter)[1] = static_cast<uint32>(seed0 >> 32);
  (*out_counter)[2] = static_cast<uint32>(seed1);
  (*out_counter)[3] = static_cast<uint32>(seed1 >> 32);

  const auto mix = random::PhiloxRandom(*out_counter, *out_key)();

  (*out_key)[0] = mix[0];
  (*out_key)[1] = mix[1];
  (*out_counter)[0] = 0;
  (*out_counter)[1] = 0;
  (*out_counter)[2] = mix[2];
  (*out_counter)[3] = mix[3];

  return Status::OK();
}

template <typename T>
class MusaRandomOp : public MusaOpKernel {
 public:
  explicit MusaRandomOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    const Tensor& seed_t = ctx->input(1);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensorflow::tensor::MakeShape(shape_t, &shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    int64 num_elements = output->NumElements();
    if (num_elements == 0) return;

    Tensor tmp_host;
    AllocatorAttributes host_attr;
    host_attr.set_on_host(true);
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(output->dtype(), shape, &tmp_host, host_attr));
    T* cpu_ptr = tmp_host.flat<T>().data();

    PhiloxRandom::Key key;
    PhiloxRandom::ResultType counter;

    if (name() == "StatelessRandomUniform") {
      OP_REQUIRES_OK(ctx, InternalGenerateKey(seed_t, &key, &counter));
    } else {
      const uint32* k_ptr = (const uint32*)ctx->input(1).tensor_data().data();
      const uint32* c_ptr = (const uint32*)ctx->input(2).tensor_data().data();
      std::memcpy(&key, k_ptr, 8);
      std::memcpy(&counter, c_ptr, 16);
    }

    PhiloxRandom gen(counter, key);
    for (int64 i = 0; i < num_elements; i += 4) {
      auto samples = gen();
      for (int j = 0; j < 4 && (i + j) < num_elements; ++j) {
        cpu_ptr[i + j] = Uint32ToFloatOfficial<T>(samples[j]);
      }
    }

    // Use async memcpy with the kernel's stream for better concurrency
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    mStatus s = tensorflow::musa::MusaMemcpyAsyncH2D(
        output->data(), tmp_host.data(), num_elements * sizeof(T), stream);
    OP_REQUIRES_OK(ctx, FromMusaStatus(s));
    // Synchronize only the current stream, not all streams on the device
    musaStreamSynchronize(stream);
  }
};

}  // namespace

#define REGISTER_MUSA_RANDOM(TYPE)                            \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomUniform")      \
                              .Device("MUSA")                 \
                              .HostMemory("shape")            \
                              .HostMemory("seed")             \
                              .TypeConstraint<TYPE>("dtype"), \
                          MusaRandomOp<TYPE>);                \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomUniformV2")    \
                              .Device("MUSA")                 \
                              .HostMemory("shape")            \
                              .HostMemory("key")              \
                              .HostMemory("counter")          \
                              .HostMemory("alg")              \
                              .TypeConstraint<TYPE>("dtype"), \
                          MusaRandomOp<TYPE>);

REGISTER_MUSA_RANDOM(float);
REGISTER_MUSA_RANDOM(double);

}  // namespace musa
}  // namespace tensorflow
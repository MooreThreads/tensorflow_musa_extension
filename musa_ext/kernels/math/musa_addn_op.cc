#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "../utils_op.h"
#include "utils/logging.h"

extern "C" {
  void LaunchAddNKernelFloat(const float** inputs, float* output, int num_inputs,
                             int size, musaStream_t stream);
  void LaunchAddNKernelDouble(const double** inputs, double* output,
                              int num_inputs, int size, musaStream_t stream);
  void LaunchAddNKernelHalf(const void** inputs, void* output, int num_inputs,
                            int size, musaStream_t stream);
  void LaunchAddNKernelBFloat16(const void** inputs, void* output,
                                int num_inputs, int size, musaStream_t stream);
  void LaunchAddNKernelInt32(const int** inputs, int* output, int num_inputs,
                             int size, musaStream_t stream);
  void LaunchAddNKernelInt64(const int64_t** inputs, int64_t* output,
                             int num_inputs, int size, musaStream_t stream);
}

namespace tensorflow {
namespace musa {

template <typename T>
void AddNCompute(OpKernelContext* ctx, mFormat format,
                 void (*launcher)(const T**, T*, int, int, musaStream_t)) {
  MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, "AddN");

  const int num_inputs = ctx->num_inputs();
  OP_REQUIRES(ctx, num_inputs >= 1,
              errors::InvalidArgument("AddN requires at least one input."));

  // PERFORMANCE OPTIMIZATION: Zero-Copy for single input case
  // When AddN has only one input: output = input[0]
  // Use set_output instead of allocating and copying
  if (num_inputs == 1) {
    const Tensor& input = ctx->input(0);
    ctx->set_output(0, input);
    return;
  }

  // Validate shapes
  const Tensor& input0 = ctx->input(0);
  TensorShape output_shape = input0.shape();
  const size_t num_elements = input0.NumElements();
  for (int i = 1; i < num_inputs; ++i) {
    OP_REQUIRES(ctx, ctx->input(i).shape() == output_shape,
                errors::InvalidArgument("All inputs must have same shape"));
  }

  // Allocate output
  Tensor* output = nullptr;
  MUSA_KERNEL_TRACE_START("Mem Alloc");
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
  MUSA_KERNEL_TRACE_END("Mem Alloc");
  if (num_elements == 0) return;

  // Handle two inputs - use muDNN Binary
  if (num_inputs == 2) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor t0 = CreateMTensor(input0, format);
    mTensor t1 = CreateMTensor(ctx->input(1), format);
    mTensor t_out = CreateMTensor(*output, format);
    ::musa::dnn::Binary op;
    op.SetMode(::musa::dnn::Binary::Mode::ADD);
    MUSA_KERNEL_TRACE_START("Kernel");
    auto status = op.Run(handle, t_out, t0, t1);
    MUSA_KERNEL_TRACE_END("Kernel");
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA AddN two inputs failed."));
    return;
  }

  // 3+ inputs - custom kernel
  musaStream_t stream = reinterpret_cast<musaStream_t>(
      GetHandleByCtx(ctx).GetStream());

  std::vector<const void*> input_ptrs(num_inputs);
  for (int i = 0; i < num_inputs; ++i)
    input_ptrs[i] = ctx->input(i).tensor_data().data();

  const void** d_inputs = nullptr;
  MUSA_KERNEL_TRACE_START("Mem Alloc");
  musaMalloc(reinterpret_cast<void**>(&d_inputs),
             num_inputs * sizeof(const void*));
  MUSA_KERNEL_TRACE_END("Mem Alloc");
  MUSA_KERNEL_TRACE_START("Mem Cpy");
  musaMemcpy(const_cast<void**>(d_inputs), input_ptrs.data(),
             num_inputs * sizeof(const void*), musaMemcpyHostToDevice);
  MUSA_KERNEL_TRACE_END("Mem Cpy");

  void* output_ptr = const_cast<char*>(output->tensor_data().data());
  MUSA_KERNEL_TRACE_START("Kernel");
  launcher(reinterpret_cast<const T**>(d_inputs),
           reinterpret_cast<T*>(output_ptr),
           num_inputs, static_cast<int>(num_elements), stream);
  MUSA_KERNEL_TRACE_END("Kernel");

  musaFree(const_cast<void**>(d_inputs));
}

template <typename T>
class MusaAddNOp : public MusaOpKernel {
 public:
  explicit MusaAddNOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    AddNCompute<T>(ctx, format_, GetLauncher());
  }

 private:
  static void (*GetLauncher())(const T**, T*, int, int, musaStream_t);
};

#define DEFINE_ADDN_LAUNCHER_GETTER(T, launcher, input_cast, output_cast)       \
  template <>                                                                   \
  void (*MusaAddNOp<T>::GetLauncher())(const T**, T*, int, int, musaStream_t) { \
    return [](const T** inputs, T* output, int num_inputs, int size,            \
              musaStream_t stream) {                                            \
      launcher(input_cast(inputs), output_cast(output),                         \
               num_inputs, size, stream);                                       \
    };                                                                          \
  }

DEFINE_ADDN_LAUNCHER_GETTER(float, LaunchAddNKernelFloat,
                            reinterpret_cast<const float**>,
                            reinterpret_cast<float*>)
DEFINE_ADDN_LAUNCHER_GETTER(double, LaunchAddNKernelDouble,
                            reinterpret_cast<const double**>,
                            reinterpret_cast<double*>)
DEFINE_ADDN_LAUNCHER_GETTER(Eigen::half, LaunchAddNKernelHalf,
                            reinterpret_cast<const void**>,
                            reinterpret_cast<void*>)
DEFINE_ADDN_LAUNCHER_GETTER(bfloat16, LaunchAddNKernelBFloat16,
                            reinterpret_cast<const void**>,
                            reinterpret_cast<void*>)
DEFINE_ADDN_LAUNCHER_GETTER(int32, LaunchAddNKernelInt32,
                            reinterpret_cast<const int**>,
                            reinterpret_cast<int*>)
DEFINE_ADDN_LAUNCHER_GETTER(int64, LaunchAddNKernelInt64,
                            reinterpret_cast<const int64_t**>,
                            reinterpret_cast<int64_t*>)

#undef DEFINE_ADDN_LAUNCHER_GETTER

#define REGISTER_MUSA_ADDN(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("AddN").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaAddNOp<TYPE>);

REGISTER_MUSA_ADDN(float);
REGISTER_MUSA_ADDN(double);
REGISTER_MUSA_ADDN(Eigen::half);
REGISTER_MUSA_ADDN(bfloat16);
REGISTER_MUSA_ADDN(int32);
REGISTER_MUSA_ADDN(int64);

}  // namespace musa
}  // namespace tensorflow

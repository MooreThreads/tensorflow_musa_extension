#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

// ============================================================================
// MUSA AddN custom kernel launcher declarations from musa_addn_kernel.mu
// ============================================================================

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
  void LaunchAddNKernelInt64(const long long** inputs, long long* output,
                             int num_inputs, int size, musaStream_t stream);
}

namespace tensorflow {
namespace musa {

// Common implementation for AddN Compute
template <typename T>
void AddNCompute(OpKernelContext* ctx, mFormat format,
                 void (*launcher)(const T**, T*, int, int, musaStream_t)) {
  const int num_inputs = ctx->num_inputs();
  OP_REQUIRES(ctx, num_inputs >= 1,
              errors::InvalidArgument("AddN requires at least one input."));

  // Handle single input - direct copy
  if (num_inputs == 1) {
    const Tensor& input = ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    if (input.NumElements() == 0) return;
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());
    mStatus copy_status = MusaMemcpyAsyncD2D(
        const_cast<char*>(output->tensor_data().data()),
        input.tensor_data().data(), input.TotalBytes(), stream);
    OP_REQUIRES(ctx, copy_status == mStatus::SUCCESS,
                errors::Internal("MUSA AddN single input copy failed."));
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
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
  if (num_elements == 0) return;

  // Handle two inputs - use muDNN Binary
  if (num_inputs == 2) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor t0 = CreateMTensor(input0, format);
    mTensor t1 = CreateMTensor(ctx->input(1), format);
    mTensor t_out = CreateMTensor(*output, format);
    ::musa::dnn::Binary op;
    op.SetMode(::musa::dnn::Binary::Mode::ADD);
    auto status = op.Run(handle, t_out, t0, t1);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA AddN two inputs failed."));
    return;
  }

  // OPTIMIZED: 3+ inputs - custom kernel (SINGLE LAUNCH!)
  musaStream_t stream = reinterpret_cast<musaStream_t>(
      GetHandleByCtx(ctx).GetStream());

  // Build device pointer array
  std::vector<const void*> input_ptrs(num_inputs);
  for (int i = 0; i < num_inputs; ++i)
    input_ptrs[i] = ctx->input(i).tensor_data().data();

  const void** d_inputs = nullptr;
  musaMalloc(reinterpret_cast<void**>(&d_inputs),
             num_inputs * sizeof(const void*));
  musaMemcpy(const_cast<void**>(d_inputs), input_ptrs.data(),
             num_inputs * sizeof(const void*), musaMemcpyHostToDevice);

  // Launch custom kernel
  void* output_ptr = const_cast<char*>(output->tensor_data().data());
  launcher(reinterpret_cast<const T**>(d_inputs),
           reinterpret_cast<T*>(output_ptr),
           num_inputs, static_cast<int>(num_elements), stream);

  musaFree(const_cast<void**>(d_inputs));
}

// ============================================================================
// AddN operator class
// ============================================================================

template <typename T>
class MusaAddNOp : public MusaOpKernel {
 public:
  explicit MusaAddNOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    AddNCompute<T>(ctx, format_, GetLauncher());
  }

 private:
  // Getter for the appropriate launcher function
  static void (*GetLauncher())(const T**, T*, int, int, musaStream_t);
};

// ============================================================================
// Launcher function getters - specialized for each type
// ============================================================================

#define DEFINE_ADDN_LAUNCHER_GETTER(T, launcher, input_cast, output_cast)       \
  template <>                                                                   \
  void (*MusaAddNOp<T>::GetLauncher())(const T**, T*, int, int, musaStream_t) { \
    return [](const T** inputs, T* output, int num_inputs, int size,            \
              musaStream_t stream) {                                            \
      launcher(input_cast(inputs), output_cast(output),                         \
               num_inputs, size, stream);                                       \
    };                                                                          \
  }

// Float and double - direct casts
DEFINE_ADDN_LAUNCHER_GETTER(float, LaunchAddNKernelFloat,
                            reinterpret_cast<const float**>,
                            reinterpret_cast<float*>)
DEFINE_ADDN_LAUNCHER_GETTER(double, LaunchAddNKernelDouble,
                            reinterpret_cast<const double**>,
                            reinterpret_cast<double*>)

// Half and BFloat16 - need reinterpret_cast to void**
DEFINE_ADDN_LAUNCHER_GETTER(Eigen::half, LaunchAddNKernelHalf,
                            reinterpret_cast<const void**>,
                            reinterpret_cast<void*>)
DEFINE_ADDN_LAUNCHER_GETTER(bfloat16, LaunchAddNKernelBFloat16,
                            reinterpret_cast<const void**>,
                            reinterpret_cast<void*>)

// Int32 and Int64 - direct casts
DEFINE_ADDN_LAUNCHER_GETTER(int32, LaunchAddNKernelInt32,
                            reinterpret_cast<const int**>,
                            reinterpret_cast<int*>)
DEFINE_ADDN_LAUNCHER_GETTER(int64, LaunchAddNKernelInt64,
                            reinterpret_cast<const long long**>,
                            reinterpret_cast<long long*>)

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

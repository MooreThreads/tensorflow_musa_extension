#include <algorithm>
#include <numeric>
#include <vector>

#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils/logging.h"

// ============================================================================
// MUSA AddN custom kernel launcher declarations from musa_addn_kernel.mu
// ============================================================================

#define MAX_INLINE_ADDN_INPUTS 8

struct InlinePointers {
  const void* ptrs[MAX_INLINE_ADDN_INPUTS];
};

extern "C" {
void LaunchAddNKernelFloat(const float** inputs, InlinePointers inline_inputs, float* output, int num_inputs,
                           int size, musaStream_t stream);
void LaunchAddNKernelDouble(const double** inputs, InlinePointers inline_inputs, double* output,
                            int num_inputs, int size, musaStream_t stream);
void LaunchAddNKernelHalf(const void** inputs, InlinePointers inline_inputs, void* output, int num_inputs,
                          int size, musaStream_t stream);
void LaunchAddNKernelBFloat16(const void** inputs, InlinePointers inline_inputs, void* output, int num_inputs,
                              int size, musaStream_t stream);
void LaunchAddNKernelInt32(const int** inputs, InlinePointers inline_inputs, int* output, int num_inputs,
                           int size, musaStream_t stream);
void LaunchAddNKernelInt64(const int64_t** inputs, InlinePointers inline_inputs, int64_t* output,
                           int num_inputs, int size, musaStream_t stream);
}

namespace tensorflow {
namespace musa {

// Helper: Compute broadcasted shape manually
// Returns true if successful, false if incompatible
bool ComputeBroadcastShape(const std::vector<TensorShape>& input_shapes,
                           TensorShape* output_shape) {
  if (input_shapes.empty()) {
    return false;
  }

  *output_shape = input_shapes[0];

  for (size_t i = 1; i < input_shapes.size(); ++i) {
    const TensorShape& shape_b = input_shapes[i];
    TensorShape result_shape;

    // Use TensorFlow's internal logic if available, otherwise manual
    // Trying to use TensorShape::BroadcastToShapes again? No, we established it
    // doesn't exist. Let's implement a robust manual broadcast.

    int ndim_a = output_shape->dims();
    int ndim_b = shape_b.dims();
    int ndim_out = std::max(ndim_a, ndim_b);

    std::vector<int64_t> out_dims(ndim_out);

    for (int d = 0; d < ndim_out; ++d) {
      int64_t dim_a = (d < ndim_a) ? output_shape->dim_size(ndim_a - 1 - d) : 1;
      int64_t dim_b = (d < ndim_b) ? shape_b.dim_size(ndim_b - 1 - d) : 1;

      if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
        return false;  // Incompatible
      }

      out_dims[ndim_out - 1 - d] = std::max(dim_a, dim_b);
    }

    // Construct new shape
    TensorShape proposed_shape;
    for (int64_t s : out_dims) {
      proposed_shape.AddDim(s);
    }
    *output_shape = proposed_shape;
  }

  return true;
}

// Helper: Zero initialize memory
// Assumes MusaMemcpyAsyncH2D and others return mStatus based on previous
// errors. If they return musaError_t, we need to adjust. Based on error:
// "cannot convert 'mStatus' ... to 'musaError_t'", it implies the functions
// RETURN mStatus. So we will treat them as returning mStatus directly.

inline mStatus MusaZeroMemoryAsync(void* ptr, size_t size,
                                   musaStream_t stream) {
#ifdef MUSA_HAS_MEMSET
  // Assuming musaMemsetAsync returns mStatus or needs wrapper
  // If musaMemsetAsync returns musaError_t, we need conversion.
  // Let's assume for now the project uses a wrapper that returns mStatus.
  // If compilation fails here, we will switch to H2D copy.
  // For safety, let's stick to the H2D copy method which we know works with
  // MusaMemcpyAsyncH2D unless we are sure about musaMemsetAsync signature.
  // Given the errors, let's rely on MusaMemcpyAsyncH2D which is confirmed to
  // return mStatus.
#endif

  // Fallback: Host-side zero fill + H2D copy
  // This is safe and portable
  std::vector<char> zeros(size, 0);
  // MusaMemcpyAsyncH2D is confirmed to return mStatus by the compiler error
  // context
  return MusaMemcpyAsyncH2D(ptr, zeros.data(), size, stream);
}

// Common implementation for AddN Compute
template <typename T>
void AddNCompute(OpKernelContext* ctx, mFormat format,
                 void (*launcher)(const T**, InlinePointers, T*, int, int, musaStream_t)) {
  MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, "AddN");
  MUSA_KERNEL_TRACE_START("FULL");

  const int num_inputs = ctx->num_inputs();
  OP_REQUIRES(ctx, num_inputs >= 1,
              errors::InvalidArgument("AddN requires at least one input."));

  auto& handle = GetHandleByCtx(ctx);
  musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

  // Handle single input - direct copy
  if (num_inputs == 1) {
    const Tensor& input = ctx->input(0);
    Tensor* output = nullptr;
    MUSA_KERNEL_TRACE_START("Mem Alloc");
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    MUSA_KERNEL_TRACE_END("Mem Alloc");
    if (input.NumElements() == 0) return;
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());
    MUSA_KERNEL_TRACE_START("Mem Cpy");
    mStatus copy_status = MusaMemcpyAsyncD2D(
        const_cast<char*>(output->tensor_data().data()),
        input.tensor_data().data(), input.TotalBytes(), stream);
    MUSA_KERNEL_TRACE_END("Mem Cpy");
    OP_REQUIRES(ctx, copy_status == mStatus::SUCCESS,
                errors::Internal("MUSA AddN single input copy failed."));
    MUSA_KERNEL_TRACE_END("FULL");
    return;
  }

  // ==========================================================================
  // STEP 1: Calculate Broadcasted Output Shape
  // ==========================================================================
  std::vector<TensorShape> input_shapes;
  input_shapes.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_shapes.push_back(ctx->input(i).shape());
  }

  TensorShape output_shape;
  bool shapes_compatible = ComputeBroadcastShape(input_shapes, &output_shape);

  if (!shapes_compatible) {
    std::string err_msg =
        "Incompatible shapes for MUSA AddN (broadcasting failed): [";
    err_msg += input_shapes[0].DebugString();
    for (int i = 1; i < num_inputs; ++i) {
      err_msg += ", ";
      err_msg += input_shapes[i].DebugString();
    }
    err_msg += "]";
    OP_REQUIRES(ctx, false, errors::InvalidArgument(err_msg));
  }

  const size_t num_elements = output_shape.num_elements();

  // Allocate output tensor
  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  if (num_elements == 0) return;

  // ==========================================================================
  // STEP 2: Check if all inputs already match the output shape (Fast Path)
  // ==========================================================================
  bool all_shapes_match = true;
  for (int i = 0; i < num_inputs; ++i) {
    if (ctx->input(i).shape() != output_shape) {
      all_shapes_match = false;
      break;
    }
  }

  if (all_shapes_match) {
    // ----------------------------------------------------------------------
    // FAST PATH: All shapes identical. Use optimized custom kernel.
    // ----------------------------------------------------------------------

    if (num_inputs == 2) {
      mTensor t0 = CreateMTensor(ctx->input(0), format);
      mTensor t1 = CreateMTensor(ctx->input(1), format);
      mTensor t_out = CreateMTensor(*output, format);
      auto& handle = GetHandleByCtx(ctx);

      ::musa::dnn::Binary op;
      op.SetMode(::musa::dnn::Binary::Mode::ADD);

      MUSA_KERNEL_TRACE_START("Kernel");
      auto status = op.Run(handle, t_out, t0, t1);
      MUSA_KERNEL_TRACE_END("Kernel");
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("MUSA AddN two inputs (muDNN) failed."));
      MUSA_KERNEL_TRACE_END("FULL");
      return;
    }

    // 3+ inputs: Custom Kernel
    // Build device pointer array
    std::vector<const void*> input_ptrs(num_inputs);
    for (int i = 0; i < num_inputs; ++i)
      input_ptrs[i] = ctx->input(i).tensor_data().data();

    const void** d_inputs = nullptr;
    InlinePointers inline_inputs;

    if (num_inputs <= MAX_INLINE_ADDN_INPUTS) {
      for (int i = 0; i < num_inputs; ++i) {
        inline_inputs.ptrs[i] = input_ptrs[i];
      }
    } else {
      Tensor d_inputs_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_UINT64, TensorShape({num_inputs}),
                                             &d_inputs_tensor));
      d_inputs =
          reinterpret_cast<const void**>(d_inputs_tensor.flat<uint64>().data());

      mStatus status =
          MusaMemcpyAsyncH2D(const_cast<void**>(d_inputs), input_ptrs.data(),
                             num_inputs * sizeof(const void*), stream);
      OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                  errors::Internal("MUSA AddN input pointers copy failed."));
    }

    // Launch custom kernel
    void* output_ptr = const_cast<char*>(output->tensor_data().data());
    MUSA_KERNEL_TRACE_START("Kernel");
    launcher(reinterpret_cast<const T**>(d_inputs), inline_inputs,
             reinterpret_cast<T*>(output_ptr), num_inputs,
             static_cast<int>(num_elements), stream);
    MUSA_KERNEL_TRACE_END("Kernel");
    MUSA_KERNEL_TRACE_END("FULL");
  } else {
    // ----------------------------------------------------------------------
    // FALLBACK PATH: Broadcasting required.
    // ----------------------------------------------------------------------

    // 1. Zero-initialize output
    // MusaZeroMemoryAsync uses MusaMemcpyAsyncH2D which returns mStatus
    mStatus memset_status =
        MusaZeroMemoryAsync(const_cast<char*>(output->tensor_data().data()),
                            output->TotalBytes(), stream);
    OP_REQUIRES(ctx, memset_status == mStatus::SUCCESS,
                errors::Internal(
                    "MUSA AddN broadcast fallback: failed to zero output."));

    mTensor t_out = CreateMTensor(*output, format);

    // 2. Iteratively add
    for (int i = 0; i < num_inputs; ++i) {
      mTensor t_in = CreateMTensor(ctx->input(i), format);

      ::musa::dnn::Binary op;
      op.SetMode(::musa::dnn::Binary::Mode::ADD);

      auto status = op.Run(handle, t_out, t_out, t_in);

      OP_REQUIRES(
          ctx, status == ::musa::dnn::Status::SUCCESS,
          errors::Internal(
              "MUSA AddN broadcast fallback: muDNN Binary failed at input %d",
              i));
    }
    MUSA_KERNEL_TRACE_END("FULL");
  }
}

// ============================================================================
// Operator Class
// ============================================================================

template <typename T>
class MusaAddNOp : public MusaOpKernel {
 public:
  explicit MusaAddNOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}
  bool IsExpensive() override { return false; }
  void Compute(OpKernelContext* ctx) override {
    AddNCompute<T>(ctx, format_, GetLauncher());
  }

 private:
  static void (*GetLauncher())(const T**, InlinePointers, T*, int, int, musaStream_t);
};

// ============================================================================
// Launchers
// ============================================================================

#define DEFINE_ADDN_LAUNCHER_GETTER(T, launcher, input_cast, output_cast) \
  template <>                                                             \
  void (*MusaAddNOp<T>::GetLauncher())(const T**, InlinePointers, T*, int, int, \
                                       musaStream_t) {                    \
    return [](const T** inputs, InlinePointers inline_inputs, T* output, int num_inputs, int size, \
              musaStream_t stream) {                                      \
      launcher(input_cast(inputs), inline_inputs, output_cast(output), num_inputs, size, \
               stream);                                                   \
    };                                                                    \
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

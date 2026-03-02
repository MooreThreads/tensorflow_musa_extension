#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/stream.h"
#include "utils_op.h"

extern "C" {
void LaunchPackKernelFloat(const float** inputs, float* output, int num,
                           int before, int after, int total,
                           musaStream_t stream);
void LaunchPackKernelDouble(const double** inputs, double* output, int num,
                            int before, int after, int total,
                            musaStream_t stream);
void LaunchPackKernelHalf(const void** inputs, void* output, int num,
                          int before, int after, int total,
                          musaStream_t stream);
void LaunchPackKernelBFloat16(const void** inputs, void* output, int num,
                              int before, int after, int total,
                              musaStream_t stream);
void LaunchPackKernelInt32(const int** inputs, int* output, int num, int before,
                           int after, int total, musaStream_t stream);
void LaunchPackKernelInt64(const long long** inputs, long long* output, int num,
                           int before, int after, int total,
                           musaStream_t stream);

void LaunchUnpackKernelFloat(const float* input, float** outputs, int num,
                             int before, int after, int total,
                             musaStream_t stream);
void LaunchUnpackKernelDouble(const double* input, double** outputs, int num,
                              int before, int after, int total,
                              musaStream_t stream);
void LaunchUnpackKernelHalf(const void* input, void** outputs, int num,
                            int before, int after, int total,
                            musaStream_t stream);
void LaunchUnpackKernelBFloat16(const void* input, void** outputs, int num,
                                int before, int after, int total,
                                musaStream_t stream);
void LaunchUnpackKernelInt32(const int* input, int** outputs, int num,
                             int before, int after, int total,
                             musaStream_t stream);
void LaunchUnpackKernelInt64(const long long* input, long long** outputs,
                             int num, int before, int after, int total,
                             musaStream_t stream);
}

namespace tensorflow {
namespace musa {

/**
 * Pack Op 优化实现
 * 
 * 优化点：
 * 1. 添加 IsExpensive() 标记
 * 2. 延迟初始化：预分配设备内存，避免每次 Compute 都 malloc/free
 * 3. 使用异步 H2D 拷贝
 */
template <typename T>
class MusaPackOp : public MusaOpKernel {
 public:
  explicit MusaPackOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
    d_inputs_ = nullptr;
    d_inputs_capacity_ = 0;
  }

  ~MusaPackOp() override {
    if (d_inputs_ != nullptr) {
      musaFree(d_inputs_);
    }
  }

  // Pack is memory-intensive but computationally simple
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const int num_inputs = ctx->num_inputs();
    const Tensor& first_input = ctx->input(0);

    int expanded_num_dims = first_input.dims() + 1;
    int axis = axis_ < 0 ? axis_ + expanded_num_dims : axis_;

    TensorShape output_shape(first_input.shape());
    output_shape.InsertDim(axis, num_inputs);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    const int total_elements = output->NumElements();
    if (total_elements == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    int64_t before_size = 1;
    for (int i = 0; i < axis; ++i) before_size *= output_shape.dim_size(i);

    int64_t after_size = 1;
    for (int i = axis + 1; i < output_shape.dims(); ++i)
      after_size *= output_shape.dim_size(i);

    // OPTIMIZATION: Pre-allocate device memory if needed
    if (d_inputs_capacity_ < num_inputs) {
      if (d_inputs_ != nullptr) {
        musaFree(d_inputs_);
      }
      musaMalloc(reinterpret_cast<void**>(&d_inputs_),
                 num_inputs * sizeof(const void*));
      d_inputs_capacity_ = num_inputs;
    }

    // Use stack array for small inputs to avoid vector allocation
    const void* input_ptrs[16];
    const void** ptr_array = (num_inputs <= 16) ? input_ptrs : 
        new const void*[num_inputs];
    
    for (int i = 0; i < num_inputs; ++i) {
      ptr_array[i] = ctx->input(i).tensor_data().data();
    }

    // Async H2D copy
    tensorflow::musa::MusaMemcpyAsyncH2D(const_cast<void**>(d_inputs_),
                                         ptr_array,
                                         num_inputs * sizeof(const void*),
                                         stream);

    void* output_ptr = const_cast<char*>(output->tensor_data().data());
    LaunchKernel(reinterpret_cast<const void**>(d_inputs_), output_ptr,
                 num_inputs, before_size, after_size, total_elements, stream);

    if (num_inputs > 16) {
      delete[] ptr_array;
    }
  }

 private:
  int axis_;
  const void** d_inputs_;
  int d_inputs_capacity_;
  
  void LaunchKernel(const void** inputs, void* output, int num, int before,
                    int after, int total, musaStream_t stream);
};

template <>
void MusaPackOp<float>::LaunchKernel(const void** inputs, void* output, int num,
                                     int before, int after, int total,
                                     musaStream_t stream) {
  LaunchPackKernelFloat(reinterpret_cast<const float**>(inputs),
                        reinterpret_cast<float*>(output), num, before, after,
                        total, stream);
}
template <>
void MusaPackOp<double>::LaunchKernel(const void** inputs, void* output,
                                      int num, int before, int after, int total,
                                      musaStream_t stream) {
  LaunchPackKernelDouble(reinterpret_cast<const double**>(inputs),
                         reinterpret_cast<double*>(output), num, before, after,
                         total, stream);
}
template <>
void MusaPackOp<Eigen::half>::LaunchKernel(const void** inputs, void* output,
                                           int num, int before, int after,
                                           int total, musaStream_t stream) {
  LaunchPackKernelHalf(inputs, output, num, before, after, total, stream);
}
template <>
void MusaPackOp<bfloat16>::LaunchKernel(const void** inputs, void* output,
                                        int num, int before, int after,
                                        int total, musaStream_t stream) {
  LaunchPackKernelBFloat16(inputs, output, num, before, after, total, stream);
}
template <>
void MusaPackOp<int32>::LaunchKernel(const void** inputs, void* output, int num,
                                     int before, int after, int total,
                                     musaStream_t stream) {
  LaunchPackKernelInt32(reinterpret_cast<const int**>(inputs),
                        reinterpret_cast<int*>(output), num, before, after,
                        total, stream);
}
template <>
void MusaPackOp<int64>::LaunchKernel(const void** inputs, void* output, int num,
                                     int before, int after, int total,
                                     musaStream_t stream) {
  LaunchPackKernelInt64(reinterpret_cast<const long long**>(inputs),
                        reinterpret_cast<long long*>(output), num, before,
                        after, total, stream);
}

template <typename T>
class MusaUnpackOp : public MusaOpKernel {
 public:
  explicit MusaUnpackOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
    d_outputs_ = nullptr;
    d_outputs_capacity_ = 0;
  }

  ~MusaUnpackOp() override {
    if (d_outputs_ != nullptr) {
      musaFree(d_outputs_);
    }
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    int axis = axis_ < 0 ? axis_ + input.dims() : axis_;
    const int num_outputs = input.dim_size(axis);

    TensorShape output_shape = input.shape();
    output_shape.RemoveDim(axis);

    const int total_elements = input.NumElements();

    void* output_ptrs[16];
    void** ptr_array = (num_outputs <= 16) ? output_ptrs :
        new void*[num_outputs];
    
    for (int i = 0; i < num_outputs; ++i) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &output));
      if (output->NumElements() > 0) {
        ptr_array[i] = const_cast<char*>(output->tensor_data().data());
      } else {
        ptr_array[i] = nullptr;
      }
    }

    if (total_elements == 0) {
      if (num_outputs > 16) delete[] ptr_array;
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    int64_t before_size = 1;
    for (int i = 0; i < axis; ++i) before_size *= input.dim_size(i);
    int64_t after_size = 1;
    for (int i = axis + 1; i < input.dims(); ++i)
      after_size *= input.dim_size(i);

    // OPTIMIZATION: Pre-allocate device memory if needed
    if (d_outputs_capacity_ < num_outputs) {
      if (d_outputs_ != nullptr) {
        musaFree(d_outputs_);
      }
      musaMalloc(reinterpret_cast<void**>(&d_outputs_),
                 num_outputs * sizeof(void*));
      d_outputs_capacity_ = num_outputs;
    }

    tensorflow::musa::MusaMemcpyAsyncH2D(d_outputs_, ptr_array,
                                         num_outputs * sizeof(void*),
                                         stream);

    const void* input_ptr = input.tensor_data().data();
    LaunchKernel(input_ptr, d_outputs_, num_outputs, before_size, after_size,
                 total_elements, stream);

    if (num_outputs > 16) delete[] ptr_array;
  }

 private:
  int axis_;
  void** d_outputs_;
  int d_outputs_capacity_;
  
  void LaunchKernel(const void* input, void** outputs, int num, int before,
                    int after, int total, musaStream_t stream);
};

template <>
void MusaUnpackOp<float>::LaunchKernel(const void* input, void** outputs,
                                       int num, int before, int after,
                                       int total, musaStream_t stream) {
  LaunchUnpackKernelFloat(reinterpret_cast<const float*>(input),
                          reinterpret_cast<float**>(outputs), num, before,
                          after, total, stream);
}
template <>
void MusaUnpackOp<double>::LaunchKernel(const void* input, void** outputs,
                                        int num, int before, int after,
                                        int total, musaStream_t stream) {
  LaunchUnpackKernelDouble(reinterpret_cast<const double*>(input),
                           reinterpret_cast<double**>(outputs), num, before,
                           after, total, stream);
}
template <>
void MusaUnpackOp<Eigen::half>::LaunchKernel(const void* input, void** outputs,
                                             int num, int before, int after,
                                             int total, musaStream_t stream) {
  LaunchUnpackKernelHalf(input, outputs, num, before, after, total, stream);
}
template <>
void MusaUnpackOp<bfloat16>::LaunchKernel(const void* input, void** outputs,
                                          int num, int before, int after,
                                          int total, musaStream_t stream) {
  LaunchUnpackKernelBFloat16(input, outputs, num, before, after, total, stream);
}
template <>
void MusaUnpackOp<int32>::LaunchKernel(const void* input, void** outputs,
                                       int num, int before, int after,
                                       int total, musaStream_t stream) {
  LaunchUnpackKernelInt32(reinterpret_cast<const int*>(input),
                          reinterpret_cast<int**>(outputs), num, before, after,
                          total, stream);
}
template <>
void MusaUnpackOp<int64>::LaunchKernel(const void* input, void** outputs,
                                       int num, int before, int after,
                                       int total, musaStream_t stream) {
  LaunchUnpackKernelInt64(reinterpret_cast<const long long*>(input),
                          reinterpret_cast<long long**>(outputs), num, before,
                          after, total, stream);
}

#define REGISTER_MUSA_STACK_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Pack").Device(DEVICE_MTGPU).TypeConstraint<type>("T"),   \
      MusaPackOp<type>);                                             \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Unpack").Device(DEVICE_MTGPU).TypeConstraint<type>("T"), \
      MusaUnpackOp<type>);

REGISTER_MUSA_STACK_KERNELS(float);
REGISTER_MUSA_STACK_KERNELS(double);
REGISTER_MUSA_STACK_KERNELS(int32);
REGISTER_MUSA_STACK_KERNELS(int64);
REGISTER_MUSA_STACK_KERNELS(Eigen::half);
REGISTER_MUSA_STACK_KERNELS(bfloat16);

}  // namespace musa
}  // namespace tensorflow

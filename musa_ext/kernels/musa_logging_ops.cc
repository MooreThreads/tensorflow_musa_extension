#include <musa_runtime.h>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

std::string TensorToSummary(OpKernelContext* c, const Tensor& device_tensor,
                            int summarize) {
  Tensor cpu_tensor(device_tensor.dtype(), device_tensor.shape());
  MusaMemcpyD2H(const_cast<char*>(cpu_tensor.tensor_data().data()),
                device_tensor.tensor_data().data(), device_tensor.TotalBytes());
  return cpu_tensor.SummarizeValue(summarize);
}

class MusaPrintOp : public OpKernel {
 public:
  explicit MusaPrintOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("message", &message_));
    OP_REQUIRES_OK(c, c->GetAttr("first_n", &first_n_));
    OP_REQUIRES_OK(c, c->GetAttr("summarize", &summarize_));
  }

  void Compute(OpKernelContext* c) override {
    if (c->num_inputs() > 0) {
      c->set_output(0, c->input(0));
    }

    static std::atomic<tensorflow::int64> printed_count{0};
    if (first_n_ >= 0 && printed_count.load() >= first_n_) return;
    printed_count++;

    musaDeviceSynchronize();

    std::string result = message_;
    for (int i = 0; i < c->num_inputs(); ++i) {
      if (i == 0)
        result += "[";
      else
        result += ", ";
      result += TensorToSummary(c, c->input(i), summarize_);
    }
    result += "]";
    std::cerr << result << std::endl;
  }

 private:
  std::string message_;
  tensorflow::int64 first_n_;
  int32 summarize_;
};

class MusaPrintV2Op : public OpKernel {
 public:
  explicit MusaPrintV2Op(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("output_stream", &output_stream_));
    OP_REQUIRES_OK(c, c->GetAttr("end", &end_));
  }

  void Compute(OpKernelContext* c) override {
    musaDeviceSynchronize();

    const Tensor& input = c->input(0);

    Tensor cpu_tensor(input.dtype(), input.shape());
    MusaMemcpyD2H(const_cast<char*>(cpu_tensor.tensor_data().data()),
                  input.tensor_data().data(), input.TotalBytes());

    if (input.dtype() == DT_STRING) {
      auto flat = cpu_tensor.flat<tstring>();
      for (int i = 0; i < flat.size(); ++i) {
        std::cerr << flat(i);
      }
      std::cerr << end_;
    } else {
      std::cerr << "MusaPrintV2Op: Unsupported input type." << std::endl;
    }
  }

 private:
  std::string output_stream_;
  std::string end_;
};

class MusaStringFormatOp : public OpKernel {
 public:
  explicit MusaStringFormatOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("template", &template_));
    OP_REQUIRES_OK(c, c->GetAttr("placeholder", &placeholder_));
    OP_REQUIRES_OK(c, c->GetAttr("summarize", &summarize_));
  }

  void Compute(OpKernelContext* c) override {
    musaDeviceSynchronize();

    std::string result;
    size_t template_pos = 0;
    int input_idx = 0;

    while (true) {
      size_t placeholder_pos = template_.find(placeholder_, template_pos);
      if (placeholder_pos == std::string::npos) {
        result.append(template_.substr(template_pos));
        break;
      }
      result.append(
          template_.substr(template_pos, placeholder_pos - template_pos));
      if (input_idx < c->num_inputs()) {
        result.append(TensorToSummary(c, c->input(input_idx), summarize_));
        input_idx++;
      } else {
        result.append(placeholder_);
      }
      template_pos = placeholder_pos + placeholder_.length();
    }

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &output_tensor));
    output_tensor->scalar<tstring>()() = result;
  }

 private:
  std::string template_;
  std::string placeholder_;
  int32 summarize_;
};

REGISTER_KERNEL_BUILDER(Name("Print").Device("MUSA"), MusaPrintOp);
REGISTER_KERNEL_BUILDER(Name("PrintV2").Device("MUSA"), MusaPrintV2Op);  // 新增
REGISTER_KERNEL_BUILDER(Name("StringFormat").Device("MUSA"),
                        MusaStringFormatOp);

}  // namespace musa
}  // namespace tensorflow
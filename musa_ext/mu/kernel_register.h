#ifndef TENSORFLOW_MUSA_MU_KERNEL_REGISTER_H_
#define TENSORFLOW_MUSA_MU_KERNEL_REGISTER_H_

#include "../kernels/utils_op.h"
#include "./device_register.h"
#include "tensorflow/core/framework/op_kernel.h"
#ifdef MUSA_PROFILE
static std::mutex op_lock_;
#define MUSA_PROFILE_OP                                              \
  std::lock_guard<std::mutex> guard(op_lock_);                       \
  std::string kernel_label = "Musa" + OpKernel::def().op();          \
  std::string tf_op = OpKernel::name() + ":" + OpKernel::def().op(); \
  tensorflow::profiler::musa::AnnotatedTraceMe activity(             \
      [&] {                                                          \
        std::string op = tensorflow::profiler::TraceMeOp(            \
            absl::string_view(OpKernel::name()),                     \
            absl::string_view(kernel_label));                        \
        return tensorflow::profiler::TraceMeEncode(                  \
            kernel_label, {{"tf_op", tf_op},                         \
                           {"group_id", 0},                          \
                           {"is_eager", 0},                          \
                           {"context_id", "$$1"},                    \
                           {"correlation_id", correlation_id++},     \
                           {"kernel_details", "kernel_details"}});   \
      },                                                             \
      3);
#else
#define MUSA_PROFILE_OP
#endif
namespace tensorflow {
namespace musa {

typedef void (*RegFuncPtr)();

bool musaKernelRegFunc(RegFuncPtr regFunc);

class MusaAnnotatedTraceMe {
 public:
  template <typename... Args>
  explicit MusaAnnotatedTraceMe(Args&&... args) {}
};

#define MTOP_CHECK_LOG(status, str)             \
  if (::musa::dnn::Status::SUCCESS != status) { \
    LOG(ERROR) << "mudnn " << str << " faied!"; \
  }
#define MTOP_CHECK_OK_RUN(status, str, context)                            \
  {                                                                        \
    MUSA_PROFILE_OP                                                        \
    if (mStatus::SUCCESS != status) {                                      \
      OP_REQUIRES_OK(context, errors::Internal("mtdnn ", str, " error!")); \
    }                                                                      \
  }

#define MTOP_CHECK_OK(status, str, context)                              \
  if (mStatus::SUCCESS != status) {                                      \
    OP_REQUIRES_OK(context, errors::Internal("mtdnn ", str, " error!")); \
  }

}  // namespace musa
}  // namespace tensorflow

#define MUSA_KERNEL_REGISTER(name)                                 \
  static void musaKernelReg_##name();                              \
  static bool musa_kernel_registered_##name =                      \
      ::tensorflow::musa::musaKernelRegFunc(musaKernelReg_##name); \
  static void musaKernelReg_##name()

#endif  // TENSORFLOW_MUSA_MU_KERNEL_REGISTER_H_

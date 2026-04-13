#include "utils_op.h"

#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace tensorflow {
namespace musa {

namespace {
constexpr const char* kKernelDebugReset = "\033[0m";
constexpr const char* kKernelDebugTypeColor = "\033[36m";
constexpr const char* kKernelDebugShapeColor = "\033[33m";
constexpr const char* kKernelDebugEndColor = "\033[1;32m";
constexpr const char* kKernelDebugFailColor = "\033[1;31m";

mType GetType(DataType t) {
  switch (t) {
    case DataType::DT_FLOAT:
      return mType::FLOAT;
    case DataType::DT_DOUBLE:
      return mType::DOUBLE;
    case DataType::DT_INT32:
      return mType::INT32;
    case DataType::DT_UINT8:
      return mType::UINT8;
    case DataType::DT_INT16:
      return mType::INT16;
    case DataType::DT_INT8:
      return mType::INT8;
    case DataType::DT_INT64:
      return mType::INT64;
    case DataType::DT_BFLOAT16:
      return mType::BFLOAT16;
    case DataType::DT_UINT16:
      return mType::UINT16;
    case DataType::DT_HALF:
      return mType::HALF;
    case DataType::DT_UINT32:
      return mType::UINT32;
    case DataType::DT_UINT64:
      return mType::UINT64;
    case DataType::DT_BOOL:
      return mType::BOOL;
    default:
      CHECK(false);
      throw;
  }
}

std::string JoinStrings(const std::vector<std::string>& items) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << items[i];
  }
  oss << "]";
  return oss.str();
}

bool ParseBoolEnv(const char* value, bool* enabled) {
  if (value == nullptr || enabled == nullptr) {
    return false;
  }

  const std::string env_value(value);
  if (env_value == "1" || env_value == "true" || env_value == "TRUE" ||
      env_value == "on" || env_value == "ON" || env_value == "yes" ||
      env_value == "YES") {
    *enabled = true;
    return true;
  }
  if (env_value == "0" || env_value == "false" || env_value == "FALSE" ||
      env_value == "off" || env_value == "OFF" || env_value == "no" ||
      env_value == "NO") {
    *enabled = false;
    return true;
  }
  return false;
}

bool ShouldColorizeKernelDebugValues() {
  bool enabled = false;
  if (ParseBoolEnv(std::getenv("MUSA_KERNEL_DEBUG_COLOR"), &enabled)) {
    return enabled;
  }
  if (std::getenv("NO_COLOR")) {
    return false;
  }

  const char* term = std::getenv("TERM");
  if (term == nullptr || std::string(term) == "dumb") {
    return false;
  }

  return isatty(fileno(stderr)) != 0;
}

std::string ColorizeKernelDebugValue(const std::string& value,
                                     const char* color_code) {
  if (!ShouldColorizeKernelDebugValues()) {
    return value;
  }

  std::ostringstream oss;
  oss << color_code << value << kKernelDebugReset;
  return oss.str();
}

std::string GetKernelDebugOpType(const OpKernel& op) {
  return op.type_string().empty() ? op.def().op() : op.type_string();
}
}  // namespace

// Helper function to convert musaError_t to mStatus (mudnn Status).
static inline mStatus FromMusaError(musaError_t err) {
  if (err == musaSuccess) return mStatus::SUCCESS;
  return mStatus::INTERNAL_ERROR;
}

mStatus MusaFree(void* ptr) {
  if (ptr) {
    musaError_t err = musaFree(ptr);
    return FromMusaError(err);
  }
  return mStatus::SUCCESS;
}

mStatus MusaAllocate(size_t size, void** ptr) {
  musaError_t err = musaMalloc(ptr, size);
  return FromMusaError(err);
}

mTensor CreateMTensor(const Tensor& t, mFormat format) {
  mTensor rst;
  rst.SetAddr(
      const_cast<void*>(static_cast<const void*>(t.tensor_data().data())));
  rst.SetType(GetType(t.dtype()));

  auto dims_raw = t.shape().dim_sizes();
  const int rank = static_cast<int>(dims_raw.size());
  const int64_t* dims = reinterpret_cast<const int64_t*>(dims_raw.data());

  if (rank >= 4) {
    rst.SetFormat(format);
  } else {
    rst.SetFormat(mFormat::NCHW);
  }

  rst.SetNdInfo(rank, dims);
  return rst;
}

mTensor CreateMTensor(const Tensor& t) {
  mTensor rst;
  CHECK(rst.SetAddr(t.data()) == ::musa::dnn::Status::SUCCESS)
      << "SetAddr failed";
  CHECK(rst.SetType(GetType(t.dtype())) == ::musa::dnn::Status::SUCCESS)
      << "SetType failed";
  auto dims_int = t.shape().dim_sizes();
  CHECK(rst.SetNdInfo(static_cast<int>(dims_int.size()),
                      reinterpret_cast<const int64_t*>(dims_int.data())) ==
        ::musa::dnn::Status::SUCCESS)
      << "SetNdInfo failed";
  return rst;
}

mFormat GetMusaFormat(OpKernelConstruction* ctx) {
  string df;
  if (ctx->HasAttr("data_format")) {
    if (ctx->GetAttr("data_format", &df).ok()) {
      return (df == "NCHW") ? mFormat::NCHW : mFormat::NHWC;
    }
  }
  return mFormat::NHWC;
}

std::string FormatKernelDebugInputTypes(OpKernelContext* context) {
  std::vector<std::string> input_types;
  input_types.reserve(context->num_inputs());
  for (int i = 0; i < context->num_inputs(); ++i) {
    if (!context->has_input(i)) {
      input_types.push_back("<dead>");
      continue;
    }
    input_types.push_back(DataTypeString(context->input_dtype(i)));
  }
  return JoinStrings(input_types);
}

std::string FormatKernelDebugInputShapes(OpKernelContext* context) {
  std::vector<std::string> input_shapes;
  input_shapes.reserve(context->num_inputs());
  for (int i = 0; i < context->num_inputs(); ++i) {
    if (!context->has_input(i)) {
      input_shapes.push_back("<dead>");
      continue;
    }
    if (context->input_is_ref(i)) {
      input_shapes.push_back("<ref>");
      continue;
    }
    input_shapes.push_back(context->input(i).shape().DebugString());
  }
  return JoinStrings(input_shapes);
}

void LogKernelDebugStart(const std::string& op_type, OpKernelContext* context) {
#ifdef MUSA_KERNEL_DEBUG
  if (context == nullptr) {
    return;
  }

  const std::string input_types =
      ColorizeKernelDebugValue(FormatKernelDebugInputTypes(context),
                               kKernelDebugTypeColor);
  const std::string input_shapes =
      ColorizeKernelDebugValue(FormatKernelDebugInputShapes(context),
                               kKernelDebugShapeColor);

  std::fprintf(stderr,
               "[MUSA_KERNEL_DEBUG] op_type=%s input_types=%s input_shapes=%s\n",
               op_type.c_str(), input_types.c_str(), input_shapes.c_str());
  std::fflush(stderr);
#else
  (void)op_type;
  (void)context;
#endif
}

void LogKernelDebugEnd(const std::string& op_type, bool ok) {
#ifdef MUSA_KERNEL_DEBUG
  const char* phase = ok ? "END" : "FAIL";
  const char* color = ok ? kKernelDebugEndColor : kKernelDebugFailColor;
  if (ShouldColorizeKernelDebugValues()) {
    std::fprintf(stderr, "[MUSA_KERNEL_DEBUG] %s%s %s%s\n", color, phase,
                 op_type.c_str(), kKernelDebugReset);
  } else {
    std::fprintf(stderr, "[MUSA_KERNEL_DEBUG] %s %s\n", phase, op_type.c_str());
  }
  std::fflush(stderr);
#else
  (void)op_type;
  (void)ok;
#endif
}

KernelDebugScope::KernelDebugScope(const OpKernel& op, OpKernelContext* context)
    : op_type_(GetKernelDebugOpType(op)), context_(context) {
#ifdef MUSA_KERNEL_DEBUG
  LogKernelDebugStart(op_type_, context_);
#endif
}

KernelDebugScope::~KernelDebugScope() {
#ifdef MUSA_KERNEL_DEBUG
  LogKernelDebugEnd(op_type_, context_ == nullptr || context_->status().ok());
#endif
}

MusaDevice* GetDeviceByCtx(tensorflow::OpKernelContext* context) {
  DeviceBase* device_base = context->device();
  if (!device_base) {
    LOG(ERROR) << "GetDeviceByCtx: device_base is null";
    return nullptr;
  }
  MusaDevice* musa_device = reinterpret_cast<MusaDevice*>(device_base);
  if (!musa_device) {
    LOG(ERROR) << "GetDeviceByCtx: musa_device is null";
    return nullptr;
  }
  return musa_device;
}

}  // namespace musa
}  // namespace tensorflow

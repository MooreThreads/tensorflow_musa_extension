// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <iomanip>
#include <limits>
#include <sstream>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace musa {

// AsString op: Converts each entry in the given tensor to strings.
// Supports many numeric types and boolean.
//
// Attributes:
//   - precision: The post-decimal precision for floating point numbers.
//   - scientific: Use scientific notation for floating point numbers.
//   - shortest: Use shortest representation for floating point numbers.
//   - width: Pad pre-decimal numbers to this width.
//   - fill: The value to pad if width > -1.
//
// This is a CPU-only operation since the output is string type.

// Helper functions for formatting different types
namespace {

// Format integer types
template <typename T>
typename std::enable_if<std::is_integral<T>::value, std::string>::type
FormatValueImpl(T value, int32 precision, bool scientific, bool shortest,
                int32 width, const string& fill) {
  std::ostringstream oss;
  if (width > 0) {
    oss << std::setw(width);
    if (!fill.empty()) {
      oss << std::setfill(fill[0]);
    }
  }
  oss << value;
  return oss.str();
}

// Format floating point types
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, std::string>::type
FormatValueImpl(T value, int32 precision, bool scientific, bool shortest,
                int32 width, const string& fill) {
  std::ostringstream oss;
  if (width > 0) {
    oss << std::setw(width);
    if (!fill.empty()) {
      oss << std::setfill(fill[0]);
    }
  }

  if (shortest) {
    // Use shortest representation
    std::ostringstream tmp;
    tmp << std::setprecision(std::numeric_limits<T>::max_digits10) << value;
    std::string str = tmp.str();
    // Check if scientific notation is shorter
    std::ostringstream sci;
    sci << std::scientific << std::setprecision(6) << value;
    std::string sci_str = sci.str();
    oss << (sci_str.length() < str.length() ? sci_str : str);
  } else if (scientific) {
    oss << std::scientific;
    if (precision > -1) {
      oss << std::setprecision(precision);
    } else {
      oss << std::setprecision(6);
    }
    oss << value;
  } else {
    if (precision > -1) {
      oss << std::fixed << std::setprecision(precision);
    } else {
      oss << std::setprecision(6);
    }
    oss << value;
  }
  return oss.str();
}

// Format boolean
std::string FormatBool(bool value, int32 width, const string& fill) {
  std::ostringstream oss;
  if (width > 0) {
    oss << std::setw(width);
    if (!fill.empty()) {
      oss << std::setfill(fill[0]);
    }
  }
  oss << (value ? "true" : "false");
  return oss.str();
}

// Format complex numbers
template <typename T>
std::string FormatComplex(std::complex<T> value, int32 precision,
                          bool scientific, bool shortest, int32 width,
                          const string& fill) {
  std::ostringstream oss;
  oss << "("
      << FormatValueImpl(value.real(), precision, scientific, shortest, -1, "")
      << ","
      << FormatValueImpl(value.imag(), precision, scientific, shortest, -1, "")
      << ")";
  return oss.str();
}

// Format Eigen::half
std::string FormatHalf(Eigen::half value, int32 precision, bool scientific,
                       bool shortest, int32 width, const string& fill) {
  return FormatValueImpl(static_cast<float>(value), precision, scientific,
                         shortest, width, fill);
}

// Format bfloat16
std::string FormatBfloat16(bfloat16 value, int32 precision, bool scientific,
                           bool shortest, int32 width, const string& fill) {
  return FormatValueImpl(static_cast<float>(value), precision, scientific,
                         shortest, width, fill);
}

}  // namespace

template <typename T>
class MusaAsStringOp : public OpKernel {
 public:
  explicit MusaAsStringOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("precision", &precision_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scientific", &scientific_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shortest", &shortest_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("width", &width_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill", &fill_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    // Handle empty tensor
    if (input.NumElements() == 0) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
      return;
    }

    // Allocate output tensor with same shape as input
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    auto output_flat = output->flat<tstring>();
    auto input_flat = input.flat<T>();

    const int64 num_elements = input.NumElements();

    for (int64 i = 0; i < num_elements; ++i) {
      output_flat(i) = FormatValue(input_flat(i));
    }
  }

 private:
  int32 precision_;
  bool scientific_;
  bool shortest_;
  int32 width_;
  string fill_;

  // Generic FormatValue that dispatches to the appropriate helper
  std::string FormatValue(T value) {
    return FormatValueImpl(value, precision_, scientific_, shortest_, width_,
                           fill_);
  }
};

// Specialization for bool
template <>
std::string MusaAsStringOp<bool>::FormatValue(bool value) {
  return FormatBool(value, width_, fill_);
}

// Specialization for complex64
template <>
std::string MusaAsStringOp<complex64>::FormatValue(complex64 value) {
  return FormatComplex(value, precision_, scientific_, shortest_, width_,
                       fill_);
}

// Specialization for complex128
template <>
std::string MusaAsStringOp<complex128>::FormatValue(complex128 value) {
  return FormatComplex(value, precision_, scientific_, shortest_, width_,
                       fill_);
}

// Specialization for Eigen::half
template <>
std::string MusaAsStringOp<Eigen::half>::FormatValue(Eigen::half value) {
  return FormatHalf(value, precision_, scientific_, shortest_, width_, fill_);
}

// Specialization for bfloat16
template <>
std::string MusaAsStringOp<bfloat16>::FormatValue(bfloat16 value) {
  return FormatBfloat16(value, precision_, scientific_, shortest_, width_,
                        fill_);
}

// Register the kernel for all supported types
// Note: This is a CPU-only operation, so we use HostMemory for input and output

#define REGISTER_AS_STRING_KERNEL(T)                   \
  REGISTER_KERNEL_BUILDER(Name("AsString")             \
                              .Device("MUSA")          \
                              .HostMemory("input")     \
                              .HostMemory("output")    \
                              .TypeConstraint<T>("T"), \
                          MusaAsStringOp<T>);

// Register for all supported types
REGISTER_AS_STRING_KERNEL(int8);
REGISTER_AS_STRING_KERNEL(int16);
REGISTER_AS_STRING_KERNEL(int32);
REGISTER_AS_STRING_KERNEL(int64);
REGISTER_AS_STRING_KERNEL(uint8);
REGISTER_AS_STRING_KERNEL(uint16);
REGISTER_AS_STRING_KERNEL(uint32);
REGISTER_AS_STRING_KERNEL(uint64);
REGISTER_AS_STRING_KERNEL(float);
REGISTER_AS_STRING_KERNEL(double);
REGISTER_AS_STRING_KERNEL(bool);
REGISTER_AS_STRING_KERNEL(complex64);
REGISTER_AS_STRING_KERNEL(complex128);
REGISTER_AS_STRING_KERNEL(Eigen::half);
REGISTER_AS_STRING_KERNEL(bfloat16);

#undef REGISTER_AS_STRING_KERNEL

}  // namespace musa
}  // namespace tensorflow
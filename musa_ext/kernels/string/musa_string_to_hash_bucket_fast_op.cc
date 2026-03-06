/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// StringToHashBucketFast operator for MUSA device.
// This is a CPU-based operator since string hashing is typically performed
// on the host. Both input and output tensors are in host memory.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/fingerprint.h"

namespace tensorflow {
namespace musa {

class StringToHashBucketFastOp : public OpKernel {
 public:
  explicit StringToHashBucketFastOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_buckets", &num_buckets_));
    OP_REQUIRES(ctx, num_buckets_ > 0,
                errors::InvalidArgument("num_buckets must be > 0, got ",
                                        num_buckets_));
  }

  // String hashing is compute-intensive, mark as expensive to allow
  // the TensorFlow runtime to schedule it appropriately.
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));

    // Handle empty tensor case
    if (input_tensor->NumElements() == 0) {
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, input_tensor->shape(), &output_tensor));
      return;
    }

    const auto& input_flat = input_tensor->flat<tstring>();
    const int64 N = input_tensor->NumElements();

    // Allocate output tensor (host memory since HostMemory("output") is set)
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor->shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int64>();

    // Compute hash values directly on host
    for (int64 i = 0; i < N; ++i) {
      const tstring& s = input_flat(i);
      const uint64 hash = tensorflow::Fingerprint64(
          tensorflow::StringPiece(s.data(), s.size()));
      output_flat(i) = static_cast<int64>(hash % static_cast<uint64>(num_buckets_));
    }
  }

 private:
  int64 num_buckets_;
};

// Register the kernel with HostMemory for both input and output.
// String tensors are always stored in host memory, and the hash computation
// is performed on the CPU.
#define REGISTER_MUSA_KERNEL()                                             \
  REGISTER_KERNEL_BUILDER(Name("StringToHashBucketFast")                  \
                              .Device("MUSA")                             \
                              .HostMemory("input")                        \
                              .HostMemory("output"),                      \
                          StringToHashBucketFastOp);

REGISTER_MUSA_KERNEL();

}  // namespace musa
}  // namespace tensorflow

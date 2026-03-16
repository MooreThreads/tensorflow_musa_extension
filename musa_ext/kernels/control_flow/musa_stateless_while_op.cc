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

#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace musa {

// StatelessWhile implementation for MUSA
//
// This operator implements a functional while loop without maintaining state
// between iterations. All loop state is passed through function arguments.
//
// For MUSA, we fall back to CPU execution since:
// 1. The cond/body functions may contain arbitrary TensorFlow operations
// 2. Running functional while loops on GPU requires complex kernel compilation
// 3. Most use cases involve simple loops that are fast enough on CPU
//
// The implementation follows TensorFlow's StatelessWhile semantics:
// - Input: list of tensors
// - Cond: function that takes same inputs and returns scalar boolean
// - Body: function that takes same inputs and returns same number of tensors
// - Output: final loop variables after cond returns false

class MusaStatelessWhileOp : public OpKernel {
 public:
  explicit MusaStatelessWhileOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cond", &cond_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("body", &body_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &t_types_));

    OP_REQUIRES(ctx, !cond_func_.name().empty(),
                errors::InvalidArgument("cond function name cannot be empty"));
    OP_REQUIRES(ctx, !body_func_.name().empty(),
                errors::InvalidArgument("body function name cannot be empty"));
  }

  // StatelessWhile can be expensive depending on loop count
  // Mark as expensive for better scheduling
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const int num_inputs = ctx->num_inputs();

    // Collect input tensors
    std::vector<Tensor> inputs;
    inputs.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      inputs.push_back(ctx->input(i));
    }

    // Get function library
    FunctionLibraryRuntime* flr = ctx->function_library();
    OP_REQUIRES(ctx, flr != nullptr,
                errors::Internal("Function library not available"));

    // Look up cond and body functions
    FunctionLibraryRuntime::Handle cond_handle;
    FunctionLibraryRuntime::Handle body_handle;

    Status s = flr->Instantiate(cond_func_.name(),
                                AttrSlice(&cond_func_.attr()), &cond_handle);
    OP_REQUIRES_OK(ctx, s);

    s = flr->Instantiate(body_func_.name(), AttrSlice(&body_func_.attr()),
                         &body_handle);
    OP_REQUIRES_OK(ctx, s);

    // Create runtime options
    FunctionLibraryRuntime::Options opts;
    opts.rendezvous = ctx->rendezvous();
    opts.step_container = ctx->step_container();
    opts.stats_collector = ctx->stats_collector();
    opts.runner = ctx->runner();

    // Maximum number of iterations to prevent infinite loops
    const int64_t max_iterations = 100000;
    int64_t iteration = 0;

    // Execute the while loop
    std::vector<Tensor> current_inputs = inputs;

    while (iteration < max_iterations) {
      // Run cond function
      std::vector<Tensor> cond_outputs;
      s = flr->RunSync(opts, cond_handle, current_inputs, &cond_outputs);

      if (!s.ok()) {
        ctx->CtxFailure(
            errors::Internal("Cond function execution failed: ", s.ToString()));
        return;
      }

      OP_REQUIRES(ctx, !cond_outputs.empty(),
                  errors::Internal("Cond function must return at least one "
                                   "output"));

      // Check cond result
      const Tensor& cond_result = cond_outputs[0];
      OP_REQUIRES(ctx, cond_result.dims() == 0,
                  errors::InvalidArgument(
                      "Cond function must return a scalar, got shape: ",
                      cond_result.shape().DebugString()));

      bool cond_value = cond_result.scalar<bool>()();
      if (!cond_value) {
        break;  // Loop complete
      }

      // Run body function
      std::vector<Tensor> body_outputs;
      s = flr->RunSync(opts, body_handle, current_inputs, &body_outputs);

      if (!s.ok()) {
        ctx->CtxFailure(
            errors::Internal("Body function execution failed: ", s.ToString()));
        return;
      }

      OP_REQUIRES(ctx, static_cast<int>(body_outputs.size()) == num_inputs,
                  errors::InvalidArgument(
                      "Body function must return same number of inputs: ",
                      body_outputs.size(), " vs ", num_inputs));

      // Update loop variables
      current_inputs = std::move(body_outputs);
      ++iteration;
    }

    if (iteration >= max_iterations) {
      ctx->CtxFailure(errors::ResourceExhausted(
          "StatelessWhile loop exceeded maximum iterations (", max_iterations,
          ")"));
      return;
    }

    // Set outputs
    for (int i = 0; i < num_inputs; ++i) {
      ctx->set_output(i, current_inputs[i]);
    }
  }

 private:
  NameAttrList cond_func_;
  NameAttrList body_func_;
  DataTypeVector t_types_;
};

// Register StatelessWhile for MUSA
// Note: We register with all input types since the actual computation
// is done by the cond/body functions which handle their own type constraints

REGISTER_KERNEL_BUILDER(Name("StatelessWhile")
                            .Device(DEVICE_MTGPU)
                            .HostMemory("input")
                            .HostMemory("output"),
                        MusaStatelessWhileOp);

}  // namespace musa
}  // namespace tensorflow
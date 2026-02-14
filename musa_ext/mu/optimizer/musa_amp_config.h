#ifndef MUSA_AMP_CONFIG_H_
#define MUSA_AMP_CONFIG_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

class MusaAmpConfig {
 public:
  std::unordered_set<string> fp16_compute_ops = {"MatMul",
                                                 "BatchMatMul",
                                                 "BatchMatMulV2",
                                                 "Conv2D",
                                                 "Conv2DBackpropInput",
                                                 "Conv2DBackpropFilter",
                                                 "DepthwiseConv2dNative",
                                                 "Conv3D",
                                                 "FusedBatchNorm",
                                                 "FusedBatchNormV2",
                                                 "FusedBatchNormV3",
                                                 "MusaFusedMatMul",
                                                 "MusaInteract",
                                                 "Einsum"};

  std::unordered_set<string> fp32_keep_ops = {
      "Softmax",
      "LogSoftmax",
      "SoftmaxCrossEntropyWithLogits",
      "SparseSoftmaxCrossEntropyWithLogits",
      "SigmoidCrossEntropyWithLogits",
      "Mean",
      "Sum",
      "Prod",
      "L2Loss",
      "Norm",
      "Exp",
      "Log",
      "Sqrt",
      "Rsqrt",
      "Reciprocal",
      "Square"};

  std::unordered_set<string> conditional_ops = {
      "Add", "AddV2", "Sub", "Mul", "Div", "BiasAdd", "BiasAddGrad"};

  std::unordered_set<string> activation_ops = {"Relu", "Relu6",     "Elu",
                                               "Selu", "LeakyRelu", "Sigmoid",
                                               "Tanh", "Gelu",      "Swish"};

  bool aggressive_mode = false;
  bool use_loss_scaling = false;
  float loss_scale = 128.0f;
  int min_convert_elements = 1024;

  DataType target_dtype = DT_HALF;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // MUSA_AMP_CONFIG_H_

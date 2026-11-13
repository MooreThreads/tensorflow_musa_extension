# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Empty shapes for custom fusion kernels, compared with unfused CPU math."""
import numpy as np
import tensorflow as tf

from empty_dimensions_op_test import EmptyDimensionsOpTest, array
from musa_test_utils import MUSATestCase, load_musa_ops

# Importing a TestCase into this module would make unittest run its tests twice.
_Base = EmptyDimensionsOpTest
del EmptyDimensionsOpTest


class EmptyFusedOpsTest(MUSATestCase):
  check_case = _Base.check_case

  def testEmptyMatrixFusions(self):
    ops = load_musa_ops()
    for m, k, n in [(0, 3, 4), (2, 0, 4), (2, 3, 0)]:
      a, b, bias = array((m, k)), array((k, n)), np.arange(n, dtype=np.float32) - 2
      cases = [
          (ops.musa_mat_mul_bias_add, {'a': a, 'b': b, 'bias': bias}, {},
           lambda a, b, bias, name: tf.add(tf.matmul(a, b), bias, name=name)),
          (ops.musa_linear_activation, {'a': a, 'b': b, 'bias': bias}, {},
           lambda a, b, bias, name: tf.nn.relu(tf.matmul(a, b) + bias, name=name)),
          (ops.musa_reshape_mat_mul, {'x': a, 'w': b}, {},
           lambda x, w, name: tf.matmul(x, w, name=name)),
          (ops.musa_tensor_dot, {'a': a, 'b': b}, {'axes_a': [1], 'axes_b': [0]},
           lambda a, b, axes_a, axes_b, name: tf.tensordot(a, b, [axes_a, axes_b], name=name)),
          (ops.musa_tensor_dot_bias, {'a': a, 'b': b, 'bias': bias}, {'axes_a': [1], 'axes_b': [0]},
           lambda a, b, bias, axes_a, axes_b, name: tf.add(tf.tensordot(a, b, [axes_a, axes_b]), bias, name=name)),
          (ops.musa_concat_mat_mul, {'inputs': [a, a], 'axis': np.array(0, np.int32), 'other': b}, {'concat_input_idx': 0},
           lambda inputs, axis, other, concat_input_idx, name: tf.matmul(tf.concat(inputs, axis), other, name=name)),
      ]
      for op, inputs, attrs, ref in cases:
        with self.subTest(op=op.__name__, m=m, k=k, n=n):
          self.check_case(op, inputs, attrs, reference=ref)

  def testEmptyElementwiseFusions(self):
    ops = load_musa_ops()
    for shape in [(0, 3), (2, 0)]:
      x = array(shape)
      cases = [
          (ops.musa_gelu, {'x': x}, lambda x, name: tf.nn.gelu(x, name=name)),
          (ops.musa_gelu_grad, {'x': x, 'dy': x}, lambda x, dy, name: tf.identity(dy, name=name)),
          (ops.musa_clip, {'x': x, 'lo': np.array(-1., np.float32), 'hi': np.array(1., np.float32)},
           lambda x, lo, hi, name: tf.clip_by_value(x, lo, hi, name=name)),
          (ops.musa_p_relu, {'x': x, 'alpha': np.array(.2, np.float32)},
           lambda x, alpha, name: tf.add(tf.nn.relu(x), alpha * tf.minimum(x, 0), name=name)),
          (ops.musa_normalize, {'x': x, 'gamma': array((shape[-1],)), 'beta': array((shape[-1],))},
           lambda x, gamma, beta, name: tf.identity((x - tf.reduce_mean(x, -1, keepdims=True)) / tf.sqrt(tf.math.reduce_variance(x, -1, keepdims=True) + 1e-11), name=name)),
      ]
      for op, inputs, ref in cases:
        with self.subTest(op=op.__name__, shape=shape):
          self.check_case(op, inputs, reference=ref)

  def testEmptyRemainingFusions(self):
    ops = load_musa_ops()
    x = array((0, 3))
    cases = [
        (ops.musa_interact, {'input': array((2, 3, 0))}, {},
         lambda input, name: tf.matmul(input, input, transpose_b=True)),
        (ops.musa_causal_attention_forward, {'query': array((0, 2, 3, 8)), 'key': array((0, 2, 3, 8)), 'value': array((0, 2, 3, 8)), 'scale': np.array(1., np.float32)}, {},
         lambda query, key, value, scale, name: (tf.zeros([0, 2, 3, 3]), tf.identity(query))),
        (ops.musa_causal_attention_grad, {'softmax': array((0, 2, 3, 3)), 'dout': array((0, 2, 3, 8)), 'query': array((0, 2, 3, 8)), 'key': array((0, 2, 3, 8)), 'value': array((0, 2, 3, 8)), 'scale': np.array(1., np.float32)}, {},
         lambda softmax, dout, query, key, value, scale, name: (tf.identity(query), tf.identity(key), tf.identity(value))),
        (ops.musa_layer_norm, {'x': x, 'gamma': array((3,)), 'beta': array((3,))}, {},
         lambda x, gamma, beta, name: tf.identity(x, name=name)),
        (ops.musa_layer_norm_grad, {'x': x, 'dy': x, 'gamma': array((3,)), 'beta': array((3,))}, {},
         lambda x, dy, gamma, beta, name: (tf.identity(x), tf.zeros_like(gamma), tf.zeros_like(beta))),
        (ops.musa_dropout, {'x': x}, {},
         lambda x, name: (tf.identity(x), tf.cast(x, tf.bool))),
        (ops.musa_dropout_grad, {'grad': x, 'mask': array((0, 3), np.bool_)}, {},
         lambda grad, mask, name: tf.identity(grad)),
        (ops.musa_softmax_grad, {'softmax': x, 'dy': x}, {},
         lambda softmax, dy, name: tf.identity(dy)),
        (ops.musa_sigmoid_calibration, {'input': x, 'scale': np.array(1., np.float32)}, {},
         lambda input, scale, name: tf.identity(input)),
        (ops.musa_rms_norm, {'x': x, 'gamma': array((3,))}, {},
         lambda x, gamma, name: (tf.identity(x), tf.identity(x), tf.zeros([0, 1]))),
        (ops.musa_rms_norm_grad_dx, {'x': x, 'inv_rms': array((0, 1)), 'gamma': array((3,)), 'dy': x}, {},
         lambda x, inv_rms, gamma, dy, name: tf.identity(x)),
        (ops.musa_token_mixer, {'x': array((0, 2, 6))}, {'num_T': 2, 'num_H': 2, 'd_k': 3},
         lambda x, num_T, num_H, d_k, name: tf.zeros([0, num_H, num_T * d_k])),
        (ops.musa_bias_add_relu_mat_mul, {'input': array((2, 0)), 'bias': array((0,)), 'other': array((0, 3))}, {'relu_input_slot': 0},
         lambda input, bias, other, relu_input_slot, name: tf.matmul(tf.nn.relu(input + bias), other)),
    ]
    for op, inputs, attrs, ref in cases:
      with self.subTest(op=op.__name__):
        self.check_case(op, inputs, attrs, reference=ref)


# Do not expose the imported base TestCase to discovery.
del _Base

if __name__ == '__main__':
  tf.test.main()

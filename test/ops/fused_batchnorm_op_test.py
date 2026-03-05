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

"""Tests for MUSA FusedBatchNormV3 operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class FusedBatchNormOpTest(MUSATestCase):

  def testFusedBatchNormV3Forward(self):
    shape = [2, 2, 2, 4]
    dtype = tf.float32
    
    x_np = np.random.randn(*shape).astype(np.float32)
    scale_np = np.random.rand(shape[-1]).astype(np.float32)
    offset_np = np.random.rand(shape[-1]).astype(np.float32)
    mean_np = np.zeros(shape[-1]).astype(np.float32)
    var_np = np.ones(shape[-1]).astype(np.float32)

    def forward_op(x, scale, offset, mean, var):
      y, _, _, _, _, _ = tf.raw_ops.FusedBatchNormV3(
          x=x,
          scale=scale,
          offset=offset,
          mean=mean,
          variance=var,
          epsilon=0.001,
          exponential_avg_factor=1.0,
          data_format="NHWC",
          is_training=True)
      return y

    self._compare_cpu_musa_results(
        forward_op,
        [x_np, scale_np, offset_np, mean_np, var_np],
        dtype,
        rtol=1e-4,
        atol=1e-4
    )

  def testFusedBatchNormV3GradientDX(self):
    shape = [2, 2, 2, 4]
    dtype = tf.float32
    
    x_np = np.random.randn(*shape).astype(np.float32)
    scale_np = np.random.rand(shape[-1]).astype(np.float32)
    offset_np = np.random.rand(shape[-1]).astype(np.float32)
    mean_np = np.zeros(shape[-1]).astype(np.float32)
    var_np = np.ones(shape[-1]).astype(np.float32)

    def grad_dx_op(x, scale, offset, mean, var):
      # 【核心修复】强制将输入转换为 Tensor，否则 tape.watch 会报错
      x = tf.convert_to_tensor(x)
      
      with tf.GradientTape() as tape:
        tape.watch(x)
        y, _, _, _, _, _ = tf.raw_ops.FusedBatchNormV3(
            x=x,
            scale=scale,
            offset=offset,
            mean=mean,
            variance=var,
            epsilon=0.001,
            exponential_avg_factor=1.0,
            data_format="NHWC",
            is_training=True)
        loss = tf.reduce_sum(y)
      
      dx = tape.gradient(loss, x)
      return dx

    self._compare_cpu_musa_results(
        grad_dx_op,
        [x_np, scale_np, offset_np, mean_np, var_np],
        dtype,
        rtol=1e-3,
        atol=1e-3
    )


if __name__ == "__main__":
  tf.test.main()

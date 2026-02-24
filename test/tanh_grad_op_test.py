#Copyright 2026 The TensorFlow MUSA Authors.All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http:  // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#== == == == == == == == == == == == == == == == == == == == == == == == == == \

"""Tests for MUSA TanhGrad operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class TanhGradOpTest(MUSATestCase):
  """Tests for MUSA TanhGrad operator."""

  def testTanhGradBasic(self):
    """Basic tanh grad test with small tensor."""
    input_shape = [2, 3]

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

      x_np = np.random.uniform(-5.0, 5.0, size=input_shape).astype(np_dtype)
      dy_np = np.random.uniform(-2.0, 2.0, size=input_shape).astype(np_dtype)

      x = tf.constant(x_np, dtype=dtype)
      dy = tf.constant(dy_np, dtype=dtype)

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5

      def wrapper(x_in, dy_in):
        y = tf.nn.tanh(x_in)
        return tf.raw_ops.TanhGrad(y=y, dy=dy_in)

      self._compare_cpu_musa_results(
          wrapper,
          [x, dy],
          dtype=dtype,
          rtol=rtol,
          atol=atol,
      )

  def testTanhGradIntegrationTape(self):
    """Integration test: tanh gradient via tf.GradientTape (eager/tape mode)."""
    input_shape = [2, 3]

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

#生成输入（用 fp32 生成，避免极端数值；再 cast 到目标 dtype）
      x_np = np.random.uniform(-5.0, 5.0, size=input_shape).astype(np_dtype)
      x = tf.constant(x_np, dtype=dtype)

#容忍度（Tape 反向通常会有额外数值路径，低精度放宽）
      rtol = 2e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 2e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5

#Tape 计算：grad(sum(tanh(x)), x)
#用 reduce_sum 保证输出是标量，Tape 更稳定；梯度应等于 1 - tanh(x) ^ 2
      def op_func(x_in):
        with tf.GradientTape() as tape:
          tape.watch(x_in)
          y = tf.nn.tanh(x_in)
          loss = tf.reduce_sum(y)
        return tape.gradient(loss, x_in)

      self._compare_cpu_musa_results(
          op_func,
          [x],
          dtype=dtype,
          rtol=rtol,
          atol=atol,
      )    


if __name__ == "__main__":
  tf.test.main()
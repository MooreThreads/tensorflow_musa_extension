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

"""Tests for MUSA TopKV2 operator."""

import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from musa_test_utils import MUSATestCase


class TopKV2OpTest(MUSATestCase):

  def _run_topk_on_device(self, x, k_tensor, sorted, device):
    with tf.device(device):
      values, indices = tf.raw_ops.TopKV2(
          input=x,
          k=k_tensor,
          sorted=sorted)
    return values, indices

  def _test_topk(self,
                 shape,
                 k,
                 dtype,
                 sorted=True,
                 rtol=1e-5,
                 atol=1e-5):
    np_dtype = dtype.as_numpy_dtype
    if dtype == tf.bfloat16:
      
      np_dtype = np.float32

    if dtype == tf.float16:
      low, high = -5.0, 5.0
    elif dtype == tf.bfloat16:
      low, high = -3.0, 3.0
    else:
      low, high = -10.0, 10.0

    x_np = np.random.uniform(low, high, size=shape).astype(np_dtype)
    x = tf.constant(x_np, dtype=dtype)
    k_tensor = tf.constant(k, dtype=tf.int32)

    cpu_values, cpu_indices = self._run_topk_on_device(
        x, k_tensor, sorted, '/CPU:0')
    musa_values, musa_indices = self._run_topk_on_device(
        x, k_tensor, sorted, '/device:MUSA:0')

    if dtype in [tf.float16, tf.bfloat16]:
      self.assertAllClose(
          tf.cast(cpu_values, tf.float32).numpy(),
          tf.cast(musa_values, tf.float32).numpy(),
          rtol=rtol,
          atol=atol)
    else:
      self.assertAllClose(
          cpu_values.numpy(),
          musa_values.numpy(),
          rtol=rtol,
          atol=atol)

    self.assertAllEqual(cpu_indices.numpy(), musa_indices.numpy())

  def testTopKV2Float32(self):
    self._test_topk(
        shape=[10, 20],
        k=5,
        dtype=tf.float32,
        sorted=True,
        rtol=1e-4,
        atol=1e-4)

  def testTopKV2Float16(self):
    self._test_topk(
        shape=[2, 3, 16],
        k=4,
        dtype=tf.float16,
        sorted=True,
        rtol=1e-2,
        atol=1e-2)

  def testTopKV2BFloat16(self):
    self._test_topk(
        shape=[4, 12],
        k=3,
        dtype=tf.bfloat16,
        sorted=True,
        rtol=1e-1,
        atol=1e-1)

  def testTopKV2Float16K1(self):
    self._test_topk(
        shape=[3, 8],
        k=1,
        dtype=tf.float16,
        sorted=True,
        rtol=1e-2,
        atol=1e-2)

  def testTopKV2Float32ThreeDim(self):
    self._test_topk(
        shape=[2, 4, 10],
        k=2,
        dtype=tf.float32,
        sorted=True,
        rtol=1e-4,
        atol=1e-4)


if __name__ == "__main__":
  tf.test.main()
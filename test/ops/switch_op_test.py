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

"""Tests for MUSA Switch operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class SwitchOpTest(MUSATestCase):
  """Tests for tf.raw_ops.Switch on MUSA."""

  def _run_branch(self, data, pred, branch_index):
    return tf.raw_ops.Switch(data=data, pred=pred)[branch_index]

  def _compare_active_branch(self, dtype, pred_value):
    np_dtype = dtype.as_numpy_dtype
    if dtype == tf.bool:
      data_np = np.array([[True, False], [False, True]], dtype=np.bool_)
    elif np.issubdtype(np_dtype, np.integer):
      data_np = np.random.randint(-5, 6, size=(2, 3)).astype(np_dtype)
    else:
      data_np = np.random.uniform(-1.0, 1.0, size=(2, 3)).astype(np_dtype)

    data = tf.constant(data_np, dtype=dtype)
    pred = tf.constant(pred_value)

    rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
    atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8

    active_branch = 1 if pred_value else 0
    self._compare_cpu_musa_results(
        lambda input_data, input_pred: self._run_branch(
            input_data, input_pred, active_branch),
        [data, pred],
        dtype,
        rtol=rtol,
        atol=atol)
    self._compare_cpu_musa_results(
        lambda input_data, input_pred: self._run_branch(
            input_data, input_pred, 1 - active_branch),
        [data, pred],
        dtype,
        rtol=rtol,
        atol=atol)

  def _assert_invalid_pred_on_device(self, data, pred, device):
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "must be a scalar"):
      self._test_op_device_placement(
          lambda input_data, input_pred: tf.raw_ops.Switch(
              data=input_data, pred=input_pred),
          [data, pred],
          device)

  def test_switch_true_branch(self):
    for dtype in [tf.float32, tf.float16, tf.int32, tf.bool]:
      self._compare_active_branch(dtype, True)

  def test_switch_false_branch(self):
    for dtype in [tf.float32, tf.float16, tf.int32, tf.bool]:
      self._compare_active_branch(dtype, False)

  def test_pred_must_be_scalar(self):
    data = tf.constant([1, 2, 3], dtype=tf.int32)
    pred = tf.constant([True, False], dtype=tf.bool)

    self._assert_invalid_pred_on_device(data, pred, '/CPU:0')
    self._assert_invalid_pred_on_device(data, pred, '/device:MUSA:0')


if __name__ == "__main__":
  tf.test.main()

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

"""Tests for MUSA LogicalAnd operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class LogicalAndOpTest(MUSATestCase):

  def _test_logical_and(self, shape_x, shape_y):
    x_np = np.random.choice([True, False], size=shape_x).astype(np.bool_)
    y_np = np.random.choice([True, False], size=shape_y).astype(np.bool_)

    x = tf.constant(x_np, dtype=tf.bool)
    y = tf.constant(y_np, dtype=tf.bool)
    self._compare_cpu_musa_results(tf.logical_and, [x, y], tf.bool)

  def testLogicalAndBasic(self):
    self._test_logical_and([1024], [1024])

  def testLogicalAndMatrix(self):
    self._test_logical_and([64, 128], [64, 128])

  def testLogicalAndBroadcastRow(self):
    self._test_logical_and([64, 1], [64, 128])

  def testLogicalAndBroadcastScalar(self):
    self._test_logical_and([], [64, 128])

  def testLogicalAndAllTrue(self):
    x = tf.constant(np.ones([256], dtype=np.bool_), dtype=tf.bool)
    y = tf.constant(np.random.choice([True, False], size=[256]).astype(np.bool_),
                    dtype=tf.bool)
    self._compare_cpu_musa_results(tf.logical_and, [x, y], tf.bool)

  def testLogicalAndAllFalse(self):
    x = tf.constant(np.zeros([256], dtype=np.bool_), dtype=tf.bool)
    y = tf.constant(np.random.choice([True, False], size=[256]).astype(np.bool_),
                    dtype=tf.bool)
    self._compare_cpu_musa_results(tf.logical_and, [x, y], tf.bool)


if __name__ == "__main__":
  tf.test.main()

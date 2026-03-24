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

  def testLogicalAndBasic(self):
    x_np = np.random.choice([True, False], size=[1024]).astype(np.bool_)
    y_np = np.random.choice([True, False], size=[1024]).astype(np.bool_)

    x = tf.constant(x_np, dtype=tf.bool)
    y = tf.constant(y_np, dtype=tf.bool)

    self._compare_cpu_musa_results(tf.logical_and, [x, y], tf.bool)

  def testLogicalAndBroadcastRow(self):
    x_np = np.random.choice([True, False], size=[64, 1]).astype(np.bool_)
    y_np = np.random.choice([True, False], size=[64, 128]).astype(np.bool_)

    x = tf.constant(x_np, dtype=tf.bool)
    y = tf.constant(y_np, dtype=tf.bool)

    self._compare_cpu_musa_results(tf.logical_and, [x, y], tf.bool)

  def testLogicalAndScalarBroadcast(self):
    x = tf.constant(False, dtype=tf.bool)
    y_np = np.random.choice([True, False], size=[64, 128]).astype(np.bool_)
    y = tf.constant(y_np, dtype=tf.bool)

    self._compare_cpu_musa_results(tf.logical_and, [x, y], tf.bool)

  def testLogicalAndAllTrue(self):
    x_np = np.ones([256], dtype=np.bool_)
    y_np = np.ones([256], dtype=np.bool_)

    x = tf.constant(x_np, dtype=tf.bool)
    y = tf.constant(y_np, dtype=tf.bool)

    self._compare_cpu_musa_results(tf.logical_and, [x, y], tf.bool)

  def testLogicalAndAllFalse(self):
    x_np = np.zeros([256], dtype=np.bool_)
    y_np = np.random.choice([True, False], size=[256]).astype(np.bool_)

    x = tf.constant(x_np, dtype=tf.bool)
    y = tf.constant(y_np, dtype=tf.bool)

    self._compare_cpu_musa_results(tf.logical_and, [x, y], tf.bool)


if __name__ == "__main__":
  tf.test.main()

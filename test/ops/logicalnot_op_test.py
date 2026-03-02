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

"""Tests for MUSA LogicalNot operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class LogicalNotOpTest(MUSATestCase):

  def testLogicalNotBasic(self):
    x_np = np.array([True, False, True, False], dtype=np.bool_)
    x = tf.constant(x_np, dtype=tf.bool)
    self._compare_cpu_musa_results(tf.math.logical_not, [x], tf.bool)

  def testLogicalNotRandom1D(self):
    x_np = np.random.choice([True, False], size=[1024]).astype(np.bool_)
    x = tf.constant(x_np, dtype=tf.bool)
    self._compare_cpu_musa_results(tf.math.logical_not, [x], tf.bool)

  def testLogicalNot2D(self):
    x_np = np.random.choice([True, False], size=[64, 128]).astype(np.bool_)
    x = tf.constant(x_np, dtype=tf.bool)
    self._compare_cpu_musa_results(tf.math.logical_not, [x], tf.bool)

  def testLogicalNot3D(self):
    x_np = np.random.choice([True, False], size=[8, 16, 32]).astype(np.bool_)
    x = tf.constant(x_np, dtype=tf.bool)
    self._compare_cpu_musa_results(tf.math.logical_not, [x], tf.bool)

  def testLogicalNotAllTrue(self):
    x_np = np.ones([256], dtype=np.bool_)
    x = tf.constant(x_np, dtype=tf.bool)
    self._compare_cpu_musa_results(tf.math.logical_not, [x], tf.bool)

  def testLogicalNotAllFalse(self):
    x_np = np.zeros([256], dtype=np.bool_)
    x = tf.constant(x_np, dtype=tf.bool)
    self._compare_cpu_musa_results(tf.math.logical_not, [x], tf.bool)


if __name__ == "__main__":
  tf.test.main()

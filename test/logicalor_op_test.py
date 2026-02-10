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

"""Tests for MUSA Logic operators."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase

class LogicOpsTest(MUSATestCase):

  def testLogicalOr(self):
    x_np = np.random.choice([True, False], size=[1024])
    y_np = np.random.choice([True, False], size=[1024])
    
    x = tf.constant(x_np)
    y = tf.constant(y_np)
    
    self._compare_cpu_musa_results(tf.logical_or, [x, y], tf.bool)

  def testLogicalOrBroadcasting(self):
    x_np = np.random.choice([True, False], size=[64, 1])
    y_np = np.random.choice([True, False], size=[64, 128])
    
    x = tf.constant(x_np)
    y = tf.constant(y_np)
    
    self._compare_cpu_musa_results(tf.logical_or, [x, y], tf.bool)

  def testEqualFloat16(self):
    shape = [32, 32]
    x_np = np.random.randn(*shape).astype(np.float16)
    y_np = x_np.copy()
    y_np[0, 0] += 1.0
    
    x = tf.constant(x_np, dtype=tf.float16)
    y = tf.constant(y_np, dtype=tf.float16)
    
    self._compare_cpu_musa_results(tf.equal, [x, y], tf.float16)

  def testEqualBFloat16(self):
    shape = [16]
    x_np = np.random.randn(*shape).astype(np.float32)
    
    x = tf.constant(x_np, dtype=tf.bfloat16)
    y = tf.constant(x_np, dtype=tf.bfloat16)
    
    self._compare_cpu_musa_results(tf.equal, [x, y], tf.bfloat16)

if __name__ == "__main__":
  tf.test.main()



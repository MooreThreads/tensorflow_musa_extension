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

"""Tests for MUSA StatelessWhile operator."""

import tensorflow as tf

from musa_test_utils import load_musa_plugin

# Load plugin early to avoid missing device registration in test runner lifecycle.
load_musa_plugin()
MUSA_DEVICES = tf.config.list_physical_devices('MUSA')


class StatelessWhileOpTest(tf.test.TestCase):
  """Tests for MUSA StatelessWhile operator."""

  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    if not MUSA_DEVICES:
      self.skipTest("No MUSA devices found.")

  def testStatelessWhileBasic(self):
    """Test basic while loop: sum from 0 to n-1."""
    n = 10

    def cond(counter, sum_val):
      return counter < n

    def body(counter, sum_val):
      return counter + 1, sum_val + counter

    with tf.device('/device:MUSA:0'):
      final_counter, final_sum = tf.while_loop(
          cond, body, (tf.constant(0), tf.constant(0)), maximum_iterations=100)

    # Expected: counter goes 0->1->2->...->9->10, sum = 0+1+2+...+9 = 45
    self.assertEqual(final_counter.numpy(), n)
    self.assertEqual(final_sum.numpy(), 45)

  def testStatelessWhileWithTensorInput(self):
    """Test while loop with tensor as input."""
    initial = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
    n = 3

    def cond(counter, values):
      return counter < n

    def body(counter, values):
      return counter + 1, values * 2

    with tf.device('/device:MUSA:0'):
      _, result = tf.while_loop(
          cond, body, (tf.constant(0), initial), maximum_iterations=100)

    # Expected: [1,2,3,4,5] * 2^3 = [8, 16, 24, 32, 40]
    expected = initial * 8
    self.assertAllClose(result.numpy(), expected.numpy())

  def testStatelessWhileMaximumIterations(self):
    """Test that maximum_iterations is respected."""

    def cond(counter):
      return True  # Always true, should hit max_iterations

    def body(counter):
      return [counter + 1]

    with tf.device('/device:MUSA:0'):
      result = tf.while_loop(
          cond,
          body,
          [tf.constant(0)],
          maximum_iterations=10)[0]

    self.assertEqual(result.numpy(), 10)

  def testStatelessWhileZeroIterations(self):
    """Test while loop with zero iterations."""
    initial = tf.constant(100.0)

    def cond(counter):
      return False  # Never executes

    def body(counter):
      return [counter + 1]

    with tf.device('/device:MUSA:0'):
      result = tf.while_loop(
          cond,
          body,
          [initial],
          maximum_iterations=10)[0]

    self.assertEqual(result.numpy(), 100.0)

  def testStatelessWhileMultipleInputs(self):
    """Test while loop with multiple input tensors."""
    # Compute factorial of 5: 5! = 120
    n = 5

    def cond(counter, result):
      return counter < n

    def body(counter, result):
      return counter + 1, result * (counter + 1)

    with tf.device('/device:MUSA:0'):
      _, final_result = tf.while_loop(
          cond, body, (tf.constant(1), tf.constant(1)), maximum_iterations=100)

    self.assertEqual(final_result.numpy(), 120)

  def testStatelessWhileFloat(self):
    """Test while loop with float values."""
    # Compute sum of geometric series: 1 + 0.5 + 0.25 + ... while sum < 2

    def cond(counter, current_val, total):
      del counter
      return current_val > 0.01

    def body(counter, current_val, total):
      return counter + 1, current_val * 0.5, total + current_val

    with tf.device('/device:MUSA:0'):
      _, _, final_sum = tf.while_loop(
          cond,
          body,
          (tf.constant(0), tf.constant(1.0), tf.constant(0.0)),
          maximum_iterations=100)

    # Expected: sum approaches 2.0 (geometric series sum = a/(1-r) = 1/(1-0.5) = 2)
    self.assertAllClose(final_sum.numpy(), 2.0, rtol=0.1)

  def testStatelessWhileCompareWithCPU(self):
    """Compare MUSA result with CPU result."""
    n = 20

    def cond(counter, sum_val):
      return counter < n

    def body(counter, sum_val):
      return counter + 1, sum_val + counter * 2

    # Run on CPU
    with tf.device('/CPU:0'):
      cpu_counter, cpu_sum = tf.while_loop(
          cond, body, (tf.constant(0), tf.constant(0)), maximum_iterations=100)

    # Run on MUSA
    with tf.device('/device:MUSA:0'):
      musa_counter, musa_sum = tf.while_loop(
          cond, body, (tf.constant(0), tf.constant(0)), maximum_iterations=100)

    self.assertEqual(cpu_counter.numpy(), musa_counter.numpy())
    self.assertEqual(cpu_sum.numpy(), musa_sum.numpy())


if __name__ == "__main__":
  tf.test.main()

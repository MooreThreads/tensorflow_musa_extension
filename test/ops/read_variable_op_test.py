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

"""Tests for MUSA ReadVariableOp."""

import tensorflow as tf

from musa_test_utils import load_musa_plugin

# Load plugin early to avoid missing device registration in test runner lifecycle.
load_musa_plugin()
MUSA_DEVICES = tf.config.list_physical_devices('MUSA')


class ReadVariableOpTest(tf.test.TestCase):
  """Dedicated tests for ReadVariableOp on MUSA."""

  def testReadVariableOpRaw(self):
    """ReadVariableOp returns the current ResourceVariable value on MUSA."""
    if not MUSA_DEVICES:
      self.skipTest("No MUSA devices found.")

    with tf.device('/device:MUSA:0'):
      var = tf.Variable([1.0, 2.0, 3.0], dtype=tf.float32)
      out = tf.raw_ops.ReadVariableOp(resource=var.handle, dtype=tf.float32)

    self.assertAllClose([1.0, 2.0, 3.0], out.numpy())

  def testReadVariableOpAfterAssign(self):
    """ReadVariableOp sees updated value after assign."""
    if not MUSA_DEVICES:
      self.skipTest("No MUSA devices found.")

    with tf.device('/device:MUSA:0'):
      var = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32)
      var.assign([4.0, 5.0, 6.0])
      out = tf.raw_ops.ReadVariableOp(resource=var.handle, dtype=tf.float32)

    self.assertAllClose([4.0, 5.0, 6.0], out.numpy())

  def testReadValueMatchesRawOp(self):
    """tf.Variable.read_value() matches tf.raw_ops.ReadVariableOp()."""
    if not MUSA_DEVICES:
      self.skipTest("No MUSA devices found.")

    with tf.device('/device:MUSA:0'):
      var = tf.Variable([[7.0, 8.0], [9.0, 10.0]], dtype=tf.float32)
      raw_val = tf.raw_ops.ReadVariableOp(resource=var.handle, dtype=tf.float32)
      high_level_val = var.read_value()

    self.assertAllClose(raw_val.numpy(), high_level_val.numpy())

  def testReadUninitializedVarFails(self):
    """ReadVariableOp should fail on uninitialized resource variable."""
    if not MUSA_DEVICES:
      self.skipTest("No MUSA devices found.")

    handle_name = "read_variable_uninit_test"
    with tf.device('/device:MUSA:0'):
      handle = tf.raw_ops.VarHandleOp(
          dtype=tf.float32,
          shape=tf.TensorShape([2]),
          container="",
          shared_name=handle_name)

      with self.assertRaises((tf.errors.NotFoundError, tf.errors.FailedPreconditionError)):
        _ = tf.raw_ops.ReadVariableOp(resource=handle, dtype=tf.float32)


if __name__ == '__main__':
  tf.test.main()

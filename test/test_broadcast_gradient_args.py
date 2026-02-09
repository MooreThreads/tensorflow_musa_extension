from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test as test_lib

def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        tf.load_library(plugin_path)

class BroadcastGradientArgsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(BroadcastGradientArgsTest, self).setUp()
    load_musa_plugin()

  def _testBroadcastGradientArgs(self, s0, s1, expected_r0, expected_r1):
    # Ensure inputs are on MUSA device context, though HostMemory kernel handles placement
    with tf.device('/device:MUSA:0'):
        r0, r1 = gen_array_ops.broadcast_gradient_args(s0, s1)
        self.assertAllEqual(r0, expected_r0)
        self.assertAllEqual(r1, expected_r1)

  def testBasic(self):
    if not tf.config.list_physical_devices('MUSA'):
        self.skipTest("MUSA device not found")

    # Case 1: No broadcasting needed
    self._testBroadcastGradientArgs(
        [2, 3, 5], [2, 3, 5], 
        [], [])

    # Case 2: Simple broadcast dim 0
    self._testBroadcastGradientArgs(
        [1, 3, 5], [2, 3, 5], 
        [0], [])

    # Case 3: Simple broadcast dim 1
    self._testBroadcastGradientArgs(
        [2, 1, 5], [2, 3, 5], 
        [1], [])

    # Case 4: Both broadcast
    self._testBroadcastGradientArgs(
        [2, 1, 5], [1, 3, 5], 
        [1], [0])

  def testComplexShapes(self):
    if not tf.config.list_physical_devices('MUSA'):
        self.skipTest("MUSA device not found")

    # Case: [2, 3, 4] vs [4] -> [2, 3, 4] vs [1, 1, 4]
    # s0 needs no reduction. s1 needs to reduce dims 0 and 1.
    self._testBroadcastGradientArgs(
        [2, 3, 4], [4], 
        [], [0, 1])

    # Case: [1] vs [2, 3, 4]
    self._testBroadcastGradientArgs(
        [1], [2, 3, 4], 
        [0, 1, 2], [])

  def testIncompatibleShapes(self):
    if not tf.config.list_physical_devices('MUSA'):
        self.skipTest("MUSA device not found")

    with self.assertRaisesRegex(errors.InvalidArgumentError, "Incompatible shapes"):
        with tf.device('/device:MUSA:0'):
            gen_array_ops.broadcast_gradient_args([2, 3], [2, 4])

  def testInt64Types(self):
    if not tf.config.list_physical_devices('MUSA'):
        self.skipTest("MUSA device not found")

    s0 = constant_op.constant([2, 1, 5], dtype=dtypes.int64)
    s1 = constant_op.constant([1, 3, 5], dtype=dtypes.int64)
    expected_r0 = np.array([1], dtype=np.int64)
    expected_r1 = np.array([0], dtype=np.int64)
    
    with tf.device('/device:MUSA:0'):
        r0, r1 = gen_array_ops.broadcast_gradient_args(s0, s1)
        self.assertAllEqual(r0, expected_r0)
        self.assertAllEqual(r1, expected_r1)
        self.assertEqual(r0.dtype, dtypes.int64)
        self.assertEqual(r1.dtype, dtypes.int64)

if __name__ == "__main__":
  test_lib.main()

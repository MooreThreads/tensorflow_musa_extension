from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        tf.load_library(plugin_path)

@test_util.run_all_in_graph_and_eager_modes
class UnsortedSegmentSumTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(UnsortedSegmentSumTest, self).setUp()
    load_musa_plugin()

  def test_basic_np_array(self):
    if not tf.config.list_physical_devices('MUSA'):
        self.skipTest("MUSA device not found")

    inputs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = 2
    output_array = np.array([[8, 10, 12], [4, 5, 6]], dtype=np.float32)

    with tf.device('/device:MUSA:0'):
        res = self.evaluate(
            math_ops.unsorted_segment_sum(
                data=inputs,
                segment_ids=segment_ids,
                num_segments=num_segments))
    self.assertAllEqual(res.shape, output_array.shape)
    self.assertAllClose(res, output_array)

  def test_segment_id_and_input_empty(self):
    if not tf.config.list_physical_devices('MUSA'):
        self.skipTest("MUSA device not found")

    inputs = np.array([], dtype=np.float32)
    segment_ids = np.array([], dtype=np.int32)
    num_segments = 3
    output_array = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    with tf.device('/device:MUSA:0'):
        res = self.evaluate(
            math_ops.unsorted_segment_sum(
                data=inputs,
                segment_ids=segment_ids,
                num_segments=num_segments))
    self.assertAllEqual(res.shape, output_array.shape)
    self.assertAllClose(res, output_array)

  def test_type_check(self):
    if not tf.config.list_physical_devices('MUSA'):
        self.skipTest("MUSA device not found")

    inputs = np.array([[1, 2], [3, 4]], dtype=np.int32)
    segment_ids = np.array([1, 0], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    output_array = np.array([[3, 4], [1, 2]], dtype=np.int32)

    with tf.device('/device:MUSA:0'):
        res = self.evaluate(
            math_ops.unsorted_segment_sum(
                data=inputs,
                segment_ids=segment_ids,
                num_segments=num_segments))
    self.assertAllEqual(res.shape, output_array.shape)
    self.assertAllEqual(res, output_array)

    segment_ids = np.array([1, 0], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int64)
    with tf.device('/device:MUSA:0'):
        res = self.evaluate(
            math_ops.unsorted_segment_sum(
                data=inputs,
                segment_ids=segment_ids,
                num_segments=num_segments))
    self.assertAllEqual(res.shape, output_array.shape)
    self.assertAllEqual(res, output_array)

  def test_basic_tensor(self):
    if not tf.config.list_physical_devices('MUSA'):
        self.skipTest("MUSA device not found")

    inputs = constant_op.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    segment_ids = constant_op.constant([1, 0, 1])
    num_segments = 2
    output_array = constant_op.constant([[3.0, 4.0], [6.0, 8.0]])

    with tf.device('/device:MUSA:0'):
        res = self.evaluate(
            math_ops.unsorted_segment_sum(
                data=inputs,
                segment_ids=segment_ids,
                num_segments=num_segments))
    self.assertAllClose(res, output_array)
    self.assertAllEqual(res.shape, output_array.get_shape())

  @parameterized.parameters([
      {
          'inputs': [[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]],
          'segment_ids': [0, 1, 0],
          'num_segments': 2,
          'output_array': [[5, 5, 5, 5], [5, 6, 7, 8]]
      },
      {
          'inputs': [[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]],
          'segment_ids': [0, 1, 2],
          'num_segments': 3,
          'output_array': [[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]]
      },
      {
          'inputs': [[1, 2], [3, 4], [5, 6]],
          'segment_ids': [1, 0, 1],
          'num_segments': 2,
          'output_array': [[3, 4], [6, 8]]
      },
  ])
  def test_multiple_cases(self, inputs, segment_ids, num_segments, output_array):
    if not tf.config.list_physical_devices('MUSA'):
        self.skipTest("MUSA device not found")
    
    with tf.device('/device:MUSA:0'):
        res = self.evaluate(
            math_ops.unsorted_segment_sum(
                data=np.array(inputs, dtype=np.int32),
                segment_ids=np.array(segment_ids, dtype=np.int32),
                num_segments=num_segments))
    self.assertAllEqual(res, np.array(output_array, dtype=np.int32))

  def test_fail_segment_id_negative(self):
    if not tf.config.list_physical_devices('MUSA'):
        self.skipTest("MUSA device not found")

    inputs = [[1, 2], [3, 4]]
    segment_ids = [-1, 0] 
    num_segments = 2
    
    # UnsortedSegmentSum drops negative indices rather than raising error, 
    # so we test that it sums correctly (ignoring the negative one).
    expected_output = [[3, 4], [0, 0]]
    
    with tf.device('/device:MUSA:0'):
        res = self.evaluate(
            math_ops.unsorted_segment_sum(
                data=np.array(inputs, dtype=np.int32),
                segment_ids=np.array(segment_ids, dtype=np.int32),
                num_segments=num_segments))
    self.assertAllEqual(res, expected_output)

  def test_fail_segment_id_out_of_range(self):
    if not tf.config.list_physical_devices('MUSA'):
        self.skipTest("MUSA device not found")

    inputs = [[1, 2], [3, 4]]
    segment_ids = [2, 0] 
    num_segments = 2
    
    # Indices >= num_segments are dropped.
    expected_output = [[3, 4], [0, 0]]
    
    with tf.device('/device:MUSA:0'):
        res = self.evaluate(
            math_ops.unsorted_segment_sum(
                data=np.array(inputs, dtype=np.int32),
                segment_ids=np.array(segment_ids, dtype=np.int32),
                num_segments=num_segments))
    self.assertAllEqual(res, expected_output)

if __name__ == '__main__':
  test.main()

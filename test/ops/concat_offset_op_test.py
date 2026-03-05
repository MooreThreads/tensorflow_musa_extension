# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA ConcatOffset operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ConcatOffsetOpTest(MUSATestCase):

  def _test_concat_offset(self, concat_dim, shapes):
    t_concat_dim = tf.constant(concat_dim, dtype=tf.int32)
    t_shapes = [tf.constant(s, dtype=tf.int32) for s in shapes]

    # ConcatOffset returns a list of tensors [offset0, offset1, ...]
    # We need wrappers to test each output individually against CPU baseline

    for i in range(len(shapes)):
        def op_wrapper(*inputs):
            # inputs: [concat_dim, shape0, shape1, ...]
            dim = inputs[0]
            input_shapes = inputs[1:]
            return tf.raw_ops.ConcatOffset(concat_dim=dim, shape=input_shapes)[i]

        # Combine concat_dim and shapes into a single input list for the wrapper
        input_tensors = [t_concat_dim] + t_shapes

        self._compare_cpu_musa_results(
            op_wrapper,
            input_tensors,
            dtype=tf.int32,
            rtol=0, atol=0
        )

  def testConcatOffsetBasic(self):
    concat_dim = 1
    shape0 = [2, 3]
    shape1 = [2, 5]
    self._test_concat_offset(concat_dim, [shape0, shape1])

  def testConcatOffsetThreeTensors(self):
    concat_dim = 0
    shape0 = [10, 2]
    shape1 = [20, 2]
    shape2 = [30, 2]
    self._test_concat_offset(concat_dim, [shape0, shape1, shape2])

  def testConcatOffsetLargeDim(self):
    concat_dim = 3
    shape0 = [2, 2, 2, 5, 2]
    shape1 = [2, 2, 2, 10, 2]
    self._test_concat_offset(concat_dim, [shape0, shape1])


if __name__ == "__main__":
  tf.test.main()

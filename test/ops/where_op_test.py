# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA Where operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class WhereOpTest(MUSATestCase):
  """Tests for MUSA Where operator."""

  def _compare_cpu_musa(self, condition, x=None, y=None, dtype=None, rtol=1e-5, atol=1e-8):
    """Internal helper to compare CPU and MUSA results for Where op."""
    if x is None and y is None:
      # tf.where(condition) returns indices where condition is True
      def op_func(c):
        return tf.where(c)
      
      with tf.device("/cpu:0"):
        cpu_result = op_func(condition)
      with tf.device("/device:MUSA:0"):
        musa_result = op_func(condition)
      
      self.assertAllClose(cpu_result.numpy(), musa_result.numpy(), rtol=rtol, atol=atol)
    else:
      # tf.where(condition, x, y) returns elements from x or y based on condition
      def op_func(c, vx, vy):
        return tf.where(c, vx, vy)
      
      with tf.device("/cpu:0"):
        cpu_result = op_func(condition, x, y)
      with tf.device("/device:MUSA:0"):
        musa_result = op_func(condition, x, y)
      
      if dtype in [tf.float16, tf.bfloat16]:
        cpu_result = tf.cast(cpu_result, tf.float32)
        musa_result = tf.cast(musa_result, tf.float32)

      self.assertAllClose(cpu_result.numpy(), musa_result.numpy(), rtol=rtol, atol=atol)

  def testWhereIndices(self):
    """Test tf.where(condition) which returns coordinates of True values."""
    # 1D
    condition = tf.constant([True, False, True, True])
    self._compare_cpu_musa(condition)

    # 2D
    condition = tf.constant([[True, False], [False, True]])
    self._compare_cpu_musa(condition)

    # 3D
    condition = tf.random.uniform([2, 3, 4]) > 0.5
    self._compare_cpu_musa(condition)

  def testWhereSelect(self):
    """Test tf.where(condition, x, y) with various dtypes."""
    test_shapes = [
        [4],
        [2, 3],
        [2, 3, 4]
    ]

    # Based on musa_where_op.cc and musa_select_op.cc, the following types are supported
    dtypes = [
        tf.float32, tf.float16, tf.bfloat16,
        tf.int32, tf.int64, tf.bool,
        tf.int8, tf.uint8, tf.int16, tf.uint16
    ]

    for shape in test_shapes:
      condition_np = np.random.choice([True, False], size=shape)
      condition = tf.constant(condition_np)
      
      for dtype in dtypes:
        if dtype == tf.bfloat16:
          x_np = np.random.uniform(-10, 10, size=shape).astype(np.float32)
          y_np = np.random.uniform(-10, 10, size=shape).astype(np.float32)
          x = tf.cast(tf.constant(x_np), tf.bfloat16)
          y = tf.cast(tf.constant(y_np), tf.bfloat16)
        elif dtype == tf.bool:
          x = tf.constant(np.random.choice([True, False], size=shape))
          y = tf.constant(np.random.choice([True, False], size=shape))
        elif dtype.is_integer:
          x = tf.constant(np.random.randint(-100, 100, size=shape), dtype=dtype)
          y = tf.constant(np.random.randint(-100, 100, size=shape), dtype=dtype)
        else:
          x = tf.constant(np.random.uniform(-10, 10, size=shape), dtype=dtype)
          y = tf.constant(np.random.uniform(-10, 10, size=shape), dtype=dtype)

        rtol, atol = 1e-5, 1e-8
        if dtype in [tf.float16, tf.bfloat16]:
          rtol, atol = 1e-2, 1e-2
        
        self._compare_cpu_musa(condition, x, y, dtype=dtype, rtol=rtol, atol=atol)

  def testWhereBroadcasting(self):
    """Test tf.where(condition, x, y) with broadcasting."""
    # Condition is 1D, x and y are 2D
    condition = tf.constant([True, False, True])
    x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    y = tf.constant([[7, 8, 9], [10, 11, 12]], dtype=tf.float32)
    # Note: tf.where supports limited broadcasting
    self._compare_cpu_musa(condition, x, y)

  def testWhereEmpty(self):
    """Test where with empty inputs."""
    condition = tf.constant([], dtype=tf.bool)
    self._compare_cpu_musa(condition)


if __name__ == "__main__":
  tf.test.main()

# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA BroadcastGradientArgs operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class BroadcastGradientArgsTest(MUSATestCase):
  """Tests for MUSA BroadcastGradientArgs operator."""

  def _test_broadcast_args(self, s0, s1, dtype=tf.int32):
    """
    辅助函数：分别验证 BroadcastGradientArgs 的两个输出 r0 和 r1。
    """
    # 1. 构造输入 Tensor
    t_s0 = tf.constant(s0, dtype=dtype)
    t_s1 = tf.constant(s1, dtype=dtype)
    
    # 2. 定义 Wrapper，规避工具类只能处理单个 Tensor 返回值的限制
    # BroadcastGradientArgs 返回 [r0, r1]，工具类无法直接处理列表返回值
    
    # op_wrapper_r0 只返回第一个输出 Tensor (r0)
    def op_wrapper_r0(x, y):
        return tf.raw_ops.BroadcastGradientArgs(s0=x, s1=y)[0]

    # op_wrapper_r1 只返回第二个输出 Tensor (r1)
    def op_wrapper_r1(x, y):
        return tf.raw_ops.BroadcastGradientArgs(s0=x, s1=y)[1]

    # 3. 验证 r0 (s0 的广播索引)

    self._compare_cpu_musa_results(
        op_wrapper_r0,
        [t_s0, t_s1],
        dtype=dtype,
        rtol=0, atol=0 # 整数索引必须严格相等
    )
    
    # 4. 验证 r1 (s1 的广播索引)
    self._compare_cpu_musa_results(
        op_wrapper_r1,
        [t_s0, t_s1],
        dtype=dtype,
        rtol=0, atol=0
    )

  def testBasic(self):
    """Test basic broadcasting scenarios."""
    # Case 1: 无需广播
    self._test_broadcast_args([2, 3, 5], [2, 3, 5])

    # Case 2: 简单广播 dim 0
    self._test_broadcast_args([1, 3, 5], [2, 3, 5])

    # Case 3: 简单广播 dim 1
    self._test_broadcast_args([2, 1, 5], [2, 3, 5])

    # Case 4: 双向广播
    self._test_broadcast_args([2, 1, 5], [1, 3, 5])

  def testComplexShapes(self):
    """Test more complex broadcasting shapes."""
    # Case: [2, 3, 4] vs [4]
    self._test_broadcast_args([2, 3, 4], [4])

    # Case: [1] vs [2, 3, 4]
    self._test_broadcast_args([1], [2, 3, 4])
    
    # Case: 高维复杂广播 [5, 1, 3, 1] vs [1, 4, 1, 2]
    self._test_broadcast_args([5, 1, 3, 1], [1, 4, 1, 2])

  def testInt64Types(self):
    """Test with int64 input types (Important for large models)."""
    # 验证 int64 类型的处理能力
    self._test_broadcast_args([2, 1, 5], [1, 3, 5], dtype=tf.int64)

  def testIncompatibleShapes(self):
    """Test that incompatible shapes raise an error."""

    s0 = tf.constant([2, 3], dtype=tf.int32)
    s1 = tf.constant([2, 4], dtype=tf.int32) 
    
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "Incompatible"):
        with tf.device('/device:MUSA:0'):
             tf.raw_ops.BroadcastGradientArgs(s0=s0, s1=s1)


if __name__ == "__main__":
  tf.test.main()

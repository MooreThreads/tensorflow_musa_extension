# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA ArgMax operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ArgMaxOpTest(MUSATestCase):
  """Tests for MUSA ArgMax operator."""

  def testArgMaxBasic(self):
    """Test ArgMax with 1D array across different input types."""
    shape = [1000]
    
    # 覆盖官方测试中的常见类型，并加上 bf16
    input_dtypes = [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64, tf.bool]
    
    for dtype in input_dtypes:
        # 1. 数据生成
        if dtype == tf.bool:
            x_np = np.random.choice([False, True], size=shape)
            
            x_np[100] = True 
        else:
            np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
            x_np = np.random.uniform(-100, 100, size=shape).astype(np_dtype)
        
            x_np[500] = 10000 
            
        x = tf.constant(x_np, dtype=dtype)
        
        # 2. 调用对比
            
        self._compare_cpu_musa_results(
            lambda t: tf.argmax(t, axis=0), 
            [x], 
            dtype=tf.int64
        )

  def testArgMaxAxes(self):
    """
    Test ArgMax along different axes for a multi-dimensional tensor.
 
    """
    
    shape = [2, 3, 4, 5]
    x_np = np.random.randn(*shape).astype(np.float32)
    x = tf.constant(x_np, dtype=tf.float32)
    
    # 测试正向轴 (0, 1, 2, 3) 和 负向轴 (-1, -2, ...)
    axes_to_test = range(-len(shape), len(shape))
    
    for axis in axes_to_test:
        self._compare_cpu_musa_results(
            lambda t: tf.argmax(t, axis=axis),
            [x],
            dtype=tf.int64
        )

  def testArgMaxOutputType(self):
    """Test output_type argument (int32 vs int64)."""

    x = tf.constant([1.0, 5.0, 2.0, 4.0], dtype=tf.float32)
    
    # Case 1: 默认输出 (int64)
    self._compare_cpu_musa_results(
        lambda t: tf.argmax(t, axis=0),
        [x],
        dtype=tf.int64
    )
    
    # Case 2: 指定输出为 int32
    self._compare_cpu_musa_results(
        lambda t: tf.argmax(t, axis=0, output_type=tf.int32),
        [x],
        dtype=tf.int32     )

  def testArgMaxTieBreaking(self):
    """Test behavior when multiple max values exist (Should return first index)."""
    # 构造数据：索引 2 和 索引 4 都是最大值 10.0
    x_np = np.array([1.0, 5.0, 10.0, 8.0, 10.0, 2.0], dtype=np.float32)
    x = tf.constant(x_np)
    
    # 预期结果应该是 2 (第一个最大值的索引)
    self._compare_cpu_musa_results(
        lambda t: tf.argmax(t),
        [x],
        dtype=tf.int64
    )


if __name__ == "__main__":
  tf.test.main()

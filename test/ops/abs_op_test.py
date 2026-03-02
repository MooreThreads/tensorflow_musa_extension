# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA Abs operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class AbsOpTest(MUSATestCase):
  """Tests for MUSA Abs operator."""

  def testAbsCornerCases(self):
    """Test abs with specific corner cases (negative, zero, positive)."""
    # 1. 典型值测试
    data = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    
    # 针对 float32 进行高精度测试
    dtype = tf.float32
    x = tf.constant(data, dtype=dtype)
    
    # 调用基类的对比方法
    self._compare_cpu_musa_results(tf.abs, [x], dtype=dtype, rtol=1e-6, atol=1e-6)

  def testAbsBasic(self):
    """Test basic abs operation with random data."""
    shape = [1024, 1024]
    
    # 2. 覆盖三种核心数据类型
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      # 处理 numpy 类型兼容性 (numpy 不支持 bfloat16，用 float32 代替生成)
      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
      
      # 生成 -10 到 10 的随机数
      x_np = np.random.uniform(-10, 10, size=shape).astype(np_dtype)
      x = tf.constant(x_np, dtype=dtype)
      
     
      rtol = 1e-2 if dtype != tf.float32 else 1e-5
      atol = 1e-2 if dtype != tf.float32 else 1e-8
      
      self._compare_cpu_musa_results(tf.abs, [x], dtype=dtype, rtol=rtol, atol=atol)

  def testAbsDifferentShapes(self):
    """Test abs with various different shapes."""
    # 3. 覆盖不同形状，测试内存布局和广播兼容性
    test_shapes = [
        [1],            # 标量
        [5],            # 向量
        [3, 4],         # 矩阵
        [2, 3, 4],      # 3D 张量
        [5, 1, 10],     # 含 1 的维度 (广播常见场景)
    ]
    
    dtype = tf.float32
    for shape in test_shapes:
      x_np = np.random.uniform(-10, 10, size=shape).astype(np.float32)
      x = tf.constant(x_np, dtype=dtype)
      
      self._compare_cpu_musa_results(tf.abs, [x], dtype=dtype)


if __name__ == "__main__":
  
  tf.test.main()

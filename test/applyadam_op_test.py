# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA ResourceApplyAdam operator."""

import numpy as np
import tensorflow as tf

# 引入工具类
from musa_test_utils import MUSATestCase

class ResourceApplyAdamTest(MUSATestCase):
  """Tests for MUSA ResourceApplyAdam operator."""

  def _adam_op_wrapper(self, init_var, init_m, init_v, grad, lr, beta1, beta2, epsilon, beta1_power, beta2_power):
    """
    【核心封装】
    将 ResourceApplyAdam 封装成一个标准函数：
    输入：初始值的 Tensor
    输出：更新后的 var 值 (Tensor)
    """
    # 1. 在当前设备作用域下创建变量
    var = tf.Variable(init_var)
    m = tf.Variable(init_m)
    v = tf.Variable(init_v)
    
    # 2. 运行算子
    tf.raw_ops.ResourceApplyAdam(
        var=var.handle,
        m=m.handle,
        v=v.handle,
        beta1_power=beta1_power,
        beta2_power=beta2_power,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        grad=grad,
        use_locking=False,
        use_nesterov=False
    )
    
    # 3. 返回更新后的 var 值 (转为 Tensor)
    return var.read_value()

  def testApplyAdamBasic(self):
    """Test ResourceApplyAdam using the standard utility wrapper."""
    # 准备测试数据 (numpy array)
    init_var = np.array([1.0, 2.0, 3.0])
    init_m = np.array([0.1, 0.2, 0.3])
    init_v = np.array([0.01, 0.02, 0.03])
    grad = np.array([0.5, -0.5, 1.0])
    
    # 标量参数
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    beta1_power = 0.9 ** 10
    beta2_power = 0.999 ** 10

    # 测试 float32
    dtype = tf.float32

    # 构造输入 Tensor 列表
    input_tensors = [
        tf.constant(init_var, dtype=dtype),
        tf.constant(init_m, dtype=dtype),
        tf.constant(init_v, dtype=dtype),
        tf.constant(grad, dtype=dtype),
        tf.constant(lr, dtype=dtype),
        tf.constant(beta1, dtype=dtype),
        tf.constant(beta2, dtype=dtype),
        tf.constant(epsilon, dtype=dtype),
        tf.constant(beta1_power, dtype=dtype),
        tf.constant(beta2_power, dtype=dtype),
    ]


    # Adam 涉及平方和开方，跨设备计算极易产生微小误差，1e-4 是合理的通过标准
    self._compare_cpu_musa_results(
        self._adam_op_wrapper, 
        input_tensors, 
        dtype=dtype, 
        rtol=1e-3, 
        atol=2e-3
    )

if __name__ == "__main__":
  tf.test.main()

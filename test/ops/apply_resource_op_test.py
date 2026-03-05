# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA ResourceApplyAdam operator."""

import numpy as np
import tensorflow as tf

# 引入工具类 (确保该文件在 PYTHONPATH 或当前目录下)
from musa_test_utils import MUSATestCase


class ResourceApplyAdamTest(MUSATestCase):
  """Tests for MUSA ResourceApplyAdam operator."""

  def testApplyAdamBasic(self):
    """
    Test basic functionality of ResourceApplyAdam using standard test utils.
    """
    # --- 1. 准备测试数据 ---
    dtype = tf.float32
    # 使用 float32 以匹配 C++ 注册的类型
    init_var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    init_m = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    init_v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    grad_val = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    # --- 2. 定义封装函数 (Wrapper) ---
    # 这个函数会被 _compare_cpu_musa_results 调用两次：
    # 一次在 CPU 环境下，一次在 MUSA 环境下
    def op_wrapper(v_init, m_init, v_init_val, g_val):
        # A. 创建资源变量 (Variable)
        # 这里的变量会自动跟随当前上下文的设备（测试 MUSA 时在显存，测试 CPU 时在内存）
        var = tf.Variable(v_init)
        m = tf.Variable(m_init)
        v = tf.Variable(v_init_val)
        
        # B. 准备标量参数 (常量)
        # 【关键修改】：根据 C++ Kernel 的 .HostMemory(...) 定义，
        # lr, beta 等参数必须位于 CPU 内存中。
        # 即使算子在 MUSA 上运行，这些标量输入也必须来自 CPU。
        with tf.device("CPU:0"):
            lr = tf.constant(0.01, dtype=dtype)
            beta1 = tf.constant(0.9, dtype=dtype)
            beta2 = tf.constant(0.999, dtype=dtype)
            epsilon = tf.constant(1e-8, dtype=dtype)
            beta1_power = tf.constant(0.9, dtype=dtype)
            beta2_power = tf.constant(0.999, dtype=dtype)

        # C. 运行 ResourceApplyAdam 算子
        # 这里的 input 顺序必须与 C++ Compute 函数中的 ctx->input(i) 顺序严格一致
        tf.raw_ops.ResourceApplyAdam(
            var=var.handle,
            m=m.handle,
            v=v.handle,
            beta1_power=beta1_power, # input(3)
            beta2_power=beta2_power, # input(4)
            lr=lr,                   # input(5)
            beta1=beta1,             # input(6)
            beta2=beta2,             # input(7)
            epsilon=epsilon,         # input(8)
            grad=g_val,              # input(9) - 梯度通常在 Device 上
            use_locking=False
        )
        
        # D. 返回结果
        return var.read_value()

    # --- 3. 执行对比 ---
    self._compare_cpu_musa_results(
        op_wrapper,
        [init_var, init_m, init_v, grad_val],
        dtype=dtype,
        # 针对 MUSA 硬件精度差异（无 TF32 开关）进行的阈值调整
        rtol=1e-3,
        atol=2e-3
    )


if __name__ == "__main__":
  tf.test.main()
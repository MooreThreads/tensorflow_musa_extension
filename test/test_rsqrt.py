import tensorflow as tf
import os
import numpy as np

# 加载插件
plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
tf.load_library(plugin_path)

def test_rsqrt():
    print("\n--- Testing Rsqrt Op on MUSA ---")
    
    # 准备输入数据
    input_data = [1.0, 4.0, 16.0]
    expected_output = [1.0, 0.5, 0.25]

    # 使用上下文管理器指定设备
    with tf.device('/device:MUSA:0'):
        # 1. 创建常量（此时会被搬运到 MUSA）
        x = tf.constant(input_data, dtype=tf.float32)
        
        # 2. 执行 Rsqrt 计算
        # 这会触发你刚才写的 MusaRsqrtOp
        y = tf.math.rsqrt(x)
        
        print(f"Input: {x.numpy()}")
        print(f"Output: {y.numpy()}")

        # 3. 验证精度
        np.testing.assert_allclose(y.numpy(), expected_output, atol=1e-5)
        print("SUCCESS: Rsqrt calculation is accurate!")
def squeeze():
    # 显式指定设备，确保触发 Musa 算子
    with tf.device('/device:MUSA:0'):
        x = tf.ones([1, 10, 1], dtype=tf.float32)
        y = tf.squeeze(x)
        # 这里 print(y) 会触发同步并搬运数据回 CPU 展示
        print(y)

if __name__ == "__main__":
    test_rsqrt()
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    squeeze()


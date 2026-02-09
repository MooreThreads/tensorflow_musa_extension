import tensorflow as tf
import numpy as np
def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"

    if os.path.exists(plugin_path):
        print(f"DEBUG: Found MUSA plugin at {plugin_path}")
        try:
            tf.load_library(plugin_path)
            print("SUCCESS: MUSA plugin loaded successfully!")
        except Exception as e:
            print(f"FAILED: Error loading plugin: {e}")
    else:
        print(f"ERROR: Plugin not found at {plugin_path}. Did you run build.sh?")
import os
load_musa_plugin()
def test_bf16_cast():
    print("\n[Testing] float32 --> bfloat16 on MUSA")
    
    # 1. 准备数据 (使用一些 BF16 能表达但 FP16 容易溢出的较大数值)
    # BF16 最大值约为 3.39e38，与 FP32 相当
    np_data = np.array([1.0, 10.0, 65504.0, 1e10], dtype=np.float32)
    
    with tf.device('/device:MUSA:0'):
        input_t = tf.constant(np_data, dtype=tf.float32)
        # 执行 Cast
        output_t = tf.cast(input_t, dtype=tf.bfloat16)
        # 转回 float32 进行验证
        back_t = tf.cast(output_t, dtype=tf.float32)
        
        device_name = output_t.device
        print(f"  - Device: {device_name}")

    # 2. 验证
    # BF16 精度较低（尾数位少），所以比较时 atol 要放宽
    actual = back_t.numpy()
    print(f"  - Original: {np_data}")
    print(f"  - After BF16 Cast: {actual}")

    if np.allclose(actual, np_data, rtol=0.01):
        print("  - Result: SUCCESS (Within BF16 precision tolerance)")
    else:
        print("  - Result: FAILED")

# 运行测试
try:
    test_bf16_cast()
except Exception as e:
    print(f"Error during BF16 test: {e}")

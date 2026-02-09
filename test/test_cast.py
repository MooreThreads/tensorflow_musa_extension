import tensorflow as tf
import numpy as np

def test_cast_functionality(src_dtype, dst_dtype, shape=(2, 3)):
    print(f"\n[Testing] {src_dtype} --> {dst_dtype} on MUSA")
    
    # 1. 准备原始数据
    if src_dtype == tf.bool:
        np_data = np.random.choice([True, False], size=shape)
    else:
        np_data = (np.random.rand(*shape) * 10).astype(src_dtype.as_numpy_dtype)
    
    # 2. 创建 MUSA 张量
    with tf.device('/device:MUSA:0'):
        input_tensor = tf.constant(np_data, dtype=src_dtype)
        # 执行 Cast 操作
        output_tensor = tf.cast(input_tensor, dtype=dst_dtype)
        
        # 验证设备
        device_name = output_tensor.device
        print(f"  - Device: {device_name}")
    
    # 3. 验证结果
    expected = np_data.astype(dst_dtype.as_numpy_dtype)
    actual = output_tensor.numpy()
    
    if np.allclose(actual, expected, atol=1e-5):
        print(f"  - Result: SUCCESS (Values match)")
    else:
        print(f"  - Result: FAILED")
        print(f"    Expected: {expected}")
        print(f"    Actual:   {actual}")

# --- 开始测试 ---

# 测试 Wide & Deep 最常用的几种组合
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
test_cast_functionality(tf.float32, tf.int32)    # 浮点转整型
test_cast_functionality(tf.int64, tf.float32)    # ID转浮点 (Embedding前置操作)
test_cast_functionality(tf.float32, tf.half)     # 混合精度
test_cast_functionality(tf.bool, tf.float32)     # 布尔转浮点

print("\nAll MUSA Cast tests completed.")

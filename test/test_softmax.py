import numpy as np
import tensorflow as tf
import os

# 确保加载了你的 musa 插件
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

def test_softmax_on_musa(dtype, shape=(2, 8)):
    print(f"\n--- Testing DType: {dtype} ---")
    
    # 1. 构造输入数据
    data = np.random.randn(*shape).astype(np.float32)
    
    # 2. 在 CPU 上计算基准值 (作为真值)
    with tf.device('/CPU:0'):
        input_cpu = tf.convert_to_tensor(data, dtype=dtype)
        output_cpu = tf.nn.softmax(input_cpu)
        output_log_cpu = tf.nn.log_softmax(input_cpu)

    # 3. 在 MUSA 设备上计算 (你的算子)
    # 注意：确保你的设备名称与代码中定义的 DEVICE_MTGPU ("MUSA") 一致
    try:
        with tf.device('/device:MUSA:0'):
            input_musa = tf.convert_to_tensor(data, dtype=dtype)
            output_musa = tf.nn.softmax(input_musa)
            output_log_musa = tf.nn.log_softmax(input_musa)
            
        # 4. 结果比对
        # 对于 FP16/BF16，我们放宽容差 (atol)
        tolerance = 1e-3 if dtype == tf.float32 else 1e-2
        
        np.testing.assert_allclose(output_musa.numpy(), output_cpu.numpy(), 
                                   atol=tolerance, err_msg="Softmax Mismatch")
        np.testing.assert_allclose(output_log_musa.numpy(), output_log_cpu.numpy(), 
                                   atol=tolerance, err_msg="LogSoftmax Mismatch")
        
        print(f"SUCCESS: MUSA {dtype} results match CPU baseline.")
        
    except Exception as e:
        print(f"FAILED: {e}")

# 执行测试
if __name__ == "__main__":
    # 检查 MUSA 设备是否可见
    print("Available devices:", tf.config.list_physical_devices())
    load_musa_plugin()
    test_softmax_on_musa(tf.float32)
    test_softmax_on_musa(tf.float16)
    test_softmax_on_musa(tf.bfloat16)


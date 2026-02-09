import os
import time
import numpy as np
import tensorflow as tf

def test_musa_sqrt():
    # 1. 加载插件
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        try:
            tf.load_library(plugin_path)
            print(">> MUSA Plugin Loaded")
        except Exception as e:
            print(f"Warning: Could not load plugin: {e}")

    # 2. 测试配置
    test_configs = [
        ("float32", tf.float32),
        ("float16", tf.float16),
        ("bfloat16", tf.bfloat16),
    ]
    
    shape = [256, 4096]

    for name, dt in test_configs:
        print(f"\n" + "="*50)
        print(f"[Test Case] Shape: {shape}, Dtype: {name}")
        try:
            # 准备数据
            input_np = np.random.uniform(1.0, 100.0, size=shape).astype(np.float32 if dt == tf.bfloat16 else dt.as_numpy_dtype)
            
            # MUSA Run
            with tf.device('/device:MUSA:0'):
                if dt == tf.bfloat16:
                    x_musa = tf.cast(tf.constant(input_np), tf.bfloat16)
                else:
                    x_musa = tf.constant(input_np, dtype=dt)
                
                # Warmup
                for _ in range(5):
                    _ = tf.sqrt(x_musa)
                
                iters = 20
                start_musa = time.perf_counter()
                for _ in range(iters):
                    y_musa_perf = tf.sqrt(x_musa)
                # 确保 GPU 任务完成
                _ = y_musa_perf.numpy() 
                musa_time = ((time.perf_counter() - start_musa) * 1000) / iters

            # CPU Run
            with tf.device('/CPU:0'):
                if dt == tf.bfloat16:
                    x_cpu = tf.cast(tf.constant(input_np), tf.bfloat16)
                else:
                    x_cpu = tf.constant(input_np, dtype=dt)
                
                start_cpu = time.perf_counter()
                for _ in range(iters):
                    y_cpu_perf = tf.sqrt(x_cpu)
                _ = y_cpu_perf.numpy()
                cpu_time = ((time.perf_counter() - start_cpu) * 1000) / iters

            # 结果对比
            y_musa_val = y_musa_perf.numpy()
            y_cpu_val = y_cpu_perf.numpy()

            if dt == tf.bfloat16:
                y_musa_val = y_musa_val.astype(np.float32)
                y_cpu_val = y_cpu_val.astype(np.float32)

            mae = np.mean(np.abs(y_cpu_val.astype(np.float64) - y_musa_val.astype(np.float64)))
            speedup = cpu_time / musa_time
            
            # bfloat16 精度较低，阈值放宽
            threshold = 1e-2 if dt == tf.bfloat16 else 1e-3
            result = "PASS" if mae < threshold else "FAIL"

            print(f"Device    : {x_musa.device}") 
            print(f"MAE       : {mae:.7e}")
            print(f"CPU Time  : {cpu_time:.3f} ms")
            print(f"MUSA Time : {musa_time:.3f} ms")
            print(f"Speedup   : {speedup:.2f}x")
            print(f"Result    : {result}")
            
            # Functional Check
            print(f"\n[Functional Check] First 5 elements:")
            flat_in = input_np.flatten()[:5]
            flat_cpu = y_cpu_val.flatten()[:5]
            flat_musa = y_musa_val.flatten()[:5]
            print(f"  Input:  {flat_in}")
            print(f"  CPU:    {flat_cpu}")
            print(f"  MUSA:   {flat_musa}")

        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_musa_sqrt()
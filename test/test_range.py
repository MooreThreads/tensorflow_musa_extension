import tensorflow as tf
import numpy as np
import os
import traceback

def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        tf.load_library(plugin_path)
        print(">> MUSA Plugin Loaded")
    else:
        print(">> MUSA Plugin Not Found")

def assert_all_equal(v1, v2, msg=""):
    v1 = v1.numpy() if isinstance(v1, tf.Tensor) else v1
    v2 = v2.numpy() if isinstance(v2, tf.Tensor) else v2
    
    if np.allclose(v1, v2):
        print(f"[PASS] {msg}")
    else:
        print(f"[FAIL] {msg}")
        print(f"  Expected: {v2}")
        print(f"  Got:      {v1}")

def run_test(start, limit, delta, dtype):
    print(f"\n--- Range Test [{np.dtype(dtype).name}] ({start}, {limit}, {delta}) ---")
    try:
        start_val = dtype(start)
        limit_val = dtype(limit)
        delta_val = dtype(delta)

        with tf.device('/device:MUSA:0'):
            res_musa = tf.range(start_val, limit_val, delta_val, dtype=dtype)
            if dtype == np.float32:
                 print(f"  > Output Device: {res_musa.device}")

        with tf.device('/CPU:0'):
            res_cpu = tf.range(start_val, limit_val, delta_val, dtype=dtype)
        
        assert_all_equal(res_musa, res_cpu, "Value Check")
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    load_musa_plugin()
    print("MUSA Devices:", tf.config.list_physical_devices('MUSA'))
    
    if tf.config.list_physical_devices('MUSA'):
        run_test(0.0, 10.0, 1.0, np.float32)
        run_test(0.0, 5.0, 0.5, np.float32)
        run_test(0, 10, 2, np.int32)
        run_test(10, 0, -1, np.int32)
        run_test(0, 100, 20, np.int64)
        run_test(0.0, 1.0, 0.1, np.float64)

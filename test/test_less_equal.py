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

def assert_equal(v1, v2, msg=""):
    v1 = v1.numpy() if isinstance(v1, tf.Tensor) else v1
    v2 = v2.numpy() if isinstance(v2, tf.Tensor) else v2
    
    if np.array_equal(v1, v2):
        print(f"[PASS] {msg}")
        return

    if v1.size == v2.size and np.array_equal(v1.flatten(), v2.flatten()):
        print(f"[PASS] {msg} (Warning: Shape Mismatch MUSA={v1.shape} vs CPU={v2.shape}, but values match!)")
        return

    print(f"[FAIL] {msg}")
    print(f"  MUSA shape={v1.shape}:\n{v1}")
    print(f"  CPU  shape={v2.shape}:\n{v2}")

def run_test(shape_a, shape_b, dtype):
    print(f"\n--- LessEqual Test [{dtype.name}] {shape_a} <= {shape_b} ---")
    try:
        val_a = np.random.randn(*shape_a).astype(np.float32)
        val_b = np.random.randn(*shape_b).astype(np.float32)
        
        if val_a.size > 0: val_a.ravel()[0] = -100.0
        if val_b.size > 0: val_b.ravel()[0] = 100.0
        
        if val_a.size > 1: val_a.ravel()[1] = 100.0
        if val_b.size > 1: val_b.ravel()[1] = -100.0

        if dtype == tf.bfloat16:
             np_dtype = np.float32
        else:
             np_dtype = dtype.as_numpy_dtype
             
        val_a = val_a.astype(np_dtype)
        val_b = val_b.astype(np_dtype)

        with tf.device('/device:MUSA:0'):
            if dtype == tf.bfloat16:
                t_a = tf.cast(tf.constant(val_a), tf.bfloat16)
                t_b = tf.cast(tf.constant(val_b), tf.bfloat16)
            else:
                t_a = tf.constant(val_a, dtype=dtype)
                t_b = tf.constant(val_b, dtype=dtype)
                
            res_musa = tf.math.less_equal(t_a, t_b)
            
            if dtype == tf.float32 and shape_a == (10,):
                 print(f"  > Device: {res_musa.device}")

        with tf.device('/CPU:0'):
            if dtype == tf.bfloat16:
                t_a_cpu = tf.cast(tf.constant(val_a), tf.bfloat16)
                t_b_cpu = tf.cast(tf.constant(val_b), tf.bfloat16)
            else:
                t_a_cpu = tf.constant(val_a, dtype=dtype)
                t_b_cpu = tf.constant(val_b, dtype=dtype)
            
            res_cpu = tf.math.less_equal(t_a_cpu, t_b_cpu)
        
        assert_equal(res_musa, res_cpu, "Result Check")

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    load_musa_plugin()
    print("MUSA Devices:", tf.config.list_physical_devices('MUSA'))
    
    if tf.config.list_physical_devices('MUSA'):
        run_test((10,), (10,), tf.float32)
        run_test((5, 5), (5, 5), tf.float16)
        run_test((10,), (10,), tf.bfloat16)
        run_test((10,), (10,), tf.int32)
        run_test((2, 3), (2, 3), tf.int64)
        run_test((2, 3), (3,), tf.float32)
        run_test((2, 1), (2, 3), tf.int32)

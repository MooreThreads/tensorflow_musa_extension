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

def assert_all_close(v1, v2, msg="", rtol=1e-5, atol=1e-5):
    v1 = v1.numpy() if isinstance(v1, tf.Tensor) else v1
    v2 = v2.numpy() if isinstance(v2, tf.Tensor) else v2
    
    dtype_v1 = getattr(v1.dtype, 'name', str(v1.dtype))
    dtype_v2 = getattr(v2.dtype, 'name', str(v2.dtype))

    if dtype_v1 == 'bfloat16': v1 = v1.astype(np.float32)
    if dtype_v2 == 'bfloat16': v2 = v2.astype(np.float32)
    
    if np.issubdtype(v1.dtype, np.floating):
        if 'float16' in dtype_v1 or 'float16' in dtype_v2:
            rtol, atol = 1e-3, 1e-3
        
        if np.allclose(v1, v2, rtol=rtol, atol=atol):
            print(f"[PASS] {msg}")
        else:
            diff = np.abs(v1 - v2)
            print(f"[FAIL] {msg}")
            print(f"  Max Diff: {np.max(diff)}")
    else:
        if np.array_equal(v1, v2):
             print(f"[PASS] {msg}")
        else:
             print(f"[FAIL] {msg}")

def run_test(shape, dtype):
    print(f"\n--- ReluGrad Test [{dtype.name}] Shape={shape} ---")
    try:
        x_val = np.random.randn(*shape).astype(np.float32) * 5.0
        
        if x_val.size > 0:
            x_val.ravel()[0] = 0.0
            if x_val.size > 1: x_val.ravel()[1] = -1.0
            if x_val.size > 2: x_val.ravel()[2] = 1.0
        
        if dtype == tf.bfloat16:
            np_dtype = np.float32
        else:
            np_dtype = dtype.as_numpy_dtype
            
        x_val = x_val.astype(np_dtype)

        with tf.device('/device:MUSA:0'):
            if dtype == tf.bfloat16:
                x = tf.cast(tf.constant(x_val), tf.bfloat16)
            else:
                x = tf.constant(x_val, dtype=dtype)
                
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = tf.nn.relu(x)
            grad_musa = tape.gradient(y, x)
            
            if dtype == tf.float32 and shape == (10, 10):
                 print(f"  > Grad Device: {grad_musa.device}")

        with tf.device('/CPU:0'):
            if dtype == tf.bfloat16:
                x_cpu = tf.cast(tf.constant(x_val), tf.bfloat16)
            else:
                x_cpu = tf.constant(x_val, dtype=dtype)
                
            with tf.GradientTape() as tape:
                tape.watch(x_cpu)
                y_cpu = tf.nn.relu(x_cpu)
            grad_cpu = tape.gradient(y_cpu, x_cpu)
        
        assert_all_close(grad_musa, grad_cpu, f"Grad Check")

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    load_musa_plugin()
    print("MUSA Devices:", tf.config.list_physical_devices('MUSA'))
    
    if tf.config.list_physical_devices('MUSA'):
        shapes = [
            (100,), 
            (10, 10),
        ]
        
        dtypes = [tf.float32, tf.float16, tf.bfloat16]
        
        for shape in shapes:
            for dt in dtypes:
                run_test(shape, dt)

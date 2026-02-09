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
    
    if 'float16' in dtype_v1 or 'float16' in dtype_v2:
        rtol, atol = 1e-2, 1e-2

    if np.allclose(v1, v2, rtol=rtol, atol=atol):
        print(f"[PASS] {msg}")
    else:
        diff = np.abs(v1 - v2)
        print(f"[FAIL] {msg}")
        print(f"  Max Diff: {np.max(diff)}")
        print(f"  Mean Diff: {np.mean(diff)}")

def run_test(shape, dtype):
    print(f"\n--- Log Test [{np.dtype(dtype).name}] Shape={shape} ---")
    try:
        x = np.abs(np.array(np.random.randn(*shape))) + 0.1
        x = x.astype(dtype)
        
        with tf.device('/device:MUSA:0'):
            x_musa = tf.constant(x)
            res_musa = tf.math.log(x_musa)
            
            if dtype == np.float32 and shape == (10,):
                 print(f"  > Device: {res_musa.device}")

        with tf.device('/CPU:0'):
            x_cpu = tf.constant(x)
            res_cpu = tf.math.log(x_cpu)
        
        assert_all_close(res_musa, res_cpu, f"Val Check")
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()

def run_bf16_test(shape):
    print(f"\n--- Log Test [bfloat16] Shape={shape} ---")
    try:
        x = np.abs(np.array(np.random.randn(*shape))) + 0.1
        x = x.astype(np.float32)
        
        with tf.device('/device:MUSA:0'):
            vx = tf.cast(tf.constant(x), tf.bfloat16)
            res_musa = tf.math.log(vx)

        with tf.device('/CPU:0'):
             vx = tf.cast(tf.constant(x), tf.bfloat16)
             res_cpu = tf.math.log(vx)
        
        assert_all_close(res_musa, res_cpu, f"Val Check BF16")
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    load_musa_plugin()
    print("MUSA Devices:", tf.config.list_physical_devices('MUSA'))
    
    if tf.config.list_physical_devices('MUSA'):
        shapes = [
            (10,), 
            (5, 5),
            (2, 3, 4)
        ]
        
        dtypes = [np.float32, np.float64, np.float16]
        
        for shape in shapes:
            for dt in dtypes:
                run_test(shape, dt)
            run_bf16_test(shape)


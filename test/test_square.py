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

def assert_all_equal(v1, v2, msg="", rtol=1e-5, atol=1e-5):
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
            print(f"  Mean Diff: {np.mean(diff)}")
    else:
        if np.array_equal(v1, v2):
            print(f"[PASS] {msg}")
        else:
            print(f"[FAIL] {msg}")

def run_forward_test(input_shape, dtype):
    dtype_name = np.dtype(dtype).name
    print(f"\n--- Forward Test [{dtype_name}] Shape={input_shape} ---")
    try:
        if np.issubdtype(dtype, np.integer):
            x = np.random.randint(-1000, 1000, size=input_shape).astype(dtype)
        else:
            x = np.array(np.random.randn(*input_shape)).astype(dtype) * 10.0
            if x.size > 0:
                flat_x = x.ravel()
                flat_x[0] = 0.0
                if x.size > 1: flat_x[1] = -5.5
                if x.size > 2: flat_x[2] = 0.5
        
        with tf.device('/device:MUSA:0'):
            x_musa = tf.constant(x)
            y_musa = tf.math.square(x_musa)
            if dtype == np.float32 and input_shape == (2,2):
                print(f"  > Device: {y_musa.device}")

        with tf.device('/CPU:0'):
            x_cpu = tf.constant(x)
            y_cpu = tf.math.square(x_cpu)

        assert_all_equal(y_musa, y_cpu, f"Val Check ({dtype_name})")
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()

def run_bfloat16_forward_test(input_shape):
    print(f"\n--- Forward Test [bfloat16] Shape={input_shape} ---")
    try:
        x = np.array(np.random.randn(*input_shape)).astype(np.float32) * 5.0
        with tf.device('/device:MUSA:0'):
            x_bf16 = tf.cast(tf.constant(x), tf.bfloat16)
            y_musa = tf.math.square(x_bf16)
        with tf.device('/CPU:0'):
            x_bf16_cpu = tf.cast(tf.constant(x), tf.bfloat16)
            y_cpu = tf.math.square(x_bf16_cpu)
        assert_all_equal(y_musa, y_cpu, "Val Check (bfloat16)")
    except Exception as e:
        print(f"[FAIL] bfloat16 Error: {e}")
        traceback.print_exc()

def run_gradient_test(input_shape, dtype):
    dtype_name = np.dtype(dtype).name
    if not np.issubdtype(dtype, np.floating):
        return
    print(f"\n--- Gradient Test [{dtype_name}] Shape={input_shape} ---")
    try:
        x_val = np.array(np.random.randn(*input_shape)).astype(dtype)
        with tf.device('/device:MUSA:0'):
            x = tf.constant(x_val)
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = tf.math.square(x)
            grad = tape.gradient(y, x)
        expected_grad = 2.0 * x_val
        assert_all_equal(grad, expected_grad, f"Grad Check ({dtype_name})")
    except Exception as e:
        print(f"[FAIL] Gradient Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    load_musa_plugin()
    print("MUSA Devices:", tf.config.list_physical_devices('MUSA'))
    if tf.config.list_physical_devices('MUSA'):
        shapes = [
            (),
            (10,),
            (5, 5),
            (2, 3, 4, 5)
        ]
        float_types = [np.float32, np.float16]
        int_types = [np.int32, np.int64]
        print("\n=== Running Forward Tests ===")
        for shape in shapes:
            for dt in float_types + int_types:
                run_forward_test(shape, dt)
            run_bfloat16_forward_test(shape)
        print("\n=== Running Float64 Fallback Test ===")
        run_forward_test((10,), np.float64)
        print("\n=== Running Gradient Tests ===")
        grad_shapes = [(), (2, 2), (2, 3, 2)]
        for shape in grad_shapes:
            for dt in float_types:
                run_gradient_test(shape, dt)

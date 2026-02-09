import numpy as np
import tensorflow as tf
import time
import os

def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        tf.load_library(plugin_path)
        print(f"MUSA plugin loaded from: {plugin_path}")
    else:
        print(f"Warning: MUSA plugin not found at {plugin_path}. Running on available devices.")

def verify_greater_performance(shape, dtype):
    np_dtype = dtype.as_numpy_dtype
    
    if dtype in [tf.float32, tf.float16]:
        x_np = np.random.rand(*shape).astype(np_dtype) * 100 - 50
        y_np = np.random.rand(*shape).astype(np_dtype) * 100 - 50
    elif dtype in [tf.int32, tf.int64]:
        x_np = np.random.randint(-100, 100, size=shape, dtype=np_dtype)
        y_np = np.random.randint(-100, 100, size=shape, dtype=np_dtype)
    else:
        print(f"Warning: Unsupported dtype {dtype.name} for random generation in performance test, skipping.")
        return

    with tf.device('/CPU:0'):
        x_cpu = tf.constant(x_np, dtype=dtype)
        y_cpu = tf.constant(y_np, dtype=dtype)
        
        _ = tf.greater(x_cpu, y_cpu) # Warmup
        
        start_time = time.perf_counter()
        iters = 10
        for _ in range(iters):
            cpu_result = tf.greater(x_cpu, y_cpu)
        cpu_time = (time.perf_counter() - start_time) / iters * 1000

    musa_time = -1.0 
    speedup = 0.0
    mismatches = -1
    
    try:
        # MUSA Run
        with tf.device('/device:MUSA:0'):
            x_musa = tf.constant(x_np, dtype=dtype)
            y_musa = tf.constant(y_np, dtype=dtype)
            
            musa_warmup_result = tf.greater(x_musa, y_musa)
            actual_device = musa_warmup_result.device
            
            start_time = time.perf_counter()
            for _ in range(iters):
                musa_result = tf.greater(x_musa, y_musa)
            musa_time = (time.perf_counter() - start_time) / iters * 1000 
        
        mismatches = np.sum(cpu_result.numpy() != musa_result.numpy())
        speedup = cpu_time / musa_time if musa_time > 0 else 0
        
        print(f"[Test Case] Shape: {shape}, Dtype: {dtype.name}")
        print(f"Device      : {actual_device}") 
        print(f"Mismatches  : {mismatches}")
        print(f"CPU Time    : {cpu_time:.3f} ms")
        print(f"MUSA Time   : {musa_time:.3f} ms")
        print(f"Speedup     : {speedup:.2f}x")
        
        if mismatches == 0:
            print(f"Result      : PASS")
        else:
            print(f"Result      : FAIL")
        print("-" * 40)

    except tf.errors.NotFoundError:
        print(f"[Test Case] Shape: {shape}, Dtype: {dtype.name}")
        print(f"Result      : MUSA device not found or op not registered.")
        print("-" * 40)
    except Exception as e:
        print(f"[Test Case] Shape: {shape}, Dtype: {dtype.name}")
        print(f"Result      : CRASHED")
        print(f"Error       : {str(e)}")
        print("-" * 40)

def demonstrate_greater_functionality():
    print("\n" + "=" * 50)
    print("=== Greater Operator Functionality Demonstration ===")
    print("=" * 50)

    shape = (2, 3)
    dtype = tf.int32 

    x_np = np.array([
        [5, 10, 3],
        [7, 2, 8]
    ], dtype=dtype.as_numpy_dtype)

    y_np = np.array([
        [3, 12, 3],
        [7, 5, 6]
    ], dtype=dtype.as_numpy_dtype)

    print(f"Input X (dtype={dtype.name}):\n{x_np}")
    print(f"Input Y (dtype={dtype.name}):\n{y_np}")
    print("-" * 30)

    with tf.device('/CPU:0'):
        x_cpu = tf.constant(x_np, dtype=dtype)
        y_cpu = tf.constant(y_np, dtype=dtype)
        cpu_output = tf.greater(x_cpu, y_cpu)
    
    print(f"CPU Output (X > Y):\n{cpu_output.numpy()}")
    print(f"Executed on: {cpu_output.device}") # Optional: verify CPU device too
    print("-" * 30)

    try:
        with tf.device('/device:MUSA:0'):
            x_musa = tf.constant(x_np, dtype=dtype)
            y_musa = tf.constant(y_np, dtype=dtype)
            musa_output = tf.greater(x_musa, y_musa)
        
        print(f"MUSA Output (X > Y):\n{musa_output.numpy()}")
        print(f"Executed on: {musa_output.device}")
        
        mismatches = np.sum(cpu_output.numpy() != musa_output.numpy())
        if mismatches == 0:
            print("\nFunctional Test Result: PASS (CPU and MUSA outputs match)")
        else:
            print(f"\nFunctional Test Result: FAIL ({mismatches} mismatches between CPU and MUSA)")
            print("Expected (CPU):\n", cpu_output.numpy())
            print("Actual (MUSA):\n", musa_output.numpy())

    except tf.errors.NotFoundError:
        print("\nMUSA device not found or op not registered. Skipping MUSA functional test.")
    except Exception as e:
        print(f"\nError during MUSA functional test: {e}")
    
    print("=" * 50 + "\n")


if __name__ == "__main__":
    load_musa_plugin()
    
    musa_devices = tf.config.list_physical_devices('MUSA')
    if not musa_devices:
        print("No MUSA devices found. Performance tests will be skipped for MUSA.")
    else:
        print("Available MUSA devices:", musa_devices)
        
        demonstrate_greater_functionality()

        print("Starting Greater Op Performance Test...\n")
        test_cases = [
            ((256, 4096), tf.float32),
            ((256, 4096), tf.float16),
            ((1024, 1024), tf.int32),
            ((1024, 1024), tf.int64),
            ((4096, 4096), tf.float32), 
        ]
        
        for shape, dtype in test_cases:
            verify_greater_performance(shape, dtype)


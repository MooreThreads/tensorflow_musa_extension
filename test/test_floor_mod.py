import tensorflow as tf
import numpy as np
import time
import os

def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        try:
            tf.load_library(plugin_path)
            print("SUCCESS: MUSA plugin loaded!")
            return True
        except Exception as e:
            print(f"FAILED: Error loading plugin: {e}")
            return False
    else:
        print(f"ERROR: Plugin not found at {plugin_path}")
        return False

def verify_floormod(test_name, shape_x, shape_y, dtype):
    print(f"\n--- Testing {test_name} [{dtype.name}] ---")
    print(f"Shapes: {shape_x} % {shape_y}")
    
    if dtype == tf.bfloat16:
        np_dtype = np.float32
    else:
        np_dtype = dtype.as_numpy_dtype
    
    if dtype in [tf.int32, tf.int64]:
        x_np = np.random.randint(-10, 10, size=shape_x).astype(np_dtype)
        y_np = np.random.randint(1, 5, size=shape_y).astype(np_dtype)
        y_np = np.where(y_np == 0, 1, y_np)
    else:
        x_np = np.random.uniform(-10, 10, size=shape_x).astype(np_dtype)
        y_np = np.random.uniform(0.1, 5, size=shape_y).astype(np_dtype)
        y_np = np.where(y_np == 0, 0.1, y_np)
    
    with tf.device('/CPU:0'):
        if dtype == tf.bfloat16:
            x_cpu = tf.cast(tf.constant(x_np, dtype=tf.float32), tf.bfloat16)
            y_cpu = tf.cast(tf.constant(y_np, dtype=tf.float32), tf.bfloat16)
        else:
            x_cpu = tf.constant(x_np, dtype=dtype)
            y_cpu = tf.constant(y_np, dtype=dtype)
        
        start = time.time()
        res_cpu = tf.math.floormod(x_cpu, y_cpu)
        cpu_time = (time.time() - start) * 1000
    
    try:
        with tf.device('/device:MUSA:0'):
            if dtype == tf.bfloat16:
                x_musa = tf.cast(tf.constant(x_np, dtype=tf.float32), tf.bfloat16)
                y_musa = tf.cast(tf.constant(y_np, dtype=tf.float32), tf.bfloat16)
            else:
                x_musa = tf.constant(x_np, dtype=dtype)
                y_musa = tf.constant(y_np, dtype=dtype)
            
            _ = tf.math.floormod(x_musa, y_musa)
            
            start = time.time()
            res_musa = tf.math.floormod(x_musa, y_musa)
            musa_time = (time.time() - start) * 1000
        
        if dtype in [tf.bfloat16, tf.float16]:
            val_cpu = tf.cast(res_cpu, tf.float32).numpy()
            val_musa = tf.cast(res_musa, tf.float32).numpy()
        else:
            val_cpu = res_cpu.numpy()
            val_musa = res_musa.numpy()
        
        abs_diff = np.abs(val_cpu - val_musa)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        print(f"Max absolute diff: {max_abs_diff}")
        print(f"Mean absolute diff: {mean_abs_diff:.6f}")
        print(f"Time: CPU {cpu_time:.2f}ms | MUSA {musa_time:.2f}ms")
        
        if dtype == tf.bfloat16:
            threshold = 0.4  
        elif dtype == tf.float16:
            threshold = 0.1  
        elif dtype in [tf.float32, tf.float64]:
            threshold = 1e-4  
        else:
            threshold = 0
        
        if max_abs_diff <= threshold:
            print(f"Result: PASS ")
            return True
        else:
            print(f"Result: FAIL (Diff {max_abs_diff} > threshold {threshold})")
            return False
            
    except Exception as e:
        print(f"Result: CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_cases():
    test_cases = [
        ([4], [4], tf.float32, "Vector float32"),
        ([2, 3], [2, 3], tf.float32, "Matrix 2x3"),
        ([1, 5], [1, 5], tf.float32, "Row vector"),
        ([5, 1], [5, 1], tf.float32, "Column vector"),
        ([], [], tf.float32, "Scalar"),
        ([2048], [2048], tf.float32, "Large vector"),
    ]
    
    passed = 0
    total = 0
    
    for shape_x, shape_y, dtype, name in test_cases:
        if verify_floormod(name, shape_x, shape_y, dtype):
            passed += 1
        total += 1
    
    print(f"\n=== Summary: {passed}/{total} tests passed ===")
    return passed == total

def test_different_dtypes():
    test_cases = [
        ([4], [4], tf.float32, "Float32"),
        ([4], [4], tf.float16, "Float16"),
        ([4], [4], tf.bfloat16, "BFloat16"),
    ]
    
    passed = 0
    total = 0
    
    for shape_x, shape_y, dtype, name in test_cases:
        if verify_floormod(f"DataType {name}", shape_x, shape_y, dtype):
            passed += 1
        total += 1
    
    print(f"\n=== DataType Summary: {passed}/{total} tests passed ===")
    return passed == total

def test_edge_cases():
    print("\n=== Testing Edge Cases ===")
    
    test_cases = [
        ("Positive", tf.constant([10.0, 20.0, 30.0], dtype=tf.float32), 
         tf.constant([3.0, 4.0, 5.0], dtype=tf.float32)),
        ("Negative numerator", tf.constant([-10.0, -20.0, -30.0], dtype=tf.float32), 
         tf.constant([3.0, 4.0, 5.0], dtype=tf.float32)),
        ("Negative denominator", tf.constant([10.0, 20.0, 30.0], dtype=tf.float32), 
         tf.constant([-3.0, -4.0, -5.0], dtype=tf.float32)),
        ("Mixed signs", tf.constant([-10.0, 20.0, -30.0], dtype=tf.float32), 
         tf.constant([3.0, -4.0, 5.0], dtype=tf.float32)),
    ]
    
    print("CPU results:")
    for name, x, y in test_cases:
        with tf.device('/CPU:0'):
            cpu_results = tf.math.floormod(x, y).numpy()
            print(f"  {name}: {x.numpy()} % {y.numpy()} = {cpu_results}")
    
    try:
        with tf.device('/device:MUSA:0'):
            print("MUSA results:")
            for name, x, y in test_cases:
                musa_results = tf.math.floormod(x, y).numpy()
                print(f"  {name}: {x.numpy()} % {y.numpy()} = {musa_results}")
                
                with tf.device('/CPU:0'):
                    cpu_results = tf.math.floormod(x, y).numpy()
                    diff = np.max(np.abs(cpu_results - musa_results))
                    if diff <= 1e-5:
                        print(f" PASS (diff: {diff})")
                    else:
                        print(f" FAIL (diff: {diff})")
                        
    except Exception as e:
        print(f"Edge case test failed: {e}")
        import traceback
        traceback.print_exc()

def run_quick_demo():
    print("\n=== Quick Demonstration ===")
    
    try:
        with tf.device('/device:MUSA:0'):
            print("Float32 examples:")
            x_float = tf.constant([10.5, 7.8, -3.2, -9.7], dtype=tf.float32)
            y_float = tf.constant([2.0, 3.0, 1.5, 2.5], dtype=tf.float32)
            result_float = tf.math.floormod(x_float, y_float)
            print(f"  {x_float.numpy()} % {y_float.numpy()} = {result_float.numpy()}")
                
    except Exception as e:
        print(f"Demonstration failed: {e}")
        return False
    
    return True

def run_simple_test():
    print("\n=== Running Simple Tests ===")
    
    simple_cases = [
        ([3], [3], tf.float32, "Simple vector"),
        ([2, 2], [2, 2], tf.float32, "Simple matrix"),
        ([1], [1], tf.float32, "Scalar"),
    ]
    
    passed = 0
    total = 0
    
    for shape_x, shape_y, dtype, name in simple_cases:
        if verify_floormod(f"Simple {name}", shape_x, shape_y, dtype):
            passed += 1
        total += 1
    
    print(f"\n=== Simple Test Summary: {passed}/{total} tests passed ===")
    return passed == total

if __name__ == "__main__":
    if not load_musa_plugin():
        exit(1)
    
    musa_devices = tf.config.list_physical_devices('MUSA')
    print(f"\nFound MUSA Devices: {musa_devices}")
    
    if not musa_devices:
        print("No MUSA devices found.")
        exit(1)

    tf.config.set_soft_device_placement(True)
    
    print("\n" + "="*50)
    print("Starting FloorMod Operator Tests ")
    print("="*50)
    
    all_passed = True
    
    print("\n>>> Running simple tests:")
    if not run_simple_test():
        all_passed = False
    
    print("\n>>> Testing basic cases:")
    if not test_basic_cases():
        all_passed = False
    
    print("\n>>> Testing different data types:")
    if not test_different_dtypes():
        all_passed = False
    
    test_edge_cases()
    
    print("\n>>> Running quick demo:")
    if not run_quick_demo():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED")
    print("="*50)

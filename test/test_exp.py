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

def verify_exp(test_name, shape, dtype, tolerance=1e-4):
    print(f"\n--- Testing {test_name} [{dtype.name}, shape={shape}] ---")
    
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    if dtype in [tf.float32, tf.float64, tf.float16, tf.bfloat16]:
        x_np = np.random.uniform(-3, 3, size=shape).astype(np_dtype)
    else:
        x_np = np.random.randint(-5, 5, size=shape).astype(np_dtype)
    
    with tf.device('/CPU:0'):
        x_cpu = tf.constant(x_np, dtype=dtype)
        start = time.time()
        res_cpu = tf.exp(x_cpu)
        cpu_time = (time.time() - start) * 1000
    
    try:
        with tf.device('/device:MUSA:0'):
            x_musa = tf.constant(x_np, dtype=dtype)
            
            _ = tf.exp(x_musa)
            
            start = time.time()
            res_musa = tf.exp(x_musa)
            musa_time = (time.time() - start) * 1000
        
        val_cpu = tf.cast(res_cpu, tf.float32).numpy()
        val_musa = tf.cast(res_musa, tf.float32).numpy()
        
        abs_diff = np.abs(val_cpu - val_musa)
        rel_diff = np.abs((val_cpu - val_musa) / (val_cpu + 1e-10))
        
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        print(f"Max absolute diff: {max_abs_diff:.6e}")
        print(f"Max relative diff: {max_rel_diff:.6e}")
        print(f"Mean absolute diff: {mean_abs_diff:.6e}")
        print(f"Time: CPU {cpu_time:.2f}ms | MUSA {musa_time:.2f}ms")
        
        if dtype in [tf.float32, tf.float64]:
            threshold = 1e-5
        elif dtype in [tf.float16, tf.bfloat16]:
            threshold = 1e-2
        else:
            threshold = 1e-1
        
        if max_abs_diff < threshold:
            print("Result: PASS")
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
        ([4], tf.float32, "Vector float32"),
        ([2, 3], tf.float32, "Matrix 2x3"),
        ([1, 5], tf.float32, "Row vector"),
        ([5, 1], tf.float32, "Column vector"),
        ([], tf.float32, "Scalar"),
        ([4, 4], tf.float16, "Matrix float16"),
        ([3, 3], tf.bfloat16, "Matrix bfloat16"),
        ([1024, 1024], tf.float32, "Large matrix 1024x1024"),
        ([2048], tf.float32, "Large vector"),
    ]
    
    passed = 0
    total = 0
    
    for shape, dtype, name in test_cases:
        if verify_exp(name, shape, dtype):
            passed += 1
        total += 1
    
    print(f"\n=== Summary: {passed}/{total} tests passed ===")
    return passed == total

def test_edge_cases():
    print("\n=== Testing Edge Cases ===")
    
    with tf.device('/CPU:0'):
        test_values = tf.constant([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=tf.float32)
        cpu_results = tf.exp(test_values).numpy()
        print("CPU Exp results:", cpu_results)
    
    try:
        with tf.device('/device:MUSA:0'):
            musa_results = tf.exp(test_values).numpy()
            print("MUSA Exp results:", musa_results)
            
            key_points = {
                "exp(0) should be 1": musa_results[2],
                "exp(1) should be ~2.718": musa_results[3],
                "exp(-1) should be ~0.368": musa_results[1],
            }
            
            for desc, value in key_points.items():
                expected = {"exp(0)": 1.0, "exp(1)": 2.71828, "exp(-1)": 0.36788}[desc.split(" ")[0]]
                if abs(value - expected) < 0.01:
                    print(f"PASS {desc}: {value:.5f} ¡Ö {expected}")
                else:
                    print(f"FAIL {desc}: {value:.5f} ¡Ù {expected}")
                    
    except Exception as e:
        print(f"Edge case test failed: {e}")

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
    print("Starting Exp Operator Tests")
    print("="*50)
    
    all_passed = True
    
    if not test_basic_cases():
        all_passed = False
    
    test_edge_cases()
    
    print("\n" + "="*50)
    print("Demonstration")
    print("="*50)
    
    try:
        with tf.device('/device:MUSA:0'):
            demos = [
                ("Positive", [1.0, 2.0, 3.0]),
                ("Negative", [-1.0, -2.0, -3.0]),
                ("Zero", [0.0, 0.0, 0.0]),
                ("Mixed", [-2.0, 0.0, 2.0]),
            ]
            
            for name, values in demos:
                x = tf.constant(values, dtype=tf.float32)
                y = tf.exp(x)
                print(f"{name}: exp{values} = {y.numpy()}")
                
    except Exception as e:
        print(f"Demonstration failed: {e}")
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED")
    print("="*50)

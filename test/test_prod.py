import tensorflow as tf
import os
import numpy as np

so_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"

print(f"--- 1. Loading Plugin from: {so_path} ---")
try:
    if os.path.exists(so_path):
        _plugin = tf.load_op_library(so_path)
        print("Plugin loaded successfully")
    else:
        print(f"Warning: Plugin file not found at {so_path}")
except Exception as e:
    print(f"Load failed: {e}")
    
print("\n--- 2. Checking MUSA Device ---")
musa_devices = tf.config.list_physical_devices('MUSA')
print(f"Found MUSA Devices: {musa_devices}")

print("\n--- 3. Testing Prod Op (ReduceProd) ---")

tf.config.set_soft_device_placement(True)

def test_prod_case(input_data, axis=None, keepdims=False, dtype=tf.float32, case_name=""):
    print(f"\n--- Test Case: {case_name} ---")
    try:
        if isinstance(input_data, list):
            input_np = np.array(input_data)
        else:
            input_np = input_data
            
        with tf.device('/device:MUSA:0'):
            x = tf.constant(input_np, dtype=dtype)
            print(f"Input shape: {x.shape}, dtype: {dtype}")
            
            with tf.device('/device:CPU:0'):
                expected = tf.reduce_prod(x, axis=axis, keepdims=keepdims)
                expected_val = expected.numpy()
            
            if axis is None:
                z = tf.reduce_prod(x, keepdims=keepdims)
            else:
                z = tf.reduce_prod(x, axis=axis, keepdims=keepdims)
            
            print(f"Prod result: {z.numpy()}")
            print(f"Result shape: {z.shape}")
            print(f"Result device: {z.device}")
            
            actual = z.numpy()
            
            if dtype.is_floating:
                diff = np.abs(actual - expected_val)
                is_close = np.allclose(actual, expected_val, rtol=1e-4, atol=1e-5)
            else:
                is_close = np.all(actual == expected_val)
            
            if is_close:
                print(f" Prod Test Passed")
                return True
            else:
                print(f" Prod Test Failed")
                print(f"  Expected: {expected_val}")
                print(f"  Got: {actual}")
                print(f"  Diff: {np.abs(actual - expected_val)}")
                return False
                
    except Exception as e:
        print(f"Prod Test Failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

print("\n>>> 4.1 Global Prod (axis=None)")
test_prod_case(
    input_data=[1.0, 2.0, 3.0, 4.0],
    axis=None,
    case_name="Global Prod 1D"
)

print("\n>>> 4.2 Prod along axis=0")
test_prod_case(
    input_data=[[2.0, 3.0], [4.0, 5.0]],
    axis=0,
    case_name="Prod axis=0"
)

print("\n>>> 4.3 Prod along axis=1")
test_prod_case(
    input_data=[[2.0, 3.0], [4.0, 5.0]],
    axis=1,
    case_name="Prod axis=1"
)

print("\n>>> 4.4 Prod with keepdims=True")
test_prod_case(
    input_data=[[2.0, 3.0], [4.0, 5.0]],
    axis=1,
    keepdims=True,
    case_name="Prod keepdims=True"
)

print("\n>>> 4.5 Prod with negative axis")
test_prod_case(
    input_data=[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
    axis=-1,
    case_name="Prod negative axis"
)

print("\n>>> 4.6 Prod along multiple axes")
test_prod_case(
    input_data=[[[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [0.5, 2.0]]],
    axis=[0, 2],
    case_name="Prod multiple axes"
)

print("\n>>> 4.7 Prod with int32 dtype")
test_prod_case(
    input_data=[[2, 3], [4, 5]],
    axis=0,
    dtype=tf.int32,
    case_name="Prod int32"
)

print("\n>>> 4.8 Prod with float16 dtype")
test_prod_case(
    input_data=[[2.0, 0.5], [4.0, 0.25]],
    axis=0,
    dtype=tf.float16,
    case_name="Prod float16"
)

print("\n>>> 4.9 Prod with Zeros")
test_prod_case(
    input_data=[[1.0, 2.0, 0.0], [4.0, 5.0, 6.0]],
    axis=1,
    case_name="Prod with Zero"
)

print("\n>>> 4.10 Prod with Negative Numbers")
test_prod_case(
    input_data=[[-1.0, 2.0], [-3.0, -4.0]], # [-2, 12]
    axis=1,
    case_name="Prod with Negatives"
)

print("\n--- 5. Testing Edge Cases ---")

print("\n>>> 5.1 Empty axis list")
test_prod_case(
    input_data=[[2.0, 2.0], [3.0, 3.0]],
    axis=[], 
    case_name="Empty axis list (Global)"
)

print("\n>>> 5.2 Large tensor (Ones)")
try:
    with tf.device('/device:MUSA:0'):
        large_tensor = tf.ones([100, 100], dtype=tf.float32)
        result = tf.reduce_prod(large_tensor)
        expected = 1.0
        
        if abs(result.numpy() - expected) < 1e-5:
            print(f" Large tensor prod correct: {result.numpy()}")
        else:
            print(f" Large tensor prod incorrect: {result.numpy()} (expected {expected})")
            
except Exception as e:
    print(f"Large tensor test failed: {e}")

print("\n>>> 5.3 Single element tensor")
test_prod_case(
    input_data=[5.0],
    axis=0,
    case_name="Single element"
)

print("\n--- 6. Summary ---")
print("Prod (ReduceProd) operator testing completed.")


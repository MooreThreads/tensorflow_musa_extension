import tensorflow as tf
import numpy as np
import os

so_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"

def load_plugin():
    print(f"--- Loading Plugin from: {so_path} ---")
    try:
        tf.load_op_library(so_path)
        print(">> Plugin loaded successfully")
    except Exception as e:
        print(f">> Load failed (may be already loaded): {e}")

def check_musa_device():
    devices = tf.config.list_physical_devices('MUSA')
    if not devices:
        print("!! No MUSA devices found. Exiting.")
        return False
    print(f">> Found MUSA Devices: {devices}")
    return True

def run_test_dtype(dtype_name, tf_dtype, np_dtype, tolerance=1e-6):
    print(f"\n[Testing Data Type: {dtype_name}]")
    try:
        with tf.device('/device:MUSA:0'):
            cond_np = np.array([True, False, True, False], dtype=np.bool_)
            x_np = np.array([1, 2, 3, 4], dtype=np_dtype)
            y_np = np.array([10, 20, 30, 40], dtype=np_dtype)
            
            cond = tf.constant(cond_np)
            x = tf.constant(x_np, dtype=tf_dtype)
            y = tf.constant(y_np, dtype=tf_dtype)
            
            z = tf.where(cond, x, y)
            
            expected = np.where(cond_np, x_np, y_np)
            got = z.numpy()
            
            print(f"  Input X: {x_np}")
            print(f"  Input Y: {y_np}")
            print(f"  Cond:    {cond_np}")
            print(f"  Result:  {got}")
            
            if np.allclose(got, expected, atol=tolerance):
                print(f"   PASS {dtype_name}")
            else:
                print(f"   FAIL {dtype_name}")
                print(f"    Expected: {expected}")
                print(f"    Got:      {got}")
                
    except Exception as e:
        print(f"  !! Error testing {dtype_name}: {e}")
        import traceback
        traceback.print_exc()

def run_broadcast_test():
    print("\n[Testing Broadcasting]")
    try:
        with tf.device('/device:MUSA:0'):
            cond_np = np.array([False, True])
            x_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
            y_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
            
            cond = tf.constant(cond_np)
            x = tf.constant(x_np)
            y = tf.constant(y_np)
            
            z = tf.where(cond, x, y)
            
            expected = np.where(cond_np, x_np, y_np)
            got = z.numpy()
            
            print(f"  Broadcast Result:\n{got}")
            if np.allclose(got, expected):
                print("   PASS Broadcasting")
            else:
                print("   FAIL Broadcasting")
                print(f"    Expected:\n{expected}")

    except Exception as e:
        print(f"  !! Error testing Broadcasting: {e}")

def main():
    load_plugin()
    if not check_musa_device():
        return

    print("="*60)
    print("MUSA SelectV2 (tf.where) Multi-Type Test")
    print("="*60)

    run_test_dtype("Float32", tf.float32, np.float32)
    run_test_dtype("Int32", tf.int32, np.int32)
    run_test_dtype("Int64", tf.int64, np.int64)
    run_test_dtype("Float16", tf.float16, np.float16, tolerance=1e-3)

    print("\n[Testing Data Type: Bool]")
    try:
        with tf.device('/device:MUSA:0'):
            cond = tf.constant([True, False, True])
            x = tf.constant([True, True, True])
            y = tf.constant([False, False, False])
            z = tf.where(cond, x, y)
            got = z.numpy()
            expected = np.array([True, False, True])
            
            print(f"  Result: {got}")
            if np.array_equal(got, expected):
                print("   PASS Bool")
            else:
                print("   FAIL Bool")
    except Exception as e:
        print(f"  !! Error: {e}")

    run_broadcast_test()

if __name__ == "__main__":
    main()


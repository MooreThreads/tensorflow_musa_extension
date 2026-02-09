import tensorflow as tf
import numpy as np
import os
import math

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

def ref_erf(x):
    vec_erf = np.vectorize(math.erf)
    return vec_erf(x)

def test_dtype(dtype, dtype_name, size=10, tolerance=1e-5):
    print(f"\n[Testing {dtype_name}] Size={size} ...")
    
    try:
        with tf.device('/device:MUSA:0'):
            np_input = np.random.uniform(-3.0, 3.0, size).astype(
                np.float16 if dtype == tf.float16 else 
                np.float64 if dtype == tf.float64 else 
                np.float32
            )
            
            x = tf.constant(np_input, dtype=dtype)
            
            y = tf.math.erf(x)
            
            y_musa = y.numpy()
            
            y_ref = ref_erf(np_input.astype(np.float64))
            
            diff = np.abs(y_musa - y_ref)
            max_diff = np.max(diff)
            
            if np.allclose(y_musa, y_ref, atol=tolerance, rtol=tolerance):
                print(f"  PASS: Max diff = {max_diff:.2e} (within tol {tolerance})")
            else:
                print(f"  FAIL: Max diff = {max_diff:.2e} > tol {tolerance}")
                mismatch_idx = np.where(diff > tolerance)[0][:5]
                for idx in mismatch_idx:
                    print(f"    idx[{idx}]: Input={np_input[idx]:.4f} | MUSA={y_musa[idx]:.4f} | Ref={y_ref[idx]:.4f}")

    except Exception as e:
        print(f"  !! ERROR during execution: {e}")
        import traceback
        traceback.print_exc()

def main():
    load_plugin()
    if not check_musa_device():
        return

    print("="*60)
    print("MUSA Erf Kernel Verification")
    print("="*60)

    test_dtype(tf.float32, "Float32 (Small)", size=20, tolerance=1e-6)

    test_dtype(tf.float32, "Float32 (Large N=10000)", size=10000, tolerance=1e-6)

    test_dtype(tf.float16, "Float16 (Half)", size=1000, tolerance=1e-3)
    test_dtype(tf.float64, "Float64 (Double)", size=1000, tolerance=1e-14)

    print("\n[Testing Edge Cases]")
    try:
        with tf.device('/device:MUSA:0'):
            edge_vals = [0.0, float('inf'), float('-inf'), 100.0, -100.0]
            x_edge = tf.constant(edge_vals, dtype=tf.float32)
            y_edge = tf.math.erf(x_edge)
            
            print(f"  Input: {edge_vals}")
            print(f"  Output: {y_edge.numpy()}")
            
            res = y_edge.numpy()
            if res[0] == 0.0 and res[1] == 1.0 and res[2] == -1.0:
                print("  PASS: Edge cases handling correct")
            else:
                print("  FAIL: Edge cases mismatch")
    except Exception as e:
        print(f"  !! Edge case error: {e}")

if __name__ == "__main__":
    main()


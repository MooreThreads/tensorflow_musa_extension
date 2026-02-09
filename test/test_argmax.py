import tensorflow as tf
import numpy as np
import os
import traceback

def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        tf.load_library(plugin_path)
        print(">> MUSA Plugin Loaded")
        return True
    else:
        print(">> MUSA Plugin Not Found")
        return False

def run_test(input_shape, axis, dtype, output_type=tf.int64, test_name=""):
    print(f"\n--- ArgMax Test {test_name} [{dtype.name}] Shape={input_shape} Axis={axis} ---")
    try:
        np_dtype = dtype.as_numpy_dtype if dtype != tf.bfloat16 else np.float32
        
        x_np = np.random.randn(*input_shape).astype(np_dtype)
        
        with tf.device('/device:MUSA:0'):
            if dtype == tf.bfloat16:
                x = tf.cast(tf.constant(x_np), tf.bfloat16)
            else:
                x = tf.constant(x_np, dtype=dtype)
            
            res_musa = tf.math.argmax(x, axis=axis, output_type=output_type)
            
        res_expected = np.argmax(x_np, axis=axis).astype(output_type.as_numpy_dtype)
        
        res_val = res_musa.numpy()
        if np.array_equal(res_val, res_expected):
            print(f"[PASS] {test_name}")
        else:
            print(f"[FAIL] {test_name}")
            print(f"  Shape MUSA: {res_val.shape}, Exp: {res_expected.shape}")
            
            mismatch_idx = np.where(res_val != res_expected)
            print(f"  Mismatch count: {len(mismatch_idx[0])}")
            if len(mismatch_idx[0]) > 0:
                idx0 = mismatch_idx[0][0]

                flat_musa = res_val.flatten()
                flat_exp = res_expected.flatten()
                flat_diff = np.where(flat_musa != flat_exp)[0]
                if len(flat_diff) > 0:
                    i = flat_diff[0]
                    print(f"  First Diff at flat index {i}: MUSA={flat_musa[i]}, EXP={flat_exp[i]}")

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()

def main():
    if not load_musa_plugin(): 
        print("Load plugin failed.")
        return
    print("MUSA Devices:", tf.config.list_physical_devices('MUSA'))

    # 1. Float32 Tests
    run_test((10,), 0, tf.float32, test_name="1D_Float32")
    run_test((5, 5), 0, tf.float32, test_name="2D_Axis0_Float32")
    run_test((5, 5), 1, tf.float32, test_name="2D_Axis1_Float32")
    run_test((2, 3, 4), 2, tf.float32, test_name="3D_Axis2_Float32")
    
    # 2. Int32 Tests
    run_test((10, 10), 1, tf.int32, test_name="2D_Int32")
    
    # 3. Output Type Int32
    run_test((10, 10), 1, tf.float32, output_type=tf.int32, test_name="OutInt32")

    # 4. FP16 / BF16
    run_test((10, 10), 1, tf.float16, test_name="FP16")
    try:
        run_test((10, 10), 1, tf.bfloat16, test_name="BF16")
    except:
        pass

if __name__ == "__main__":
    main()

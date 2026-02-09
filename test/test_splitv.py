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

def run_test_case(name, input_data, size_splits, axis, dtype):
    print(f"\n[Test Case: {name}]")
    print(f"  Shape: {input_data.shape}, Axis: {axis}, Splits: {size_splits}, Dtype: {dtype}")
    
    try:
        with tf.device('/device:MUSA:0'):
            x = tf.constant(input_data, dtype=dtype)
            
            results = tf.split(x, num_or_size_splits=size_splits, axis=axis)
            
            np_res = np.split(input_data, np.cumsum(size_splits)[:-1], axis=axis)
            
            if len(results) != len(np_res):
                print(f"   FAIL: Output count mismatch. Expected {len(np_res)}, got {len(results)}")
                return

            all_pass = True
            for i, (musa_tensor, expected) in enumerate(zip(results, np_res)):
                musa_val = musa_tensor.numpy()
                if not np.allclose(musa_val, expected):
                    print(f"   FAIL at split index {i}")
                    print(f"    Expected:\n{expected}")
                    print(f"    Got:\n{musa_val}")
                    all_pass = False
                else:
                    pass # print(f"    Split {i} shape: {musa_val.shape} OK")
            
            if all_pass:
                print("   PASS")

    except Exception as e:
        print(f"  !! ERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    load_plugin()
    if not check_musa_device():
        return

    print("="*60)
    print("MUSA SplitV Operator Test")
    print("="*60)

    data_1d = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    run_test_case("1D Int32", data_1d, [3, 2, 1], 0, tf.int32)

    data_2d = np.arange(10).reshape(2, 5).astype(np.float32)
    run_test_case("2D Float32 Axis=1", data_2d, [2, 1, 2], 1, tf.float32)

    data_2d_v2 = np.arange(8).reshape(4, 2).astype(np.float32)
    run_test_case("2D Float32 Axis=0", data_2d_v2, [1, 3], 0, tf.float32)

    data_empty = np.array([1, 2, 3], dtype=np.float32)
    run_test_case("Empty Split", data_empty, [0, 2, 1, 0], 0, tf.float32)

    data_neg = np.random.randn(2, 3, 4).astype(np.float32)
    run_test_case("Negative Axis (-1)", data_neg, [1, 1, 2], -1, tf.float32)
    
    data_f16 = np.random.randn(4, 4).astype(np.float16)
    run_test_case("Float16 (Half)", data_f16, [2, 2], 0, tf.float16)

if __name__ == "__main__":
    main()


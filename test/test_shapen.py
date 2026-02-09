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

def run_test():
    print("\n--- ShapeN Test ---")
    try:
        shape1 = (2, 3)       # float32
        shape2 = (5, 1, 4)    # int32
        shape3 = (10,)        # float32
        
        val1 = np.random.randn(*shape1).astype(np.float32)
        val2 = np.random.randn(*shape2).astype(np.int32)
        val3 = np.random.randn(*shape3).astype(np.float32)

        with tf.device('/device:MUSA:0'):
            t1 = tf.constant(val1)
            t2 = tf.constant(val2)
            t3 = tf.constant(val3)
            

            print(">> Testing float32 group...")
            shapes_float = tf.shape_n([t1, t3], out_type=tf.int32)
            
            print(">> Testing int32 group...")
            shapes_int = tf.shape_n([t2], out_type=tf.int32)

            print(">> Testing int64 output...")
            shapes_64 = tf.shape_n([t1], out_type=tf.int64)

        res_float = [s.numpy() for s in shapes_float]
        if np.array_equal(res_float[0], shape1) and np.array_equal(res_float[1], shape3):
            print("  [PASS] Float32 group shapes correct")
        else:
            print(f"  [FAIL] Float32 shapes: {res_float}")

        res_int = [s.numpy() for s in shapes_int]
        if np.array_equal(res_int[0], shape2):
            print("  [PASS] Int32 group shapes correct")
        else:
            print(f"  [FAIL] Int32 shapes: {res_int}")
            
        res_64 = shapes_64[0].numpy()
        if np.array_equal(res_64, shape1) and (res_64.dtype == np.int64 or res_64.dtype == np.longlong):
             print("  [PASS] Output type int64 correct")
        else:
             print(f"  [FAIL] Int64 output check: {res_64.dtype} / {res_64}")

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    load_musa_plugin()
    if tf.config.list_physical_devices('MUSA'):
        run_test()
    else:
        print("No MUSA devices found.")

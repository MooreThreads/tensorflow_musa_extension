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
    print(f"\n--- ConcatOffset Test ---")
    try:
        concat_dim = 1
        shape0 = np.array([2, 3], dtype=np.int32)
        shape1 = np.array([2, 5], dtype=np.int32)
        
        with tf.device('/device:MUSA:0'):
            offsets = tf.raw_ops.ConcatOffset(
                concat_dim=tf.constant(concat_dim),
                shape=[tf.constant(shape0), tf.constant(shape1)]
            )
            print(f"  > Output 0 Device: {offsets[0].device}")

        off0 = offsets[0].numpy()
        off1 = offsets[1].numpy()
        
        print(f"  Offset 0: {off0}")
        print(f"  Offset 1: {off1}")
        
        expected0 = np.array([0, 0], dtype=np.int32)
        expected1 = np.array([0, 3], dtype=np.int32)
        
        if np.array_equal(off0, expected0) and np.array_equal(off1, expected1):
            print("[PASS] ConcatOffset Check")
        else:
            print("[FAIL] Result Mismatch")
            
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    load_musa_plugin()
    print("MUSA Devices:", tf.config.list_physical_devices('MUSA'))
    
    if tf.config.list_physical_devices('MUSA'):
        run_test()

import tensorflow as tf
import ctypes
import os
import numpy as np

def main():
    print("Testing Neg operator on MUSA")
    print("=" * 50)
    
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    print(f"Loading MUSA plugin from: {plugin_path}")
    
    try:
        ctypes.CDLL(plugin_path)
        print("Plugin loaded successfully")
    except Exception as e:
        print(f"Failed to load plugin: {e}")
        return
    
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/musa/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['MUSA_VISIBLE_DEVICES'] = '0'
    
    print("\nListing MUSA devices:")
    musa_devices = tf.config.list_physical_devices('MUSA')
    for device in musa_devices:
        print(f"  {device}")
    
    if not musa_devices:
        print("No MUSA devices found. Aborting.")
        return
    
    def run_test(input_arr, dtype, test_name):
        print(f"\n>>> Test Case: {test_name}")
        try:
            with tf.device('/device:MUSA:0'):
                x = tf.constant(input_arr, dtype=dtype)
                
                y = tf.math.negative(x)
                
                res = y.numpy()
                expected = -np.array(input_arr)
                
                print(f"Input:    {x.numpy()}")
                print(f"Output:   {res}")
                print(f"Expected: {expected}")
                
                if np.allclose(res, expected, rtol=1e-5):
                    print(f"[SUCCESS] {test_name} passed!")
                else:
                    print(f"[FAILED] {test_name} failed mismatch.")
                    
        except Exception as e:
            print(f"[ERROR] {test_name} execution failed: {e}")
            import traceback
            traceback.print_exc()

    run_test([-2.5, -1.0, 0.0, 1.0, 2.5], tf.float32, "Float32 Basic")
    
    run_test([-10, -5, 0, 5, 10], tf.int32, "Int32 Basic")

if __name__ == "__main__":
    main()


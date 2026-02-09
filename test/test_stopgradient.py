import tensorflow as tf
import ctypes
import os
import numpy as np

plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
ctypes.CDLL(plugin_path)

print("Quick test of StopGradient operator on MUSA")
print("=" * 40)

os.environ['LD_LIBRARY_PATH'] = '/usr/local/musa/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

with tf.device('/device:MUSA:0'):
    x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
    y = tf.stop_gradient(x)
    
    print(f"Input: {x.numpy()}")
    print(f"Output: {y.numpy()}")
    print(f"Input and output are equal: {np.array_equal(x.numpy(), y.numpy())}")
    
    x_int = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    y_int = tf.stop_gradient(x_int)
    
    print(f"\nInteger input: {x_int.numpy()}")
    print(f"Integer output: {y_int.numpy()}")
    
    print("\nTesting gradient blocking...")
    
    var = tf.Variable(3.0, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(var)
        result = var * 2.0
        stopped = tf.stop_gradient(result)
        final = stopped + 1.0
    
    grad = tape.gradient(final, var)
    
    print(f"Variable value: {var.numpy()}")
    print(f"Final result: {final.numpy()}")
    print(f"Gradient w.r.t variable: {grad}")
    
    if grad is None or grad == 0.0:
        print("Gradient correctly blocked by StopGradient")
        print("\nStopGradient test PASSED!")
    else:
        print("Gradient not properly blocked")
        print("\nStopGradient test FAILED")

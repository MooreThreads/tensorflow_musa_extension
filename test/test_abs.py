import tensorflow as tf
import numpy as np
import os

plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
if os.path.exists(plugin_path):
    tf.load_library(plugin_path)
    print("MUSA plugin loaded successfully")
else:
    print("Plugin not found")

devices = tf.config.list_physical_devices('MUSA')
if not devices:
    print("No MUSA device detected")
    exit(1)
print(f"Detected MUSA device: {devices}")

data = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)

with tf.device('/CPU:0'):
    cpu_result = tf.abs(data).numpy()

try:
    with tf.device('/device:MUSA:0'):
        musa_result = tf.abs(data).numpy()
    
    if np.allclose(cpu_result, musa_result):
        print("Abs test passed!")
        print(f"Input: {data}")
        print(f"CPU result: {cpu_result}")
        print(f"MUSA result: {musa_result}")
    else:
        print("Results don't match")
        
except Exception as e:
    print(f"Error: {e}")

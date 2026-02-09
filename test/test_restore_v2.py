from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test

def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        try:
            tf.load_library(plugin_path)
            print(f"Successfully loaded MUSA plugin from {plugin_path}")
        except Exception as e:
            print(f"Failed to load MUSA plugin: {e}")
    else:
        print(f"MUSA plugin not found at {plugin_path}")

class MusaRestoreV2Test(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(MusaRestoreV2Test, cls).setUpClass()
    load_musa_plugin()

  def testRestoreV2Musa(self):
    if not tf.config.list_physical_devices('MUSA'):
        print("Skipping test: No MUSA device found")
        return

    save_path = os.path.join(self.get_temp_dir(), "musa_ckpt")
    
    original_data_float = np.random.rand(5, 5).astype(np.float32)
    original_data_int = np.random.randint(0, 100, size=(10,)).astype(np.int32)

    with tf.device('/CPU:0'):
        io_ops.save_v2(
            save_path, 
            ["tensor_float", "tensor_int"], 
            ["", ""], 
            [constant_op.constant(original_data_float), constant_op.constant(original_data_int)]
        )
    
    print(f"Checkpoint saved to {save_path}")

    with tf.device('/device:MUSA:0'):
        restored_tensors = io_ops.restore_v2(
            save_path, 
            ["tensor_float", "tensor_int"], 
            ["", ""], 
            [dtypes.float32, dtypes.int32]
        )

    restored_float = self.evaluate(restored_tensors[0])
    restored_int = self.evaluate(restored_tensors[1])

    self.assertAllClose(original_data_float, restored_float)
    self.assertAllEqual(original_data_int, restored_int)
    print("MUSA RestoreV2 test passed: Data matches original.")

  def testRestoreV2MusaWithSlice(self):
    if not tf.config.list_physical_devices('MUSA'):
        return

    save_path = os.path.join(self.get_temp_dir(), "musa_ckpt_slice")
    
    full_shape = [10, 10]
    original_data = np.random.rand(*full_shape).astype(np.float32)
    
    with tf.device('/CPU:0'):
        io_ops.save_v2(
            save_path, 
            ["tensor_slice"], 
            ["10 10 0,10:0,10"], 
            [constant_op.constant(original_data)]
        )

    slice_spec = "10 10 0,5:0,5" 
    expected_data = original_data[0:5, 0:5]

    with tf.device('/device:MUSA:0'):
        restored_slice = io_ops.restore_v2(
            save_path, 
            ["tensor_slice"], 
            [slice_spec], 
            [dtypes.float32]
        )

    restored_val = self.evaluate(restored_slice[0])
    
    self.assertAllClose(expected_data, restored_val)
    print("MUSA RestoreV2 Slice test passed.")

if __name__ == "__main__":
  test.main()

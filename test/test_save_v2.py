from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np

from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test
import tensorflow as tf

def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        tf.load_library(plugin_path)

class MusaSaveV2Test(test.TestCase):

    @classmethod
    def setUpClass(cls):
        load_musa_plugin()
        cls.musa_available = len(tf.config.list_physical_devices('MUSA')) > 0

    def setUp(self):
        super(MusaSaveV2Test, self).setUp()
        self.test_dir = self.get_temp_dir()
        os.chdir(self.test_dir)
        self.prefix = os.path.join(self.test_dir, "ckpt_save_musa")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        super(MusaSaveV2Test, self).tearDown()

    def testSaveV2Musa(self):
        if not self.musa_available:
            self.skipTest("MUSA device not available")

        print("\n--- Testing SaveV2 on MUSA ---")
        
        data_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        with tf.device('/device:MUSA:0'):
            t_in = tf.constant(data_np)
            io_ops.save_v2(
                self.prefix, 
                ["var_save"], 
                [""], 
                [t_in]
            )
        
        self.assertTrue(os.path.exists(self.prefix + ".index"))
        self.assertTrue(os.path.exists(self.prefix + ".data-00000-of-00001"))
        print("SaveV2 executed successfully on MUSA.")

if __name__ == "__main__":
    test.main()

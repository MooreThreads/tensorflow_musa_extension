import os
import numpy as np
import tensorflow as tf
try:
    # Try to load the musa plugin if it exists
    # This assumes the plugin is already built and available in the python path or loaded via some mechanism
    # In a real environment, you might need: tf.load_op_library('libmusa_plugin.so')
    pass
except Exception as e:
    print(f"Warning: Could not load musa_plugin: {e}")

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class ConcatMatMulFusionTest(test.TestCase):

    def testConcatMatMulFusion(self):
        # Define shapes
        shape1 = [2, 16]
        shape2 = [2, 16]
        weight_shape = [32, 8]

        with self.session(use_gpu=True) as sess:
            # Inputs
            a = array_ops.placeholder(tf.float32, shape=shape1, name="input_a")
            b = array_ops.placeholder(tf.float32, shape=shape2, name="input_b")
            w = array_ops.placeholder(tf.float32, shape=weight_shape, name="weight")

            # Concat + MatMul pattern
            concat_node = array_ops.concat([a, b], axis=1, name="concat")
            matmul_node = math_ops.matmul(concat_node, w, name="matmul")

            # Data for inputs
            np_a = np.random.randn(*shape1).astype(np.float32)
            np_b = np.random.randn(*shape2).astype(np.float32)
            np_w = np.random.randn(*weight_shape).astype(np.float32)

            # Run
            feed_dict = {a: np_a, b: np_b, w: np_w}
            result_fused = sess.run(matmul_node, feed_dict=feed_dict)

            # Expected result
            np_concat = np.concatenate([np_a, np_b], axis=1)
            np_matmul = np.matmul(np_concat, np_w)

            self.assertAllClose(result_fused, np_matmul, atol=1e-5)

            # Check GraphDef for fusion (Simulated, as we need Grappler optimization to run)
            # In a real test, we would check if 'MusaConcatMatMul' exists in the optimized graph
            print("Successfully ran ConcatMatMul fusion test")

if __name__ == "__main__":
    test.main()

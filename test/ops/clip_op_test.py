import os
import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ClipOpTest(MUSATestCase):
    """Unit tests for custom MUSA Clip operator."""

    @classmethod
    def setUpClass(cls):
        """Load the custom op module so MusaClip is callable from Python."""
        super(ClipOpTest, cls).setUpClass()

        plugin_path = None
        current_dir = os.path.dirname(os.path.abspath(__file__))

        candidate_paths = [
            os.path.join(current_dir, "..", "..", "build", "libmusa_plugin.so"),
            os.path.join(os.path.dirname(current_dir), "..", "build", "libmusa_plugin.so"),
            os.path.join(os.getcwd(), "..", "build", "libmusa_plugin.so"),
        ]

        for path in candidate_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                plugin_path = normalized_path
                break

        if plugin_path and os.path.exists(plugin_path):
            try:
                cls._musa_ops = tf.load_op_library(plugin_path)
            except Exception as e:
                print(f"FAILED: Error loading MUSA ops from {plugin_path}: {e}")
                cls._musa_ops = None
        else:
            searched_locations = [os.path.normpath(path) for path in candidate_paths]
            print("MUSA plugin not found. Searched locations:\n" +
                  "\n".join(f"  - {loc}" for loc in searched_locations))
            cls._musa_ops = None

    def _run_reference_cpu(self, x, lo, hi):
        """Reference path on CPU using TensorFlow native clip_by_value."""
        with tf.device("/CPU:0"):
            return tf.clip_by_value(x, lo, hi)

    def _run_musa_clip(self, x, lo, hi):
        """Run custom MusaClip op on MUSA device."""
        if self._musa_ops is None or not hasattr(self._musa_ops, "musa_clip"):
            self.skipTest(
                "MUSA clip op module is not available. "
                "Make sure REGISTER_OP(\"MusaClip\") is compiled and the plugin is loaded."
            )

        with tf.device("/device:MUSA:0"):
            return self._musa_ops.musa_clip(x=x, lo=lo, hi=hi)

    def _assert_clip_close(self, x_np, lo_np, hi_np, dtype, rtol=1e-5, atol=1e-8):
        """Compare CPU reference vs custom MUSA clip op."""
        np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

        x = tf.constant(np.array(x_np, dtype=np_dtype), dtype=dtype)
        lo = tf.constant(np.array(lo_np, dtype=np_dtype), dtype=dtype)
        hi = tf.constant(np.array(hi_np, dtype=np_dtype), dtype=dtype)

        cpu_result = self._run_reference_cpu(x, lo, hi)
        musa_result = self._run_musa_clip(x, lo, hi)

        if dtype in [tf.float16, tf.bfloat16]:
            cpu_result = tf.cast(cpu_result, tf.float32)
            musa_result = tf.cast(musa_result, tf.float32)

        self.assertAllClose(cpu_result.numpy(), musa_result.numpy(),
                            rtol=rtol, atol=atol)

    def testClipBasicFloat32(self):
        x_np = np.array(
            [[-2.0, -0.5, 0.2],
             [1.5, 6.2, 9.0]],
            dtype=np.float32
        )
        self._assert_clip_close(x_np=x_np, lo_np=0.0, hi_np=6.0,
                                dtype=tf.float32, rtol=1e-5, atol=1e-8)

    def testClipBasicFloat16(self):
        x_np = np.array(
            [[-2.0, -0.5, 0.2],
             [1.5, 6.2, 9.0]],
            dtype=np.float16
        )
        self._assert_clip_close(x_np=x_np, lo_np=np.float16(0.0), hi_np=np.float16(6.0),
                                dtype=tf.float16, rtol=1e-2, atol=1e-2)

    def testClipBasicBFloat16(self):
        x_np = np.array(
            [[-2.0, -0.5, 0.2],
             [1.5, 6.2, 9.0]],
            dtype=np.float32
        )
        self._assert_clip_close(x_np=x_np, lo_np=0.0, hi_np=6.0,
                                dtype=tf.bfloat16, rtol=2e-2, atol=2e-2)

    def testClipBasicInt32(self):
        x_np = np.array(
            [[-2, -1, 0],
             [3, 7, 9]],
            dtype=np.int32
        )
        self._assert_clip_close(x_np=x_np, lo_np=np.int32(0), hi_np=np.int32(6),
                                dtype=tf.int32, rtol=0.0, atol=0.0)

    def testClipTensorBoundsSameShape(self):
        x_np = np.array(
            [[-3.0, 1.0, 8.0],
             [2.0, -4.0, 5.0]],
            dtype=np.float32
        )
        lo_np = np.array(
            [[0.0, -1.0, 2.0],
             [1.0, -3.0, 4.0]],
            dtype=np.float32
        )
        hi_np = np.array(
            [[1.0, 2.0, 6.0],
             [3.0, 0.0, 5.0]],
            dtype=np.float32
        )
        self._assert_clip_close(x_np=x_np, lo_np=lo_np, hi_np=hi_np,
                                dtype=tf.float32, rtol=1e-5, atol=1e-8)

    def testClipVectorBroadcast(self):
        x_np = np.array(
            [[-3.0, 1.0, 8.0, 10.0],
             [2.0, -4.0, 5.0, 7.0]],
            dtype=np.float32
        )
        lo_np = np.array([0.0, -1.0, 2.0, 6.0], dtype=np.float32)
        hi_np = np.array([1.0, 2.0, 6.0, 8.0], dtype=np.float32)
        self._assert_clip_close(x_np=x_np, lo_np=lo_np, hi_np=hi_np,
                                dtype=tf.float32, rtol=1e-5, atol=1e-8)

    def testClipColumnRowBroadcast(self):
        x_np = np.array(
            [[-5.0, 0.0, 3.0],
             [7.0, 2.0, 9.0]],
            dtype=np.float32
        )
        lo_np = np.array([[0.0], [1.0]], dtype=np.float32)
        hi_np = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
        self._assert_clip_close(x_np=x_np, lo_np=lo_np, hi_np=hi_np,
                                dtype=tf.float32, rtol=1e-5, atol=1e-8)

    def testClipExactBoundaryValues(self):
        x_np = np.array([-1.0, 0.0, 3.0, 6.0, 8.0], dtype=np.float32)
        self._assert_clip_close(x_np=x_np, lo_np=0.0, hi_np=6.0,
                                dtype=tf.float32, rtol=1e-5, atol=1e-8)

    def testClipLargeTensor(self):
        x_np = np.random.uniform(-10.0, 10.0, size=[256, 256]).astype(np.float32)
        self._assert_clip_close(x_np=x_np, lo_np=-1.5, hi_np=2.5,
                                dtype=tf.float32, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    tf.test.main()

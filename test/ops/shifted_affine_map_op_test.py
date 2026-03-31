import os

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ShiftedAffineMapOpTest(MUSATestCase):
    """Numerical tests for custom MusaShiftedAffineMap operator."""

    @classmethod
    def setUpClass(cls):
        super(ShiftedAffineMapOpTest, cls).setUpClass()

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
            cls._musa_ops = tf.load_op_library(plugin_path)
        else:
            cls._musa_ops = None

    def _run_reference_cpu(self, x, scale, bias):
        with tf.device("/CPU:0"):
            return x * scale + bias

    def _run_musa_shifted_affine_map(self, x, scale, bias):
        if self._musa_ops is None or not hasattr(self._musa_ops, "musa_shifted_affine_map"):
            self.skipTest(
                "MusaShiftedAffineMap op wrapper is not available. "
                "Make sure libmusa_plugin.so is rebuilt and loadable."
            )

        with tf.device("/device:MUSA:0"):
            return self._musa_ops.musa_shifted_affine_map(
                input=x, scale=scale, bias=bias
            )

    def _assert_shifted_affine_close(
        self, x_np, scale_np, bias_np, dtype, rtol=1e-5, atol=1e-6
    ):
        np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

        x = tf.constant(np.array(x_np, dtype=np_dtype), dtype=dtype)
        scale = tf.constant(np.array(scale_np, dtype=np_dtype), dtype=dtype)
        bias = tf.constant(np.array(bias_np, dtype=np_dtype), dtype=dtype)

        cpu_result = self._run_reference_cpu(x, scale, bias)
        musa_result = self._run_musa_shifted_affine_map(x, scale, bias)

        if dtype in [tf.float16, tf.bfloat16]:
            cpu_result = tf.cast(cpu_result, tf.float32)
            musa_result = tf.cast(musa_result, tf.float32)

        self.assertAllClose(
            cpu_result.numpy(), musa_result.numpy(), rtol=rtol, atol=atol
        )

    def test_shifted_affine_vector_broadcast_float32(self):
        x_np = np.array(
            [[1.0, -2.0, 3.0, 4.0], [0.5, 2.0, -1.5, 8.0]], dtype=np.float32
        )
        scale_np = np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32)
        bias_np = np.array([1.0, 0.5, -3.0, 2.0], dtype=np.float32)
        self._assert_shifted_affine_close(
            x_np=x_np,
            scale_np=scale_np,
            bias_np=bias_np,
            dtype=tf.float32,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_shifted_affine_vector_broadcast_float16(self):
        x_np = np.array(
            [[-3.0, 1.5, 0.25, 2.0], [6.0, -4.0, 1.0, -8.0]], dtype=np.float16
        )
        scale_np = np.array([1.0, 0.5, -2.0, 0.125], dtype=np.float16)
        bias_np = np.array([0.0, -1.0, 3.0, 2.0], dtype=np.float16)
        self._assert_shifted_affine_close(
            x_np=x_np,
            scale_np=scale_np,
            bias_np=bias_np,
            dtype=tf.float16,
            rtol=1e-2,
            atol=1e-2,
        )

    def test_shifted_affine_same_shape_float32(self):
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        scale_np = np.array([[2.0, 0.5], [1.5, -1.0]], dtype=np.float32)
        bias_np = np.array([[0.25, -2.0], [1.0, 3.5]], dtype=np.float32)
        self._assert_shifted_affine_close(
            x_np=x_np,
            scale_np=scale_np,
            bias_np=bias_np,
            dtype=tf.float32,
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    tf.test.main()

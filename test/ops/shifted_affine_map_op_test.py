"""Unit tests for custom MusaShiftedAffineMap op."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase, load_musa_ops


def shifted_affine_map_ref(data_left, mask, sliced_var_right):
    """NumPy reference implementation of ShiftedAffineMap."""
    return mask * data_left + sliced_var_right


class ShiftedAffineMapOpTest(MUSATestCase):
    """Tests for MusaShiftedAffineMap kernel correctness."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        try:
            cls._musa_ops = load_musa_ops()
        except Exception as exc:
            print(f"FAILED: Error loading MUSA ops from tensorflow_musa wheel: {exc}")
            cls._musa_ops = None

    def _run_musa_shifted_affine_map(self, data_left, mask,
                                     sliced_var_right):
        if (self._musa_ops is None or
                not hasattr(self._musa_ops, "musa_shifted_affine_map")):
            self.skipTest(
                "MusaShiftedAffineMap op module is not available. "
                "Make sure REGISTER_OP(\"MusaShiftedAffineMap\") is compiled "
                "and the plugin is loaded."
            )

        with tf.device("/device:MUSA:0"):
            return self._musa_ops.musa_shifted_affine_map(
                data_left=data_left,
                mask=mask,
                sliced_var_right=sliced_var_right,
            )

    def _assert_shifted_affine_map_close(self, data_left_np,
                                         mask_np, sliced_var_right_np, dtype,
                                         rtol, atol):
        np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

        data_left = tf.constant(np.array(data_left_np, dtype=np_dtype), dtype=dtype)
        mask = tf.constant(np.array(mask_np, dtype=np_dtype), dtype=dtype)
        sliced_var_right = tf.constant(
            np.array(sliced_var_right_np, dtype=np_dtype), dtype=dtype)

        actual = self._run_musa_shifted_affine_map(
            data_left, mask, sliced_var_right)
        expected = shifted_affine_map_ref(
            np.array(data_left_np, dtype=np_dtype),
            np.array(mask_np, dtype=np_dtype),
            np.array(sliced_var_right_np, dtype=np_dtype),
        )

        actual_np = actual.numpy()
        if dtype in [tf.float16, tf.bfloat16]:
            actual_np = tf.cast(actual, tf.float32).numpy()
            expected = expected.astype(np.float32)

        self.assertEqual(actual.shape, expected.shape)
        self.assertAllClose(actual_np, expected, rtol=rtol, atol=atol)

    def test_basic_float32(self):
        """Basic broadcasted test with float32 inputs."""
        rng = np.random.RandomState(42)
        data_left_np = rng.standard_normal([2, 4, 8]).astype(np.float32)
        mask_np = (rng.random([2, 4, 8]) > 0.3).astype(np.float32)
        sliced_var_right_np = rng.standard_normal([8]).astype(np.float32) * 0.1

        self._assert_shifted_affine_map_close(
            data_left_np,
            mask_np,
            sliced_var_right_np,
            tf.float32,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_basic_float16(self):
        """Basic broadcasted test with float16 inputs."""
        rng = np.random.RandomState(7)
        data_left_np = rng.standard_normal([2, 4, 8]).astype(np.float16)
        mask_np = (rng.random([2, 4, 8]) > 0.5).astype(np.float16)
        sliced_var_right_np = (rng.standard_normal([8]) * 0.1).astype(np.float16)

        self._assert_shifted_affine_map_close(
            data_left_np,
            mask_np,
            sliced_var_right_np,
            tf.float16,
            rtol=1e-2,
            atol=1e-2,
        )

    def test_basic_bfloat16(self):
        """Basic broadcasted test with bfloat16 inputs."""
        rng = np.random.RandomState(11)
        data_left_np = rng.standard_normal([2, 4, 8]).astype(np.float32)
        mask_np = (rng.random([2, 4, 8]) > 0.4).astype(np.float32)
        sliced_var_right_np = rng.standard_normal([8]).astype(np.float32) * 0.1

        self._assert_shifted_affine_map_close(
            data_left_np,
            mask_np,
            sliced_var_right_np,
            tf.bfloat16,
            rtol=2e-2,
            atol=2e-2,
        )

    def test_multi_axis_broadcast_float32(self):
        """Exercise rank expansion and multi-axis broadcasting."""
        rng = np.random.RandomState(1234)
        data_left_np = rng.standard_normal([2, 1, 4, 8]).astype(np.float32)
        mask_np = (rng.random([2, 3, 1, 8]) > 0.35).astype(np.float32)
        sliced_var_right_np = (
            rng.standard_normal([1, 3, 4, 1]).astype(np.float32) * 0.1)

        self._assert_shifted_affine_map_close(
            data_left_np,
            mask_np,
            sliced_var_right_np,
            tf.float32,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_broadcast_float64(self):
        """Verify the dedicated float64 kernel path with broadcasting."""
        rng = np.random.RandomState(2024)
        data_left_np = rng.standard_normal([2, 3, 4]).astype(np.float64)
        mask_np = (rng.random([2, 1, 4]) > 0.45).astype(np.float64)
        sliced_var_right_np = (
            rng.standard_normal([4]).astype(np.float64) * 0.1)

        self._assert_shifted_affine_map_close(
            data_left_np,
            mask_np,
            sliced_var_right_np,
            tf.float64,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_empty_tensor(self):
        """Test zero-element tensor handling."""
        data_left_np = np.zeros([0, 8], dtype=np.float32)
        mask_np = np.zeros([0, 8], dtype=np.float32)
        sliced_var_right_np = np.linspace(0.2, -0.2, num=8).astype(np.float32)

        self._assert_shifted_affine_map_close(
            data_left_np,
            mask_np,
            sliced_var_right_np,
            tf.float32,
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    tf.test.main()

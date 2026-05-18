"""Tests for MUSA MatrixBandPart operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class MatrixBandPartOpTest(MUSATestCase):
    """Tests for MUSA MatrixBandPart operator."""

    def _test_matrix_band_part(self, shape, num_lower, num_upper, dtype,
                                rtol=1e-5, atol=1e-8):
        """Run matrix_band_part on CPU and MUSA and compare the results."""
        np_dtype = dtype.as_numpy_dtype
        if dtype in (tf.float16, tf.bfloat16):
            input_np = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
            input_np = input_np.astype(np_dtype)
        elif dtype in (tf.int32, tf.int64):
            input_np = np.random.randint(-100, 100, shape).astype(np_dtype)
        else:
            input_np = np.random.uniform(-1.0, 1.0, shape).astype(np_dtype)

        input_tf = tf.constant(input_np, dtype=dtype)
        num_lower_tf = tf.constant(num_lower, dtype=tf.int64)
        num_upper_tf = tf.constant(num_upper, dtype=tf.int64)

        def op_func(inp, nl, nu):
            return tf.linalg.band_part(inp, nl, nu)

        self._compare_cpu_musa_results(
            op_func, [input_tf, num_lower_tf, num_upper_tf], dtype,
            rtol=rtol, atol=atol)

    def testBandPartSquareMatrix(self):
        """Basic square 2-D matrix."""
        for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            with self.subTest(dtype=dtype):
                self._test_matrix_band_part([6, 6], 2, 2, dtype,
                                            rtol=rtol, atol=atol)

    def testBandPartRectangularMatrix(self):
        """Non-square 2-D matrix."""
        for dtype in [tf.float32, tf.int32]:
            with self.subTest(dtype=dtype):
                self._test_matrix_band_part([4, 8], 1, 3, dtype)
                self._test_matrix_band_part([8, 4], 3, 1, dtype)

    def testBandPartLowerTriangular(self):
        """Lower triangular (num_upper=0)."""
        for dtype in [tf.float32, tf.float16, tf.int32, tf.int64]:
            rtol = 1e-2 if dtype == tf.float16 else 1e-5
            atol = 1e-2 if dtype == tf.float16 else 1e-8
            with self.subTest(dtype=dtype):
                self._test_matrix_band_part([5, 5], -1, 0, dtype,
                                            rtol=rtol, atol=atol)

    def testBandPartUpperTriangular(self):
        """Upper triangular (num_lower=0)."""
        for dtype in [tf.float32, tf.float16, tf.int32, tf.int64]:
            rtol = 1e-2 if dtype == tf.float16 else 1e-5
            atol = 1e-2 if dtype == tf.float16 else 1e-8
            with self.subTest(dtype=dtype):
                self._test_matrix_band_part([5, 5], 0, -1, dtype,
                                            rtol=rtol, atol=atol)

    def testBandPartDiagonalOnly(self):
        """Main diagonal only (num_lower=0, num_upper=0)."""
        for dtype in [tf.float32, tf.int32]:
            with self.subTest(dtype=dtype):
                self._test_matrix_band_part([6, 6], 0, 0, dtype)

    def testBandPartNoOp(self):
        """num_lower=-1 and num_upper=-1 keeps the whole matrix unchanged."""
        for dtype in [tf.float32, tf.int32]:
            with self.subTest(dtype=dtype):
                self._test_matrix_band_part([4, 7], -1, -1, dtype)

    def testBandPartBatched(self):
        """3-D batched matrices."""
        for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            with self.subTest(dtype=dtype):
                self._test_matrix_band_part([3, 5, 5], 1, 2, dtype,
                                            rtol=rtol, atol=atol)

    def testBandPartHigherRankBatched(self):
        """4-D batched matrices."""
        self._test_matrix_band_part([2, 3, 4, 4], 2, 1, tf.float32)

    def testBandPartMaxBand(self):
        """num_lower = m-1, num_upper = n-1 keeps the whole matrix (boundary values)."""
        # TF rejects num_lower > rows or num_upper > cols; m-1 and n-1 are the
        # largest valid values and should produce the same result as -1 (no-op).
        self._test_matrix_band_part([8, 8], 7, 7, tf.float32)

    def testBandPartDouble(self):
        """Double precision."""
        self._test_matrix_band_part([7, 7], 2, 3, tf.float64)

    def testBandPartInt32NumBands(self):
        """num_lower / num_upper given as int32 tensors."""
        np_dtype = np.float32
        shape = [4, 4]
        input_np = np.random.uniform(-1.0, 1.0, shape).astype(np_dtype)
        input_tf = tf.constant(input_np, dtype=tf.float32)
        nl = tf.constant(1, dtype=tf.int32)
        nu = tf.constant(1, dtype=tf.int32)

        def op_func(inp, nl, nu):
            return tf.linalg.band_part(inp, nl, nu)

        self._compare_cpu_musa_results(op_func, [input_tf, nl, nu],
                                       tf.float32)

    def testBandPartZeroElements(self):
        """Input tensor with zero elements returns empty output."""
        input_tf = tf.constant(np.zeros([0, 4, 4], dtype=np.float32),
                               dtype=tf.float32)
        num_lower = tf.constant(1, dtype=tf.int64)
        num_upper = tf.constant(1, dtype=tf.int64)

        def op_func(inp, nl, nu):
            return tf.linalg.band_part(inp, nl, nu)

        self._compare_cpu_musa_results(op_func, [input_tf, num_lower, num_upper],
                                       tf.float32)


if __name__ == "__main__":
    tf.test.main()

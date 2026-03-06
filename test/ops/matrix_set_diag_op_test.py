"""Tests for MUSA MatrixSetDiag operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class MatrixSetDiagOpTest(MUSATestCase):
    """Tests for MUSA MatrixSetDiag operator."""

    def _test_matrix_set_diag(self, input_shape, diag_shape, diag_index=None, align=None, dtype=tf.float32, rtol=1e-5, atol=1e-8):
        """Run MatrixSetDiag on CPU and MUSA and compare the results."""
        
        # Create input tensor
        input_np = np.random.uniform(-1, 1, input_shape).astype(dtype.as_numpy_dtype)
        input_tf = tf.constant(input_np, dtype=dtype)
        
        # Create diagonal tensor
        diag_np = np.random.uniform(-1, 1, diag_shape).astype(dtype.as_numpy_dtype)
        diag_tf = tf.constant(diag_np, dtype=dtype)

        def op_func(input_v, diag_v):
            if diag_index is not None:
                k = tf.constant(diag_index, dtype=tf.int32)
                return tf.linalg.set_diag(input_v, diag_v, k=k, align=align)
            else:
                return tf.linalg.set_diag(input_v, diag_v)

        # Compare CPU and MUSA results
        self._compare_cpu_musa_results(op_func, [input_tf, diag_tf], dtype, rtol=rtol, atol=atol)

    def testMatrixSetDiagBasic(self):
        """Basic MatrixSetDiag test (k=0)."""
        for dtype in [tf.float32, tf.float16, tf.int32, tf.int64]:
            rtol = 1e-3 if dtype == tf.float16 else 1e-5
            atol = 1e-3 if dtype == tf.float16 else 1e-8
            # 2D matrix
            self._test_matrix_set_diag(input_shape=[4, 4], diag_shape=[4], dtype=dtype, rtol=rtol, atol=atol)
            # Batched 3D
            self._test_matrix_set_diag(input_shape=[2, 3, 3], diag_shape=[2, 3], dtype=dtype, rtol=rtol, atol=atol)

    def testMatrixSetDiagV3ScalarK(self):
        """MatrixSetDiagV3 with scalar k (offset diagonal)."""
        dtype = tf.float32
        # k > 0 (super-diagonal)
        self._test_matrix_set_diag(input_shape=[4, 4], diag_shape=[3], diag_index=1, dtype=dtype)
        # k < 0 (sub-diagonal)
        self._test_matrix_set_diag(input_shape=[4, 4], diag_shape=[3], diag_index=-1, dtype=dtype)

    def testMatrixSetDiagV3RangeK(self):
        """MatrixSetDiagV3 with range k [lower, upper]."""
        dtype = tf.float32
        # k = [-1, 1], input [4, 4], num_diags = 3, max_diag_len = 4
        # diag_shape should be [num_diags, max_diag_len] -> [3, 4]
        self._test_matrix_set_diag(input_shape=[4, 4], diag_shape=[3, 4], diag_index=[-1, 1], align="LEFT_RIGHT", dtype=dtype)
        
        # Batched case
        self._test_matrix_set_diag(input_shape=[2, 5, 5], diag_shape=[2, 2, 5], diag_index=[-1, 0], align="LEFT_LEFT", dtype=dtype)

    def testMatrixSetDiagV3Alignments(self):
        """Test different alignment options for V3."""
        dtype = tf.float32
        input_shape = [4, 4]
        diag_index = [-1, 1] # 3 diagonals: -1, 0, 1
        diag_shape = [3, 4]
        
        for align in ["LEFT_RIGHT", "LEFT_LEFT", "RIGHT_LEFT", "RIGHT_RIGHT"]:
            self._test_matrix_set_diag(input_shape=input_shape, diag_shape=diag_shape, diag_index=diag_index, align=align, dtype=dtype)

if __name__ == "__main__":
    tf.test.main()

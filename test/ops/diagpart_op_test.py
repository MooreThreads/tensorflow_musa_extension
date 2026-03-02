"""Tests for MUSA DiagPart operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class DiagPartOpTest(MUSATestCase):
  """Tests for MUSA DiagPart operator."""

  def _build_input_tensor(self, shape, dtype):
    """Create an input tensor whose first and second halves share the same sizes."""
    size = int(np.prod(shape))
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    values = np.arange(size, dtype=np_dtype).reshape(shape)
    return tf.constant(values, dtype=dtype)

  def _test_diag_part(self, shape, dtype, rtol=1e-5, atol=1e-8):
    """Run DiagPart on CPU and MUSA and compare the results."""
    tensor = self._build_input_tensor(shape, dtype)
    self._compare_cpu_musa_results(tf.linalg.diag_part, [tensor], dtype,
                                  rtol=rtol, atol=atol)

  def testDiagPartMatrix(self):
    """Diagonal extraction on a square matrix."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_diag_part([8, 8], dtype, rtol=rtol, atol=atol)

  def testDiagPartTensor(self):
    """Diagonal extraction on higher-rank tensors."""
    test_shapes = [
        [2, 3, 2, 3],
        [4, 1, 4, 1],
        [1, 5, 1, 5],
    ]
    for dtype in [tf.float32, tf.bfloat16]:
      rtol = 1e-2 if dtype == tf.bfloat16 else 1e-5
      atol = 1e-2 if dtype == tf.bfloat16 else 1e-8
      for shape in test_shapes:
        self._test_diag_part(shape, dtype, rtol=rtol, atol=atol)


if __name__ == "__main__":
  tf.test.main()

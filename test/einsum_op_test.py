# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for the MUSA Einsum operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class EinsumOpTest(MUSATestCase):
  """Tests for the MUSA Einsum operator."""

  def _random_inputs(self, shapes, dtype):
    """Generate random inputs for the requested shapes and dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    return [
        tf.constant(
            np.random.uniform(-1.0, 1.0, size=shape).astype(np_dtype),
            dtype=dtype)
        for shape in shapes
    ]

  def _test_einsum(self, equation, shapes, dtype, rtol=1e-5, atol=1e-8):
    """Compare CPU vs MUSA for the given einsum equation."""
    inputs = self._random_inputs(shapes, dtype)
    op = lambda *tensors: tf.einsum(equation, *tensors)
    self._compare_cpu_musa_results(op, inputs, dtype, rtol=rtol, atol=atol)

  def testMatrixMultiplication(self):
    """Matrix multiplication with explicit contraction indices."""
    equation = "ij,jk->ik"
    shapes = [(128, 64), (64, 96)]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_einsum(equation, shapes, dtype, rtol=rtol, atol=atol)

  def testBatchBroadcastContraction(self):
    """Batch contraction with broadcasting over leading dims."""
    equation = "bij,jk->bik"
    shapes = [(4, 16, 32), (32, 64)]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_einsum(equation, shapes, dtype, rtol=rtol, atol=atol)

  def testDiagonalAndBroadcast(self):
    """Repeated indices that take diagonals and broadcast shapes."""
    equation = "iij,ij->ij"
    shapes = [(4, 4, 6), (4, 6)]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_einsum(equation, shapes, dtype, rtol=rtol, atol=atol)

  def testEllipsisBroadcast(self):
    """Ellipsis handling with mixed-rank operands."""
    equation = "...i,i->..."
    shapes = [(2, 3, 5), (5,)]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_einsum(equation, shapes, dtype, rtol=rtol, atol=atol)

  def testMultipleSummations(self):
    """Multiple contraction indices with more than two inputs."""
    equation = "abc,acd,db->bd"
    shapes = [(3, 4, 5), (3, 5, 6), (6, 4)]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      self._test_einsum(equation, shapes, dtype)


if __name__ == "__main__":
  tf.test.main()

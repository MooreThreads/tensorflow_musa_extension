"""Tests for MUSA Einsum operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class EinsumOpTest(MUSATestCase):
	"""Tests for MUSA Einsum operator with TensorFlow-compatible behavior."""

	def _run_cpu_musa(self, equation, inputs, rtol=1e-5, atol=1e-8):
		"""Run einsum on CPU and MUSA then compare outputs."""
		with tf.device('/CPU:0'):
			cpu_result = tf.einsum(equation, *inputs)

		with tf.device('/device:MUSA:0'):
			musa_result = tf.einsum(equation, *inputs)

		if cpu_result.dtype in [tf.float16, tf.bfloat16]:
			cpu_result = tf.cast(cpu_result, tf.float32)
			musa_result = tf.cast(musa_result, tf.float32)

		self.assertAllEqual(cpu_result.shape, musa_result.shape)
		self.assertAllClose(cpu_result.numpy(), musa_result.numpy(), rtol=rtol, atol=atol)

	def _assert_error_consistency(self, equation, inputs):
		"""Assert CPU and MUSA both raise TensorFlow exceptions for invalid inputs."""
		cpu_error = None
		musa_error = None

		try:
			with tf.device('/CPU:0'):
				tf.einsum(equation, *inputs)
		except Exception as e:  # pylint: disable=broad-except
			cpu_error = e

		try:
			with tf.device('/device:MUSA:0'):
				tf.einsum(equation, *inputs)
		except Exception as e:  # pylint: disable=broad-except
			musa_error = e

		self.assertIsNotNone(cpu_error, "CPU should raise for invalid einsum input")
		self.assertIsNotNone(musa_error, "MUSA should raise for invalid einsum input")
		self.assertEqual(type(cpu_error), type(musa_error))

	def testEinsumMatrixMultiplication(self):
		"""Test classic matrix multiplication: ij,jk->ik."""
		for dtype in [tf.float32, tf.float16, tf.bfloat16]:
			a_np = np.random.uniform(-1.0, 1.0, size=(8, 16)).astype(np.float32)
			b_np = np.random.uniform(-1.0, 1.0, size=(16, 10)).astype(np.float32)
			a = tf.constant(a_np, dtype=dtype)
			b = tf.constant(b_np, dtype=dtype)

			rtol = 2e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
			atol = 2e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-6
			self._run_cpu_musa('ij,jk->ik', [a, b], rtol=rtol, atol=atol)

	def testEinsumBatchMatMul(self):
		"""Test batched matmul: bij,bjk->bik."""
		a = tf.constant(np.random.uniform(-1.0, 1.0, size=(4, 6, 8)).astype(np.float32))
		b = tf.constant(np.random.uniform(-1.0, 1.0, size=(4, 8, 5)).astype(np.float32))
		self._run_cpu_musa('bij,bjk->bik', [a, b])

	def testEinsumBroadcastWithEllipsis(self):
		"""Test ellipsis broadcasting: ...ij,...jk->...ik."""
		a = tf.constant(np.random.uniform(-1.0, 1.0, size=(2, 3, 4, 6)).astype(np.float32))
		b = tf.constant(np.random.uniform(-1.0, 1.0, size=(1, 3, 6, 5)).astype(np.float32))
		self._run_cpu_musa('...ij,...jk->...ik', [a, b])

	def testEinsumImplicitOutput(self):
		"""Test implicit output mode: ij,jk (without ->)."""
		a = tf.constant(np.random.uniform(-1.0, 1.0, size=(7, 11)).astype(np.float32))
		b = tf.constant(np.random.uniform(-1.0, 1.0, size=(11, 9)).astype(np.float32))
		self._run_cpu_musa('ij,jk', [a, b])

	def testEinsumTranspose(self):
		"""Test axis permutation with single input: ijk->kji."""
		x = tf.constant(np.random.uniform(-1.0, 1.0, size=(3, 5, 7)).astype(np.float32))
		self._run_cpu_musa('ijk->kji', [x])

	def testEinsumDiagonalExtraction(self):
		"""Test repeated indices for diagonal extraction: ii->i."""
		x = tf.constant(np.random.uniform(-1.0, 1.0, size=(9, 9)).astype(np.float32))
		self._run_cpu_musa('ii->i', [x])

	def testEinsumReduction(self):
		"""Test reduction to scalar: ij->."""
		x = tf.constant(np.random.uniform(-1.0, 1.0, size=(13, 17)).astype(np.float32))
		self._run_cpu_musa('ij->', [x])

	def testEinsumOuterProduct(self):
		"""Test outer product: i,j->ij."""
		x = tf.constant(np.random.uniform(-1.0, 1.0, size=(12,)).astype(np.float32))
		y = tf.constant(np.random.uniform(-1.0, 1.0, size=(8,)).astype(np.float32))
		self._run_cpu_musa('i,j->ij', [x, y])

	def testEinsumThreeOperands(self):
		"""Test multi-operand contraction: ab,bc,cd->ad."""
		a = tf.constant(np.random.uniform(-1.0, 1.0, size=(4, 6)).astype(np.float32))
		b = tf.constant(np.random.uniform(-1.0, 1.0, size=(6, 5)).astype(np.float32))
		c = tf.constant(np.random.uniform(-1.0, 1.0, size=(5, 3)).astype(np.float32))
		self._run_cpu_musa('ab,bc,cd->ad', [a, b, c])

	def testEinsumInvalidEquation(self):
		"""Invalid equation should raise consistently on CPU and MUSA."""
		x = tf.constant(np.random.uniform(-1.0, 1.0, size=(2, 2)).astype(np.float32))
		self._assert_error_consistency('ij->ik', [x])

	def testEinsumMismatchedDimensions(self):
		"""Dimension mismatch should raise consistently on CPU and MUSA."""
		a = tf.constant(np.random.uniform(-1.0, 1.0, size=(4, 5)).astype(np.float32))
		b = tf.constant(np.random.uniform(-1.0, 1.0, size=(6, 3)).astype(np.float32))
		self._assert_error_consistency('ij,jk->ik', [a, b])


if __name__ == "__main__":
	np.random.seed(2026)
	tf.random.set_seed(2026)
	tf.test.main()

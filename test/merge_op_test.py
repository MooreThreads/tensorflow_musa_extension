"""Tests for MUSA Merge operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class MergeOpTest(MUSATestCase):
  """Tests for tf.raw_ops.Merge on MUSA."""

  _DTYPES = (tf.float32, tf.float16, tf.int32, tf.bool)

  def _sample_input(self, dtype):
    if dtype == tf.bool:
      return np.array([[True, False, True], [False, True, False]], dtype=np.bool_)

    np_dtype = dtype.as_numpy_dtype
    if np.issubdtype(np_dtype, np.integer):
      data = np.arange(6).reshape(2, 3) - 2
      return data.astype(np_dtype)

    base = np.array([[-1.0, -0.5, 0.0], [0.5, 1.0, 1.5]], dtype=np.float32)
    return base.astype(np_dtype)

  # Marking this function with tf.function to ensure the 'invalid' branches are properly marked as **dead** and pruned by the compiler,
  # which is essential for testing the Merge operator's behavior with inactive branches.
  @tf.function
  def _run_merge_case(self, dtype, pred_value, device):
    data_np = self._sample_input(dtype)
    with tf.device(device):
      data = tf.constant(data_np, dtype=dtype)
      pred = tf.constant(pred_value, dtype=tf.bool)
      false_out, true_out = tf.raw_ops.Switch(data=data, pred=pred)
      merged, value_index = tf.raw_ops.Merge(inputs=[false_out, true_out])
    return merged, value_index

  def _assert_merge_consistency(self, dtype, pred_value):
    cpu_val, cpu_idx = self._run_merge_case(dtype, pred_value, '/CPU:0')
    musa_val, musa_idx = self._run_merge_case(dtype, pred_value, '/device:MUSA:0')

    if dtype == tf.bool:
      self.assertAllEqual(cpu_val.numpy(), musa_val.numpy())
    else:
      rtol = 1e-2 if dtype == tf.float16 else 1e-5
      atol = 1e-2 if dtype == tf.float16 else 1e-8
      self.assertAllClose(cpu_val.numpy(), musa_val.numpy(), rtol=rtol, atol=atol)

    self.assertEqual(cpu_idx.numpy(), musa_idx.numpy())

  def test_merge_true_branch_matches_cpu(self):
    for dtype in self._DTYPES:
      self._assert_merge_consistency(dtype, True)

  def test_merge_false_branch_matches_cpu(self):
    for dtype in self._DTYPES:
      self._assert_merge_consistency(dtype, False)

  # Marking this function with tf.function to ensure the 'invalid' branches are properly marked as **dead** and pruned by the compiler,
  # which is essential for testing the Merge operator's behavior with inactive branches.
  @tf.function
  def _run_multi_input_case(self, active_index, device):
    dtype = tf.float32
    base = self._sample_input(dtype)
    with tf.device(device):
      tensors = [tf.constant(base + float(i), dtype=dtype) for i in range(3)]
      gated_inputs = []
      for idx, tensor in enumerate(tensors):
        pred = tf.constant(True)
        false_branch, true_branch = tf.raw_ops.Switch(data=tensor, pred=pred)
        # Only keep the branch we expect to be live.
        gated_inputs.append(true_branch if idx == active_index else false_branch)
      merged, value_index = tf.raw_ops.Merge(inputs=gated_inputs)
    return merged, value_index

  def test_merge_multiple_inputs_selects_live_tensor(self):
    for active_index in range(3):
      cpu_val, cpu_idx = self._run_multi_input_case(active_index, '/CPU:0')
      musa_val, musa_idx = self._run_multi_input_case(active_index, '/device:MUSA:0')
      self.assertAllClose(cpu_val.numpy(), musa_val.numpy(), rtol=1e-5, atol=1e-8)
      self.assertEqual(cpu_idx.numpy(), musa_idx.numpy())


if __name__ == "__main__":
  tf.test.main()

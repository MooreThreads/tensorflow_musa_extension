"""Shared helpers for XLA custom op tests."""

import os

os.environ.setdefault("MUSA_ENABLE_TF32", "0")

import numpy as np
import tensorflow as tf
import tensorflow_musa as tf_musa

from musa_test_utils import MUSATestCase


MUSA_OPS = tf_musa.get_musa_ops()


def as_numpy(value):
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def broadcast_gate(gate, out_shape):
    gate = np.asarray(gate, dtype=np.bool_)
    if gate.ndim == 1 and len(out_shape) >= 2 and gate.shape[0] == out_shape[0]:
        gate = gate.reshape([out_shape[0]] + [1] * (len(out_shape) - 1))
    return np.broadcast_to(gate, out_shape)


def normalize_ref(x, epsilon, max_std):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.clip(std, epsilon, max_std)
    return (x - mean) / std


def run_on_cpu(fn, *args, **kwargs):
    with tf.device("/CPU:0"):
        return fn(*args, **kwargs)


def layer_norm_ref(x, gamma, beta, epsilon):
    with tf.device("/CPU:0"):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        x_hat = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return x_hat * gamma + beta


def pln_cascade_ref(norm_out, mask, add_input, bias_input, select_on_true):
    out_shape = np.broadcast_shapes(
        np.shape(norm_out), np.shape(add_input), np.shape(bias_input)
    )
    norm = np.broadcast_to(norm_out, out_shape)
    gate = broadcast_gate(mask, out_shape)
    add = np.broadcast_to(add_input, out_shape)
    bias = np.broadcast_to(bias_input, out_shape)
    candidate = norm * add + bias
    chooser = gate if select_on_true else np.logical_not(gate)
    return np.where(chooser, candidate, norm)


def pln_cascade_block_ref(norm_out, add_table, bias_table, gates,
                          table_indices, select_on_true):
    out = np.array(norm_out, copy=True)
    width = out.shape[-1]
    row_shape = [1] * (out.ndim - 1) + [width]
    for gate, row, select_flag in zip(gates, table_indices, select_on_true):
        add = add_table[row].reshape(row_shape)
        bias = bias_table[row].reshape(row_shape)
        candidate = out * add + bias
        chooser = broadcast_gate(gate, out.shape)
        if not select_flag:
            chooser = np.logical_not(chooser)
        out = np.where(chooser, candidate, out)
    return out


class XlaOpTestCase(MUSATestCase):
    """Base class for XLA custom op tests."""

    def run_xla(self, fn, *args):
        @tf.function(jit_compile=True)
        def compiled(*inputs):
            with tf.device("/device:MUSA:0"):
                return fn(*inputs)

        return compiled(*args)

    def assertTensorClose(self, actual, expected, dtype=tf.float32,
                          rtol=None, atol=None):
        if rtol is None or atol is None:
            if dtype in (tf.float16, tf.bfloat16):
                rtol, atol = 2e-2, 2e-2
            else:
                rtol, atol = 1e-5, 1e-6

        if isinstance(actual, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assertTensorClose(
                    actual_item, expected_item, dtype=dtype, rtol=rtol, atol=atol
                )
            return

        actual_np = as_numpy(actual)
        expected_np = as_numpy(expected)
        if dtype in (tf.float16, tf.bfloat16):
            actual_np = actual_np.astype(np.float32)
            expected_np = expected_np.astype(np.float32)
        self.assertAllClose(actual_np, expected_np, rtol=rtol, atol=atol)

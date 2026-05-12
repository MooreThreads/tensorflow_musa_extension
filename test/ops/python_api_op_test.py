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

"""Tests for the public MUSA Python op API."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tensorflow as tf


if "tensorflow_musa" not in sys.modules:
    package_dir = Path(__file__).resolve().parents[2] / "python"
    spec = importlib.util.spec_from_file_location(
        "tensorflow_musa",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    tensorflow_musa = importlib.util.module_from_spec(spec)
    sys.modules["tensorflow_musa"] = tensorflow_musa
    spec.loader.exec_module(tensorflow_musa)

import tensorflow_musa
from tensorflow_musa import ops, raw_ops


class PythonApiOpTest(unittest.TestCase):

    def _patch_raw_op(self, name, return_value):
        op = mock.Mock(return_value=return_value)
        sentinel = object()
        previous = ops.raw_ops.__dict__.get(name, sentinel)
        ops.raw_ops.__dict__[name] = op
        self.addCleanup(self._restore_raw_op, name, previous, sentinel)
        return op

    def _restore_raw_op(self, name, previous, sentinel):
        if previous is sentinel:
            ops.raw_ops.__dict__.pop(name, None)
        else:
            ops.raw_ops.__dict__[name] = previous

    def testPackageExportsOpModules(self):
        self.assertIs(tensorflow_musa.ops, ops)
        self.assertIs(tensorflow_musa.raw_ops, raw_ops)
        self.assertIn("ops", tensorflow_musa.__all__)
        self.assertIn("raw_ops", tensorflow_musa.__all__)

    def testRawOpsDelegatesToGeneratedModule(self):
        generated = types.SimpleNamespace(musa_clip=lambda **kwargs: kwargs)

        with mock.patch("tensorflow_musa.raw_ops.get_musa_ops", return_value=generated):
            result = raw_ops.musa_clip(x="x", lo="lo", hi="hi", name="clip")

        self.assertEqual(
            result,
            {"x": "x", "lo": "lo", "hi": "hi", "name": "clip"},
        )

    def testRawOpsMissingOpRaisesAttributeError(self):
        with mock.patch(
            "tensorflow_musa.raw_ops.get_musa_ops",
            return_value=types.SimpleNamespace(),
        ):
            with self.assertRaisesRegex(AttributeError, "MUSA raw op 'missing_op'"):
                raw_ops.missing_op

    def testClipWrapperDelegatesToRawOp(self):
        op = self._patch_raw_op("musa_clip", "result")
        result = ops.clip("x", "lo", "hi", name="clip")

        self.assertEqual(result, "result")
        op.assert_called_once_with(x="x", lo="lo", hi="hi", name="clip")

    def testLayerNormWrapperDelegatesToRawOp(self):
        op = self._patch_raw_op("musa_layer_norm", "result")
        result = ops.layer_norm("x", "gamma", "beta", epsilon=0.1, name="ln")

        self.assertEqual(result, "result")
        op.assert_called_once_with(
            x="x",
            gamma="gamma",
            beta="beta",
            epsilon=0.1,
            name="ln",
        )

    def testShiftedAffineMapWrapperDelegatesToRawOp(self):
        op = self._patch_raw_op("musa_shifted_affine_map", "result")
        result = ops.shifted_affine_map("data", "mask", "slice", name="sam")

        self.assertEqual(result, "result")
        op.assert_called_once_with(
            data_left="data",
            mask="mask",
            sliced_var_right="slice",
            name="sam",
        )

    def testInteractWrapperDelegatesToRawOp(self):
        op = self._patch_raw_op("musa_interact", "result")
        result = ops.interact("input", name="interact")

        self.assertEqual(result, "result")
        op.assert_called_once_with(input="input", name="interact")

    def testGeluWrapperDelegatesToRawOp(self):
        op = self._patch_raw_op("musa_gelu", "result")
        result = ops.gelu("x", approximate=True, name="gelu")

        self.assertEqual(result, "result")
        op.assert_called_once_with(x="x", approximate=True, name="gelu")

    def testReshapeMatMulWrapperDelegatesToRawOp(self):
        op = self._patch_raw_op("musa_reshape_mat_mul", "result")
        result = ops.reshape_mat_mul("x", "w", transpose_b=True, name="rmm")

        self.assertEqual(result, "result")
        op.assert_called_once_with(
            x="x",
            w="w",
            transpose_b=True,
            name="rmm",
        )

    def testMatmulBiasAddWrapperDelegatesToRawOp(self):
        op = self._patch_raw_op("musa_mat_mul_bias_add", "result")
        result = ops.matmul_bias_add(
            "a",
            "b",
            "bias",
            transpose_a=True,
            transpose_b=True,
            name="mba",
        )

        self.assertEqual(result, "result")
        op.assert_called_once_with(
            a="a",
            b="b",
            bias="bias",
            transpose_a=True,
            transpose_b=True,
            name="mba",
        )

    def testResourceApplyNadamWrapperDelegatesToRawOp(self):
        op = self._patch_raw_op("ResourceApplyNadam", "result")
        result = ops.resource_apply_nadam(
            "var",
            "m",
            "v",
            "beta1_power",
            "beta2_power",
            "lr",
            "beta1",
            "beta2",
            "epsilon",
            "grad",
            use_locking=True,
            name="nadam",
        )

        self.assertEqual(result, "result")
        op.assert_called_once_with(
            var="var",
            m="m",
            v="v",
            beta1_power="beta1_power",
            beta2_power="beta2_power",
            lr="lr",
            beta1="beta1",
            beta2="beta2",
            epsilon="epsilon",
            grad="grad",
            use_locking=True,
            name="nadam",
        )

    def testDropoutWrappersDelegateToRawOps(self):
        op = self._patch_raw_op("musa_dropout", ("y", "mask"))
        result = ops.dropout("x", rate=0.25, seed=1, offset=2, name="drop")

        self.assertEqual(result, ("y", "mask"))
        op.assert_called_once_with(
            x="x",
            rate=0.25,
            seed=1,
            offset=2,
            name="drop",
        )

        grad_op = self._patch_raw_op("musa_dropout_grad", "grad")
        grad = ops.dropout_grad("dy", "mask", rate=0.25, name="drop_grad")

        self.assertEqual(grad, "grad")
        grad_op.assert_called_once_with(
            grad="dy",
            mask="mask",
            rate=0.25,
            name="drop_grad",
        )

    def testResourceSparseApplyAdamWrapperDelegatesToRawOp(self):
        op = self._patch_raw_op("musa_resource_sparse_apply_adam", "result")
        result = ops.resource_sparse_apply_adam(
                "var",
                "m",
                "v",
                "beta1_power",
                "beta2_power",
                "lr",
                "beta1",
                "beta2",
                "epsilon",
                "grad",
                "indices",
                use_locking=True,
                name="adam",
            )

        self.assertEqual(result, "result")
        op.assert_called_once_with(
            var="var",
            m="m",
            v="v",
            beta1_power="beta1_power",
            beta2_power="beta2_power",
            lr="lr",
            beta1="beta1",
            beta2="beta2",
            epsilon="epsilon",
            grad="grad",
            indices="indices",
            use_locking=True,
            name="adam",
        )

    def testRealClipComputesExpectedValues(self):
        result = ops.clip(tf.constant([-2.0, 0.5, 3.0]), 0.0, 1.0)

        np.testing.assert_allclose(result.numpy(), [0.0, 0.5, 1.0])

    def testRealLayerNormComputesExpectedValues(self):
        x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = ops.layer_norm(x, tf.ones([3]), tf.zeros([3]))

        expected = tf.nn.batch_normalization(
            x,
            mean=tf.reduce_mean(x, axis=-1, keepdims=True),
            variance=tf.math.reduce_variance(x, axis=-1, keepdims=True),
            offset=tf.zeros([3]),
            scale=tf.ones([3]),
            variance_epsilon=0.00001,
        )
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)

    def testRealShiftedAffineMapComputesExpectedValues(self):
        result = ops.shifted_affine_map(
            tf.ones([2, 3]),
            tf.ones([2, 3]),
            tf.ones([2, 3]),
        )

        np.testing.assert_allclose(result.numpy(), np.full([2, 3], 2.0))

    def testRealInteractComputesExpectedValues(self):
        result = ops.interact(tf.ones([2, 3, 4]))

        np.testing.assert_allclose(result.numpy(), np.full([2, 3, 3], 4.0))

    def testRealDropoutComputesOutputAndMask(self):
        y, mask = ops.dropout(tf.ones([2, 3]), rate=0.5, seed=1, offset=0)

        self.assertEqual(y.shape, [2, 3])
        self.assertEqual(mask.shape, [2, 3])
        np.testing.assert_allclose(y.numpy(), mask.numpy().astype(np.float32) * 2.0)
    def testRealGeluComputesExpectedValues(self):
        x = tf.constant([-2.0, -0.5, 0.0, 0.5, 2.0])
        for approximate in [False, True]:
            with self.subTest(approximate=approximate):
                result = ops.gelu(x, approximate=approximate)
                expected = tf.nn.gelu(x, approximate=approximate)

                np.testing.assert_allclose(
                    result.numpy(),
                    expected.numpy(),
                    rtol=1e-5,
                    atol=1e-6,
                )

    def testRealReshapeMatMulComputesExpectedValues(self):
        x = tf.constant([[[1.0, 2.0], [3.0, 4.0]]])
        w = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        result = ops.reshape_mat_mul(x, w)

        np.testing.assert_allclose(result.numpy(), tf.matmul(x, w).numpy())

    def testRealMatmulBiasAddComputesExpectedValues(self):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        bias = tf.constant([0.5, -0.5])
        result = ops.matmul_bias_add(a, b, bias)

        expected = tf.matmul(a, b) + bias
        np.testing.assert_allclose(result.numpy(), expected.numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)

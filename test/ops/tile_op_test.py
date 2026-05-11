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

"""Tests for MUSA Tile operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class TileOpTest(MUSATestCase):
    """Tests for MUSA Tile operator."""

    def _test_tile(
        self, input_np, multiples, dtype, multiples_dtype=np.int32, rtol=1e-5, atol=1e-8
    ):
        """Test tile operation with given input, multiples and dtype."""
        x = tf.constant(input_np, dtype=dtype)
        m = tf.constant(
            multiples, dtype=tf.int32 if multiples_dtype == np.int32 else tf.int64
        )

        def op_func(input_tensor):
            return tf.tile(input_tensor, m)

        if dtype == tf.bool:
            cpu_result = self._test_op_device_placement(op_func, [x], "/CPU:0")
            musa_result = self._test_op_device_placement(op_func, [x], "/device:MUSA:0")
            self.assertAllEqual(cpu_result.numpy(), musa_result.numpy())
        else:
            self._compare_cpu_musa_results(
                op_func, [x], dtype=dtype, rtol=rtol, atol=atol
            )

    # ---------------------------------------------------------------------------
    # Data type coverage
    # ---------------------------------------------------------------------------

    def testFloat32(self):
        """Basic 2D tile with float32."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self._test_tile(x_np, [2, 3], tf.float32)

    def testFloat16(self):
        """Basic 2D tile with float16."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        self._test_tile(x_np, [2, 3], tf.float16, rtol=1e-3, atol=1e-3)

    def testFloat64(self):
        """Basic 2D tile with float64."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        self._test_tile(x_np, [2, 3], tf.float64)

    def testInt32(self):
        """Basic 2D tile with int32."""
        x_np = np.array([[1, 2], [3, 4]], dtype=np.int32)
        self._test_tile(x_np, [2, 3], tf.int32)

    def testInt64(self):
        """Basic 2D tile with int64."""
        x_np = np.array([[1, 2], [3, 4]], dtype=np.int64)
        self._test_tile(x_np, [2, 3], tf.int64)

    def testBool(self):
        """Basic 2D tile with bool."""
        x_np = np.array([[True, False], [False, True]], dtype=np.bool_)
        self._test_tile(x_np, [2, 3], tf.bool)

    # ---------------------------------------------------------------------------
    # Tmultiples dtype coverage: int32 and int64
    # ---------------------------------------------------------------------------

    def testMultiplesInt64(self):
        """Tile with int64 multiples."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self._test_tile(x_np, [2, 3], tf.float32, multiples_dtype=np.int64)

    # ---------------------------------------------------------------------------
    # ndims coverage: 1-D through 5-D
    # ---------------------------------------------------------------------------

    def testTile1D(self):
        """1-D input."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self._test_tile(x_np, [4], tf.float32)

    def testTile2D(self):
        """2-D input with different multiples per axis."""
        x_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        self._test_tile(x_np, [3, 2], tf.float32)

    def testTile3D(self):
        """3-D input."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        self._test_tile(x_np, [2, 1, 3], tf.float32)

    def testTile4D(self):
        """4-D input (NHWC-style)."""
        x_np = np.random.uniform(-1, 1, size=[2, 4, 4, 3]).astype(np.float32)
        self._test_tile(x_np, [1, 2, 2, 1], tf.float32)

    def testTile5D(self):
        """5-D input."""
        x_np = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 2, 2)
        self._test_tile(x_np, [2, 1, 2, 3, 1], tf.float32)

    # ---------------------------------------------------------------------------
    # Edge cases
    # ---------------------------------------------------------------------------

    def testTileNoOp(self):
        """All multiples equal 1 — output is identical to input (no-op path)."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        self._test_tile(x_np, [1, 1], tf.float32)

    def testTileScalar(self):
        """0-D scalar input — should be forwarded as-is."""
        x_np = np.array(42.0, dtype=np.float32)
        x = tf.constant(x_np, dtype=tf.float32)

        def op_func(input_tensor):
            return tf.tile(input_tensor, [])

        self._compare_cpu_musa_results(op_func, [x], dtype=tf.float32)

    def testTileMixedMultiples(self):
        """Some dimensions tiled, others kept at 1."""
        x_np = np.random.randint(0, 10, size=[3, 1, 4]).astype(np.int32)
        self._test_tile(x_np, [1, 5, 1], tf.int32)

    def testTileSingleElementPerDim(self):
        """Input shape is all-ones — broadcasting-like behaviour."""
        x_np = np.array([[[2.5]]], dtype=np.float32)  # shape [1, 1, 1]
        self._test_tile(x_np, [4, 3, 2], tf.float32)

    def testTileLargeMultiples(self):
        """Large tile factor to stress memory copy paths."""
        x_np = np.array([1.0, 2.0], dtype=np.float32)
        self._test_tile(x_np, [512], tf.float32)

    def testTileAllTypes2D(self):
        """Sweep all registered dtypes over a single 2-D shape."""
        x_np_int = np.arange(1, 7, dtype=np.int32).reshape(2, 3)
        x_np_float = x_np_int.astype(np.float32)

        configs = [
            (tf.float32, x_np_float, {}),
            (tf.float16, x_np_float, {"rtol": 1e-3, "atol": 1e-3}),
            (tf.float64, x_np_float.astype(np.float64), {}),
            (tf.int32, x_np_int, {}),
            (tf.int64, x_np_int.astype(np.int64), {}),
            (tf.bool, (x_np_int % 2 == 0).astype(np.bool_), {}),
        ]
        for dtype, data, kw in configs:
            self._test_tile(data, [2, 3], dtype, **kw)


if __name__ == "__main__":
    tf.test.main()

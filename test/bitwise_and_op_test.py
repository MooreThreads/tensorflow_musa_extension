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

"""Tests for MUSA BitwiseAnd operator."""

import tensorflow as tf
import numpy as np

from musa_test_utils import MUSATestCase


class BitwiseAndOpTest(MUSATestCase):
  """Tests for MUSA BitwiseAnd operator."""

  ALL_DTYPES = [tf.int8, tf.int16, tf.int32, tf.int64,
                tf.uint8, tf.uint16, tf.uint32, tf.uint64]

  SIGNED_DTYPES = [tf.int8, tf.int16, tf.int32, tf.int64]

  UNSIGNED_DTYPES = [tf.uint8, tf.uint16, tf.uint32, tf.uint64]

  def _test_bitwise_and(self, shape, dtype=tf.int32):
    """Test bitwise and operation with given shape and dtype."""
    np_dtype = dtype.as_numpy_dtype
    info = np.iinfo(np_dtype)

    a_np = np.random.randint(info.min, info.max, size=shape, dtype=np_dtype)
    b_np = np.random.randint(info.min, info.max, size=shape, dtype=np_dtype)

    a = tf.constant(a_np, dtype=dtype)
    b = tf.constant(b_np, dtype=dtype)

    with tf.device('/CPU:0'):
      cpu_result = tf.bitwise.bitwise_and(a, b)

    with tf.device('/device:MUSA:0'):
      musa_result = tf.bitwise.bitwise_and(a, b)

    self.assertAllEqual(cpu_result.numpy(), musa_result.numpy())

  def _test_bitwise_and_with_values(self, a_np, b_np, dtype, expected_np=None):
    """Test bitwise and with explicit input values."""
    a = tf.constant(a_np, dtype=dtype)
    b = tf.constant(b_np, dtype=dtype)

    with tf.device('/CPU:0'):
      cpu_result = tf.bitwise.bitwise_and(a, b)

    with tf.device('/device:MUSA:0'):
      musa_result = tf.bitwise.bitwise_and(a, b)

    self.assertAllEqual(cpu_result.numpy(), musa_result.numpy())

    if expected_np is not None:
      self.assertAllEqual(musa_result.numpy(), expected_np)

  def _test_bitwise_and_broadcast(self, a_shape, b_shape, dtype=tf.int32):
    """Test bitwise and with broadcast between two different shapes."""
    np_dtype = dtype.as_numpy_dtype
    info = np.iinfo(np_dtype)

    a_np = np.random.randint(info.min, info.max, size=a_shape, dtype=np_dtype)
    b_np = np.random.randint(info.min, info.max, size=b_shape, dtype=np_dtype)

    expected = np.bitwise_and(a_np, b_np)
    self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)


  def testBitwiseAnd1D(self):
    """1D tensor bitwise and test."""
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and([16], dtype=dtype)

  def testBitwiseAnd2D(self):
    """2D tensor bitwise and test."""
    for dtype in [tf.int32, tf.int64, tf.uint32, tf.uint64]:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and([10, 5], dtype=dtype)

  def testBitwiseAnd3D(self):
    """3D tensor bitwise and test."""
    for dtype in [tf.int32, tf.int64]:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and([2, 3, 4], dtype=dtype)

  def testBitwiseAnd4D(self):
    """4D tensor bitwise and test."""
    for dtype in [tf.int32]:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and([2, 3, 4, 5], dtype=dtype)

  def testBitwiseAnd5D(self):
    """5D tensor bitwise and test."""
    self._test_bitwise_and([2, 2, 2, 3, 4], dtype=tf.int32)


  def testBitwiseAndAllZeros(self):
    """x & 0 == 0 for all x."""
    shape = [8]
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.random.randint(info.min, info.max, size=shape, dtype=np_dtype)
        b_np = np.zeros(shape, dtype=np_dtype)
        expected = np.zeros(shape, dtype=np_dtype)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)

  def testBitwiseAndAllOnes(self):
    """x & (-1) == x for signed types (all bits set)."""
    shape = [8]
    for dtype in self.SIGNED_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        a_np = np.random.randint(-100, 100, size=shape, dtype=np_dtype)
        b_np = np.full(shape, -1, dtype=np_dtype)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, a_np)

  def testBitwiseAndAllOnesUnsigned(self):
    """x & max_val == x for unsigned types (all bits set)."""
    shape = [8]
    for dtype in self.UNSIGNED_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.random.randint(0, info.max, size=shape, dtype=np_dtype)
        b_np = np.full(shape, info.max, dtype=np_dtype)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, a_np)

  def testBitwiseAndSelf(self):
    """x & x == x."""
    shape = [16]
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.random.randint(info.min, info.max, size=shape, dtype=np_dtype)
        self._test_bitwise_and_with_values(a_np, a_np, dtype, a_np)


  def testBitwiseAndMinMaxValues(self):
    """Test with min and max values of each dtype."""
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.array([info.min, info.max, info.min, info.max], dtype=np_dtype)
        b_np = np.array([info.min, info.max, info.max, info.min], dtype=np_dtype)
        expected = np.bitwise_and(a_np, b_np)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)

  def testBitwiseAndMinWithSelf(self):
    """min & min == min."""
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.array([info.min], dtype=np_dtype)
        self._test_bitwise_and_with_values(a_np, a_np, dtype, a_np)

  def testBitwiseAndMaxWithSelf(self):
    """max & max == max."""
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.array([info.max], dtype=np_dtype)
        self._test_bitwise_and_with_values(a_np, a_np, dtype, a_np)

  def testBitwiseAndMinAndMax(self):
    """Test min & max for signed types: sign bit extraction."""
    for dtype in self.SIGNED_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.array([info.min], dtype=np_dtype)
        b_np = np.array([info.max], dtype=np_dtype)
        expected = np.bitwise_and(a_np, b_np)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)


  def testBitwiseAndAlternatingBits(self):
    """0xAA & 0x55 == 0x00 (alternating bit patterns)."""
    for dtype in [tf.uint8, tf.int16, tf.uint16, tf.int32, tf.uint32]:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        a_np = np.array([0xAA, 0x55, 0xFF, 0x00], dtype=np_dtype)
        b_np = np.array([0x55, 0xAA, 0x0F, 0xFF], dtype=np_dtype)
        expected = np.bitwise_and(a_np, b_np)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)

  def testBitwiseAndPowerOfTwo(self):
    """Test with power-of-two values to check single-bit extraction."""
    np_dtype = np.int32
    a_np = np.array([255, 255, 255, 255, 255, 255, 255, 255], dtype=np_dtype)
    b_np = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np_dtype)
    expected = b_np.copy()
    self._test_bitwise_and_with_values(a_np, b_np, tf.int32, expected)

  def testBitwiseAndMaskLowBits(self):
    """Test masking lower N bits: x & ((1 << N) - 1)."""
    np_dtype = np.int32
    a_np = np.array([0x12345678, 0xABCDEF01, 0x7FFFFFFF, -1], dtype=np_dtype)
    mask_np = np.array([0xFF, 0xFF, 0xFF, 0xFF], dtype=np_dtype)
    expected = np.bitwise_and(a_np, mask_np)
    self._test_bitwise_and_with_values(a_np, mask_np, tf.int32, expected)

  def testBitwiseAndMaskHighBits(self):
    """Test masking higher bits: x & 0xFFFF0000."""
    np_dtype = np.int32
    a_np = np.array([0x12345678, 0xABCDEF01, 0x7FFFFFFF, -1], dtype=np_dtype)
    mask_np = np.full(4, np.int32(0xFFFF0000).view(np.int32), dtype=np_dtype)
    expected = np.bitwise_and(a_np, mask_np)
    self._test_bitwise_and_with_values(a_np, mask_np, tf.int32, expected)

  def testBitwiseAndNegativeValues(self):
    """Test with negative values for signed types (two's complement)."""
    for dtype in self.SIGNED_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        a_np = np.array([-1, -2, -128, 127, -1, 0], dtype=np_dtype)
        b_np = np.array([-1, -1, 127, -128, 0, -1], dtype=np_dtype)
        expected = np.bitwise_and(a_np, b_np)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)


  def testBitwiseAndCommutativity(self):
    """a & b == b & a."""
    shape = [32]
    for dtype in [tf.int32, tf.uint64]:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.random.randint(info.min, info.max, size=shape, dtype=np_dtype)
        b_np = np.random.randint(info.min, info.max, size=shape, dtype=np_dtype)

        a = tf.constant(a_np, dtype=dtype)
        b = tf.constant(b_np, dtype=dtype)

        with tf.device('/device:MUSA:0'):
          result_ab = tf.bitwise.bitwise_and(a, b)
          result_ba = tf.bitwise.bitwise_and(b, a)

        self.assertAllEqual(result_ab.numpy(), result_ba.numpy())

  def testBitwiseAndAssociativity(self):
    """(a & b) & c == a & (b & c)."""
    shape = [32]
    for dtype in [tf.int32, tf.int64]:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.random.randint(info.min, info.max, size=shape, dtype=np_dtype)
        b_np = np.random.randint(info.min, info.max, size=shape, dtype=np_dtype)
        c_np = np.random.randint(info.min, info.max, size=shape, dtype=np_dtype)

        a = tf.constant(a_np, dtype=dtype)
        b = tf.constant(b_np, dtype=dtype)
        c = tf.constant(c_np, dtype=dtype)

        with tf.device('/device:MUSA:0'):
          ab = tf.bitwise.bitwise_and(a, b)
          ab_c = tf.bitwise.bitwise_and(ab, c)
          bc = tf.bitwise.bitwise_and(b, c)
          a_bc = tf.bitwise.bitwise_and(a, bc)

        self.assertAllEqual(ab_c.numpy(), a_bc.numpy())


  def testBitwiseAndEmptyTensor(self):
    """Test with empty tensor (0 elements)."""
    for dtype in [tf.int32, tf.int64]:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        a_np = np.array([], dtype=np_dtype)
        b_np = np.array([], dtype=np_dtype)
        expected = np.array([], dtype=np_dtype)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)

  def testBitwiseAndScalar(self):
    """Test with scalar (0-D tensor)."""
    for dtype in [tf.int32, tf.uint8, tf.int64]:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        a_np = np.array(42, dtype=np_dtype)
        b_np = np.array(15, dtype=np_dtype)
        expected = np.bitwise_and(a_np, b_np)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)

  def testBitwiseAndSingleElement(self):
    """Test with single element tensor."""
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.array([info.max], dtype=np_dtype)
        b_np = np.array([info.min], dtype=np_dtype)
        expected = np.bitwise_and(a_np, b_np)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)

  def testBitwiseAndLargeTensor(self):
    """Bitwise and with larger tensors."""
    for dtype in [tf.int32, tf.int64]:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and([1000, 500], dtype=dtype)

  def testBitwiseAndVeryLarge1D(self):
    """Test with very large 1D tensor exceeding typical block size multiples."""
    self._test_bitwise_and([100003], dtype=tf.int32)


  def testBitwiseAndDifferentDtypes(self):
    """Test bitwise and with all supported integer data types."""
    shape = [5, 3]
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and(shape, dtype=dtype)


  def testBroadcastScalarAndTensor(self):
    """scalar & [N] => [N]."""
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        a_np = np.array(0xFF, dtype=np_dtype)
        b_np = np.array([0x0F, 0xF0, 0xAA, 0x55], dtype=np_dtype)
        expected = np.bitwise_and(a_np, b_np)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)

  def testBroadcastTensorAndScalar(self):
    """[N] & scalar => [N] (反向)."""
    for dtype in [tf.int32, tf.uint8]:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        a_np = np.array([0x0F, 0xF0, 0xAA, 0x55], dtype=np_dtype)
        b_np = np.array(0xFF, dtype=np_dtype)
        expected = np.bitwise_and(a_np, b_np)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)

  def testBroadcast1DTo2D(self):
    """[4] & [3, 4] => [3, 4]."""
    for dtype in [tf.int32, tf.int64, tf.uint32]:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and_broadcast([4], [3, 4], dtype=dtype)

  def testBroadcast2DTo1D(self):
    """[3, 4] & [4] => [3, 4] (反向)."""
    for dtype in [tf.int32, tf.int64]:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and_broadcast([3, 4], [4], dtype=dtype)

  def testBroadcastColumnAndRow(self):
    """[3, 1] & [1, 4] => [3, 4]."""
    for dtype in [tf.int32, tf.uint8, tf.int64]:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and_broadcast([3, 1], [1, 4], dtype=dtype)

  def testBroadcastRowAndColumn(self):
    """[1, 4] & [3, 1] => [3, 4] (反向)."""
    for dtype in [tf.int32, tf.uint32]:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and_broadcast([1, 4], [3, 1], dtype=dtype)

  def testBroadcast1DTo3D(self):
    """[4] & [2, 3, 4] => [2, 3, 4]."""
    for dtype in [tf.int32, tf.int64]:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and_broadcast([4], [2, 3, 4], dtype=dtype)

  def testBroadcast3DMixed(self):
    """[2, 1, 4] & [3, 4] => [2, 3, 4]."""
    for dtype in [tf.int32, tf.uint16]:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and_broadcast([2, 1, 4], [3, 4], dtype=dtype)

  def testBroadcast3DAllDims(self):
    """[2, 1, 4] & [1, 3, 1] => [2, 3, 4]."""
    self._test_bitwise_and_broadcast([2, 1, 4], [1, 3, 1], dtype=tf.int32)

  def testBroadcast4DComplex(self):
    """[2, 1, 3, 1] & [1, 4, 1, 5] => [2, 4, 3, 5]."""
    self._test_bitwise_and_broadcast([2, 1, 3, 1], [1, 4, 1, 5], dtype=tf.int32)

  def testBroadcast4DPartial(self):
    """[2, 3, 1, 5] & [2, 1, 4, 5] => [2, 3, 4, 5]."""
    self._test_bitwise_and_broadcast([2, 3, 1, 5], [2, 1, 4, 5], dtype=tf.int32)

  def testBroadcast5D(self):
    """[1, 2, 1, 3, 1] & [2, 1, 4, 1, 5] => [2, 2, 4, 3, 5]."""
    self._test_bitwise_and_broadcast([1, 2, 1, 3, 1], [2, 1, 4, 1, 5],
                                     dtype=tf.int32)

  def testBroadcastScalarAndHighDim(self):
    """scalar & [2, 3, 4] => [2, 3, 4]."""
    np_dtype = np.int32
    a_np = np.array(0x0F0F0F0F, dtype=np_dtype)
    b_np = np.random.randint(-1000, 1000, size=[2, 3, 4], dtype=np_dtype)
    expected = np.bitwise_and(a_np, b_np)
    self._test_bitwise_and_with_values(a_np, b_np, tf.int32, expected)

  def testBroadcastOneDimIsOne(self):
    """[1] & [8] => [8] (单维度为1的广播)."""
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and_broadcast([1], [8], dtype=dtype)

  def testBroadcastBothHaveDimOne(self):
    """[1, 4] & [3, 1] 两边都有维度1."""
    for dtype in [tf.int32, tf.int64, tf.uint8]:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and_broadcast([1, 4], [3, 1], dtype=dtype)


  def testBroadcastWithAllZeros(self):
    """广播 x & 0: [3, 4] & [1] where b=0 => all zeros."""
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.random.randint(info.min, info.max, size=[3, 4], dtype=np_dtype)
        b_np = np.array([0], dtype=np_dtype)
        expected = np.zeros([3, 4], dtype=np_dtype)
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)

  def testBroadcastWithAllOnes(self):
    """广播 x & (-1): [3, 4] & [1] where b=-1 => x."""
    for dtype in self.SIGNED_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        a_np = np.random.randint(-100, 100, size=[3, 4], dtype=np_dtype)
        b_np = np.array([-1], dtype=np_dtype)
        expected = a_np.copy()
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)

  def testBroadcastWithMinMaxValues(self):
    """广播中使用 min/max 边界值."""
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.array([[info.min], [info.max], [0]], dtype=np_dtype)  # [3, 1]
        b_np = np.array([[info.min, info.max, 0]], dtype=np_dtype)      # [1, 3]
        expected = np.bitwise_and(a_np, b_np)  # => [3, 3]
        self._test_bitwise_and_with_values(a_np, b_np, dtype, expected)


  def testBroadcastCommutativity(self):
    """广播下交换律: a & b == b & a."""
    for dtype in [tf.int32, tf.uint32, tf.int64]:
      with self.subTest(dtype=dtype):
        np_dtype = dtype.as_numpy_dtype
        info = np.iinfo(np_dtype)
        a_np = np.random.randint(info.min, info.max, size=[3, 1], dtype=np_dtype)
        b_np = np.random.randint(info.min, info.max, size=[1, 4], dtype=np_dtype)

        a = tf.constant(a_np, dtype=dtype)
        b = tf.constant(b_np, dtype=dtype)

        with tf.device('/device:MUSA:0'):
          result_ab = tf.bitwise.bitwise_and(a, b)
          result_ba = tf.bitwise.bitwise_and(b, a)

        self.assertAllEqual(result_ab.numpy(), result_ba.numpy())

  def testBroadcastAssociativity(self):
    """广播下结合律: (a & b) & c == a & (b & c)."""
    np_dtype = np.int32
    a_np = np.random.randint(-1000, 1000, size=[2, 1, 4], dtype=np_dtype)
    b_np = np.random.randint(-1000, 1000, size=[1, 3, 1], dtype=np_dtype)
    c_np = np.random.randint(-1000, 1000, size=[2, 3, 4], dtype=np_dtype)

    a = tf.constant(a_np, dtype=tf.int32)
    b = tf.constant(b_np, dtype=tf.int32)
    c = tf.constant(c_np, dtype=tf.int32)

    with tf.device('/device:MUSA:0'):
      ab = tf.bitwise.bitwise_and(a, b)
      ab_c = tf.bitwise.bitwise_and(ab, c)
      bc = tf.bitwise.bitwise_and(b, c)
      a_bc = tf.bitwise.bitwise_and(a, bc)

    self.assertAllEqual(ab_c.numpy(), a_bc.numpy())

  def testBroadcastLargeTensor(self):
    """大张量广播: [256, 1] & [1, 512] => [256, 512]."""
    self._test_bitwise_and_broadcast([256, 1], [1, 512], dtype=tf.int32)

  def testBroadcastLargeTensor3D(self):
    """大张量3D广播: [32, 1, 64] & [1, 48, 1] => [32, 48, 64]."""
    self._test_bitwise_and_broadcast([32, 1, 64], [1, 48, 1], dtype=tf.int32)

  def testBroadcastLargeWithScalar(self):
    """大张量与标量广播."""
    np_dtype = np.int32
    a_np = np.random.randint(-10000, 10000, size=[1000, 500], dtype=np_dtype)
    b_np = np.array(0xFF00FF, dtype=np_dtype)
    expected = np.bitwise_and(a_np, b_np)
    self._test_bitwise_and_with_values(a_np, b_np, tf.int32, expected)

  def testBroadcastAllDtypes(self):
    """所有类型的广播测试: [3, 1] & [1, 4]."""
    for dtype in self.ALL_DTYPES:
      with self.subTest(dtype=dtype):
        self._test_bitwise_and_broadcast([3, 1], [1, 4], dtype=dtype)

  def testBroadcastMaskColumn(self):
    """用广播实现列掩码: [4, 1] (mask) & [1, 8] (data) => [4, 8]."""
    np_dtype = np.int32
    masks = np.array([[0xFF], [0xFF00], [0xFF0000], [0x0F0F0F0F]], dtype=np_dtype)
    data = np.random.randint(-10000, 10000, size=[1, 8], dtype=np_dtype)
    expected = np.bitwise_and(masks, data)
    self._test_bitwise_and_with_values(masks, data, tf.int32, expected)

  def testBroadcastAlternatingBits(self):
    """交替位模式的广播: scalar 0xAA & [N]."""
    np_dtype = np.uint8
    a_np = np.array(0xAA, dtype=np_dtype)
    b_np = np.array([0xFF, 0x55, 0xAA, 0x00, 0x0F, 0xF0], dtype=np_dtype)
    expected = np.bitwise_and(a_np, b_np)
    self._test_bitwise_and_with_values(a_np, b_np, tf.uint8, expected)

  def testBroadcastSelfWithDifferentShape(self):
    """相同数据但不同 shape 的广播: [3, 1] & [1, 3] (outer product style)."""
    np_dtype = np.int32
    vals = np.array([0xFF, 0x0F, 0xF0], dtype=np_dtype)
    a_np = vals.reshape(3, 1)
    b_np = vals.reshape(1, 3)
    expected = np.bitwise_and(a_np, b_np)
    self._test_bitwise_and_with_values(a_np, b_np, tf.int32, expected)


if __name__ == "__main__":
  tf.test.main()
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

"""Tests for MUSA StringToHashBucketFast operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class StringToHashBucketFastOpTest(MUSATestCase):
    """Tests for MUSA StringToHashBucketFast operator."""

    def _test_hash_bucket(self, strings, num_buckets, dtype=tf.int64):
        """Test StringToHashBucketFast with given strings and num_buckets."""
        # Create input tensor
        input_tensor = tf.constant(strings, dtype=tf.string)

        # Test on CPU
        with tf.device('/CPU:0'):
            cpu_result = tf.strings.to_hash_bucket_fast(input_tensor, num_buckets)

        # Test on MUSA
        with tf.device('/device:MUSA:0'):
            musa_result = tf.strings.to_hash_bucket_fast(input_tensor, num_buckets)

        # Compare results
        self.assertAllClose(cpu_result.numpy(), musa_result.numpy())

        # Verify all hash values are in valid range [0, num_buckets)
        musa_np = musa_result.numpy()
        self.assertTrue(np.all(musa_np >= 0))
        self.assertTrue(np.all(musa_np < num_buckets))

    def testBasic(self):
        """Test basic StringToHashBucketFast operation."""
        strings = ['hello', 'world', 'tensorflow', 'musa']
        num_buckets = 100
        self._test_hash_bucket(strings, num_buckets)

    def testSingleElement(self):
        """Test with single element tensor."""
        strings = ['test_string']
        num_buckets = 50
        self._test_hash_bucket(strings, num_buckets)

    def testEmptyTensor(self):
        """Test with empty tensor."""
        strings = []
        num_buckets = 10

        input_tensor = tf.constant(strings, dtype=tf.string)

        # Test on CPU
        with tf.device('/CPU:0'):
            cpu_result = tf.strings.to_hash_bucket_fast(input_tensor, num_buckets)

        # Test on MUSA
        with tf.device('/device:MUSA:0'):
            musa_result = tf.strings.to_hash_bucket_fast(input_tensor, num_buckets)

        # Both should be empty tensors with same shape
        self.assertEqual(cpu_result.shape, musa_result.shape)
        self.assertEqual(cpu_result.shape[0], 0)

    def testLargeBatch(self):
        """Test with large batch of strings."""
        # Generate a large batch of random strings
        np.random.seed(42)
        strings = []
        for i in range(10000):
            # Create random strings of varying lengths
            length = np.random.randint(1, 100)
            s = ''.join(chr(np.random.randint(32, 127)) for _ in range(length))
            strings.append(s)

        num_buckets = 1000
        self._test_hash_bucket(strings, num_buckets)

    def testMultiDimensional(self):
        """Test with multi-dimensional string tensor."""
        strings = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]
        num_buckets = 100
        self._test_hash_bucket(strings, num_buckets)

    def testVariousBucketSizes(self):
        """Test with various bucket sizes."""
        strings = ['test1', 'test2', 'test3', 'test4', 'test5']

        # Test with different bucket sizes
        bucket_sizes = [1, 2, 10, 100, 1000, 10000]
        for num_buckets in bucket_sizes:
            self._test_hash_bucket(strings, num_buckets)

    def testSpecialCharacters(self):
        """Test with special characters in strings."""
        strings = [
            'hello world',
            'hello\tworld',
            'hello\nworld',
            'hello\\world',
            'hello"world',
            "hello'world",
            'hello!@#$%^&*()world',
            '',  # Empty string
            ' ',  # Single space
            '   ',  # Multiple spaces
        ]
        num_buckets = 100
        self._test_hash_bucket(strings, num_buckets)

    def testUnicodeStrings(self):
        """Test with unicode strings."""
        strings = [
            'hello',
            '你好',
            'こんにちは',
            '안녕하세요',
            'مرحبا',
            'שלום',
            'Привет',
            '🎉🎊🎁',  # Emojis
        ]
        num_buckets = 100
        self._test_hash_bucket(strings, num_buckets)

    def testDuplicateStrings(self):
        """Test that duplicate strings produce same hash values."""
        strings = ['hello', 'world', 'hello', 'tensorflow', 'world', 'hello']
        num_buckets = 100

        input_tensor = tf.constant(strings, dtype=tf.string)

        # Test on MUSA
        with tf.device('/device:MUSA:0'):
            musa_result = tf.strings.to_hash_bucket_fast(input_tensor, num_buckets)

        musa_np = musa_result.numpy()

        # Verify duplicates have same hash values
        self.assertEqual(musa_np[0], musa_np[2])  # 'hello' at index 0 and 2
        self.assertEqual(musa_np[0], musa_np[5])  # 'hello' at index 0 and 5
        self.assertEqual(musa_np[1], musa_np[4])  # 'world' at index 1 and 4

    def testDeterministic(self):
        """Test that hash function is deterministic."""
        strings = ['test1', 'test2', 'test3']
        num_buckets = 100

        input_tensor = tf.constant(strings, dtype=tf.string)

        # Run multiple times on MUSA
        results = []
        for _ in range(5):
            with tf.device('/device:MUSA:0'):
                result = tf.strings.to_hash_bucket_fast(input_tensor, num_buckets)
            results.append(result.numpy())

        # All results should be identical
        for i in range(1, len(results)):
            self.assertAllClose(results[0], results[i])

    def testLongStrings(self):
        """Test with very long strings."""
        # Create long strings
        long_string_1 = 'a' * 10000
        long_string_2 = 'b' * 10000
        long_string_3 = 'test' * 2500  # 10000 characters

        strings = [long_string_1, long_string_2, long_string_3]
        num_buckets = 100
        self._test_hash_bucket(strings, num_buckets)

    def testBucketBoundary(self):
        """Test that hash values are within bucket boundaries."""
        strings = ['test'] * 100
        num_buckets = 10

        input_tensor = tf.constant(strings, dtype=tf.string)

        with tf.device('/device:MUSA:0'):
            musa_result = tf.strings.to_hash_bucket_fast(input_tensor, num_buckets)

        musa_np = musa_result.numpy()

        # All values should be in [0, num_buckets)
        self.assertTrue(np.all(musa_np >= 0))
        self.assertTrue(np.all(musa_np < num_buckets))

        # All values should be the same (same input string)
        self.assertTrue(np.all(musa_np == musa_np[0]))


if __name__ == "__main__":
    tf.test.main()
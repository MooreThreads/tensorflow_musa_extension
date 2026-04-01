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

"""Tests for MUSA ResourceApplyNadam operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase, load_musa_plugin
from tensorflow.python.ops import gen_training_ops

musa_ops = tf.load_op_library(load_musa_plugin())

class ResourceApplyNadamOpTest(MUSATestCase):
    """Tests for MUSA ResourceApplyNadam operator."""

    def _get_nadam_op(self):
        if hasattr(tf.raw_ops, "ResourceApplyNadam"):
            return tf.raw_ops.ResourceApplyNadam
        try:
            from musa_test_utils import musa_ops
            if hasattr(musa_ops, "ResourceApplyNadam"):
                return musa_ops.ResourceApplyNadam
        except ImportError:
            pass
        try:
            from tensorflow.python.ops import gen_training_ops
            if hasattr(gen_training_ops, "resource_apply_nadam"):
                return gen_training_ops.resource_apply_nadam
        except ImportError:
            pass
        return None

    def _test_resource_apply_nadam(self, shape, dtype, rtol=1e-5, atol=1e-5):
        np_dtype = np.float32 if dtype in [tf.bfloat16, tf.half] else dtype.as_numpy_dtype

        var_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)
        m_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)
        v_np = np.random.uniform(0.1, 1, size=shape).astype(np_dtype)
        grad_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)

        beta1_power_val = 0.9
        beta2_power_val = 0.99
        lr_val = 0.01
        beta1_val = 0.9
        beta2_val = 0.99
        epsilon_val = 1e-8

        def run_nadam(device):
            with tf.device(device):
                var = tf.Variable(var_np, dtype=dtype)
                m = tf.Variable(m_np, dtype=dtype)
                v = tf.Variable(v_np, dtype=dtype)
                beta1_power = tf.constant(beta1_power_val, dtype=dtype)
                beta2_power = tf.constant(beta2_power_val, dtype=dtype)
                lr = tf.constant(lr_val, dtype=dtype)
                beta1 = tf.constant(beta1_val, dtype=dtype)
                beta2 = tf.constant(beta2_val, dtype=dtype)
                epsilon = tf.constant(epsilon_val, dtype=dtype)
                grad = tf.constant(grad_np, dtype=dtype)

                if "cpu" in device:
                    # Nadam algorithm implementation in TensorFlow
                    # The formula matches TF's ResourceApplyNadam:
                    # m = beta1 * m + (1 - beta1) * grad
                    # v = beta2 * v + (1 - beta2) * grad^2
                    # m_hat = (beta1 * m + (1 - beta1) * grad) / (1 - beta1_power)
                    # v_hat = v / (1 - beta2_power)
                    # var = var - lr * m_hat / (sqrt(v_hat) + epsilon)
                    m_new = beta1 * m + (1 - beta1) * grad
                    v_new = beta2 * v + (1 - beta2) * tf.square(grad)
                    
                    m_hat = (beta1 * m_new + (1 - beta1) * grad) / (1 - beta1_power)
                    v_hat = v_new / (1 - beta2_power)
                    
                    var_new = var - lr * m_hat / (tf.sqrt(v_hat) + epsilon)
                    
                    var.assign(var_new)
                    m.assign(m_new)
                    v.assign(v_new)
                else:
                    op_func = musa_ops.ResourceApplyNadam
                    op_func(
                        var=var.handle, m=m.handle, v=v.handle,
                        beta1_power=beta1_power, beta2_power=beta2_power,
                        lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, grad=grad
                    )
                return var.read_value().numpy(), m.read_value().numpy(), v.read_value().numpy()

        cpu_var, cpu_m, cpu_v = run_nadam("/cpu:0")
        musa_var, musa_m, musa_v = run_nadam("/device:MUSA:0")

        self.assertAllClose(cpu_var, musa_var, rtol=rtol, atol=atol)
        self.assertAllClose(cpu_m, musa_m, rtol=rtol, atol=atol)
        self.assertAllClose(cpu_v, musa_v, rtol=rtol, atol=atol)

    def testResourceApplyNadamBasic(self):
        """Test Nadam with basic shape and float32."""
        self._test_resource_apply_nadam([128, 128], tf.float32)

    def testResourceApplyNadamHalf(self):
        """Test Nadam with float16."""
        self._test_resource_apply_nadam([64, 64], tf.half, rtol=1e-3, atol=1e-3)

    def testResourceApplyNadamLarge(self):
        """Test Nadam with large tensor."""
        self._test_resource_apply_nadam([1024, 1024], tf.float32)

    def testResourceApplyNadamDouble(self):
        """Test Nadam with double."""
        self._test_resource_apply_nadam([16, 16], tf.double)

    def testResourceApplyNadamSmall(self):
        """Test Nadam with small 1D tensor."""
        self._test_resource_apply_nadam([7], tf.float32)

if __name__ == "__main__":
    tf.test.main()

# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================
#
# Regression note:
# This test originally exposed two independent issues on the MUSA resource
# variable path.
# 1. ResourceApplyAdam must follow TensorFlow-compatible resource
#    copy-on-write/update semantics, otherwise graph-mode updates can corrupt
#    the backing var/m/v tensors.
# 2. The plugin must be built with -DNDEBUG to match the pip TensorFlow wheel.
#    Mixing compile-time DCHECK semantics between the plugin and
#    libtensorflow_framework can trigger false refcount.h:90 aborts during
#    Session.close() even when the operator result is numerically correct.
# Keep this test in graph mode so it guards both the Adam update result and the
# session teardown path.

"""Tests for MUSA ResourceApplyAdam operator."""

import numpy as np
import tensorflow as tf

# 引入工具类 (确保该文件在 PYTHONPATH 或当前目录下)
from musa_test_utils import MUSATestCase


class ResourceApplyAdamTest(MUSATestCase):
  """Tests for MUSA ResourceApplyAdam operator."""

  def _run_resource_apply_adam(self, device, init_var, init_m, init_v, grad_val):
    """Run one ResourceApplyAdam update in graph mode on the requested device."""
    graph = tf.Graph()
    with graph.as_default():
      with tf.device(device):
        var = tf.Variable(init_var, dtype=tf.float32, name="var")
        m = tf.Variable(init_m, dtype=tf.float32, name="m")
        v = tf.Variable(init_v, dtype=tf.float32, name="v")
        grad = tf.constant(grad_val, dtype=tf.float32, name="grad")

      # Resource handles and scalar hyper-parameters stay on host memory.
      with tf.device("/CPU:0"):
        beta1_power = tf.constant(0.9, dtype=tf.float32, name="beta1_power")
        beta2_power = tf.constant(0.999, dtype=tf.float32, name="beta2_power")
        lr = tf.constant(0.01, dtype=tf.float32, name="lr")
        beta1 = tf.constant(0.9, dtype=tf.float32, name="beta1")
        beta2 = tf.constant(0.999, dtype=tf.float32, name="beta2")
        epsilon = tf.constant(1e-8, dtype=tf.float32, name="epsilon")

      update = tf.raw_ops.ResourceApplyAdam(
          var=var.handle,
          m=m.handle,
          v=v.handle,
          beta1_power=beta1_power,
          beta2_power=beta2_power,
          lr=lr,
          beta1=beta1,
          beta2=beta2,
          epsilon=epsilon,
          grad=grad,
          use_locking=False,
          use_nesterov=False)

      with tf.control_dependencies([update]):
        read_var = tf.identity(var.read_value(), name="updated_var")

      init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      return sess.run(read_var)

if __name__ == "__main__":
  tf.test.main()

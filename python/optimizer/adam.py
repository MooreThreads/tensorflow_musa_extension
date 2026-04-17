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

"""
MUSA-accelerated Adam optimizer.

This optimizer uses the fused ResourceApplyAdam kernel for improved performance
on MUSA GPUs. The fused kernel combines multiple operations (moment updates,
velocity updates, bias correction, and variable updates) into a single kernel,
reducing memory bandwidth and kernel launch overhead.

Example usage:
    import tensorflow as tf
    import tensorflow_musa as tf_musa

    # Use MUSA-accelerated Adam optimizer
    optimizer = tf_musa.optimizer.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='mse')
    model.fit(x, y)
"""

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class Adam(optimizer_v2.OptimizerV2):
    """MUSA-accelerated Adam optimizer using fused kernel.

    Implements the Adam algorithm with bias correction as described in
    "Adam: A Method for Stochastic Optimization" (Kingma et al., 2015).

    The update rule is:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g
        v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
        lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
        var = var - lr_t * m_t / (sqrt(v_t) + epsilon)

    This optimizer uses the fused ResourceApplyAdam kernel registered by
    the MUSA plugin, providing better performance on MUSA GPUs compared
    to the decomposed Adam implementation in TensorFlow.

    Args:
        learning_rate: A float, a `LearningRateSchedule` instance, or a callable
            that takes no arguments and returns the learning rate. Defaults to 0.001.
        beta_1: A float value. The exponential decay rate for the 1st moment
            estimates. Defaults to 0.9.
        beta_2: A float value. The exponential decay rate for the 2nd moment
            estimates. Defaults to 0.999.
        epsilon: A small float for numerical stability. Defaults to 1e-7.
        amsgrad: Whether to apply AMSGrad variant of Adam. Not currently
            supported in fused kernel. Defaults to False.
        name: Optional name for the operations created when applying gradients.
            Defaults to "AdamMUSA".
        **kwargs: Additional keyword arguments. Allowed to be {`clipnorm`,
            `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
            norm; `clipvalue` is clip gradients by value, `decay` is
            included for backward compatibility to allow time inverse
            decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        name="AdamMUSA",
        **kwargs
    ):
        """Initialize Adam optimizer."""
        super(Adam, self).__init__(name, **kwargs)

        if amsgrad:
            raise NotImplementedError(
                "AMSGrad variant is not supported in the fused MUSA Adam kernel. "
                "Use tf.keras.optimizers.Adam for AMSGrad support."
            )

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("epsilon", epsilon)

        # Beta powers track the iteration for bias correction
        self._beta1_power = None
        self._beta2_power = None

    def _create_slots(self, var_list):
        """Create slot variables for Adam.

        For each variable in var_list, creates two slot variables:
        - m: First moment estimates (moving average of gradients)
        - v: Second moment estimates (moving average of squared gradients)
        """
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")

    def _create_hypers(self):
        """Create hyper variables."""
        self._beta1_power = self.add_weight(
            name="beta1_power",
            shape=(),
            dtype=tf.float32,
            trainable=False,
            initializer=tf.constant_initializer(self._beta_1),
        )
        self._beta2_power = self.add_weight(
            name="beta2_power",
            shape=(),
            dtype=tf.float32,
            trainable=False,
            initializer=tf.constant_initializer(self._beta_2),
        )

    def _prepare(self, var_list):
        """Prepare hyper values before applying gradients."""
        # Create hyper variables if not already created
        if self._beta1_power is None:
            self._create_hypers()

        return {
            "beta1_power": self._beta1_power,
            "beta2_power": self._beta2_power,
            "lr": self._prepare_learning_rate(),
            "beta1": math_ops.cast(self._beta_1, var_list[0].dtype),
            "beta2": math_ops.cast(self._beta_2, var_list[0].dtype),
            "epsilon": math_ops.cast(self._epsilon, var_list[0].dtype),
        }

    def _resource_apply_dense(self, grad, var, apply_state=None):
        """Apply gradient update using fused ResourceApplyAdam kernel.

        This method dispatches to the MUSA-registered ResourceApplyAdam kernel
        when the variable is placed on a MUSA device.

        Args:
            grad: A tensor representing the gradient.
            var: A resource variable to update.
            apply_state: A dict containing hyperparameter values.

        Returns:
            An operation that updates the variable.
        """
        if apply_state is None:
            apply_state = self._prepare([var])

        var_dtype = var.dtype.base_dtype
        coefficients = apply_state.get((var_dtype, None), apply_state)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Cast coefficients to match variable dtype
        beta1_power = math_ops.cast(coefficients["beta1_power"], var_dtype)
        beta2_power = math_ops.cast(coefficients["beta2_power"], var_dtype)
        lr = math_ops.cast(coefficients["lr"], var_dtype)
        beta1 = math_ops.cast(coefficients["beta1"], var_dtype)
        beta2 = math_ops.cast(coefficients["beta2"], var_dtype)
        epsilon = math_ops.cast(coefficients["epsilon"], var_dtype)

        # Use the fused ResourceApplyAdam operation
        # This dispatches to the MUSA kernel when on MUSA device
        return tf.raw_ops.ResourceApplyAdam(
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
            use_locking=self._use_locking,
            use_nesterov=False,
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        """Apply sparse gradient update using fused MusaResourceSparseApplyAdam kernel.

        This method uses the custom MUSA sparse Adam kernel for efficient
        embedding updates on MUSA GPUs.

        Args:
            grad: A tensor representing the gradient values.
            var: A resource variable to update.
            indices: A tensor representing the indices for sparse update.
            apply_state: A dict containing hyperparameter values.

        Returns:
            An operation that updates the variable.
        """
        from .._loader import get_musa_ops_module

        musa_ops = get_musa_ops_module()

        # Check if MUSA sparse kernel is available
        if musa_ops is None or not hasattr(musa_ops, 'musa_resource_sparse_apply_adam'):
            # Fallback to densifying gradient if sparse kernel not available
            dense_grad = tf.IndexedSlices(grad, indices, tf.shape(var))
            dense_grad_tensor = tf.convert_to_tensor(dense_grad)
            return self._resource_apply_dense(dense_grad_tensor, var, apply_state)

        if apply_state is None:
            apply_state = self._prepare([var])

        var_dtype = var.dtype.base_dtype
        coefficients = apply_state.get((var_dtype, None), apply_state)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Compute bias-corrected learning rate
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = math_ops.cast(self._beta_1, var_dtype)
        beta_2_t = math_ops.cast(self._beta_2, var_dtype)
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)

        lr = math_ops.cast(coefficients["lr"], var_dtype)
        epsilon = math_ops.cast(self._epsilon, var_dtype)

        # Use the fused MusaResourceSparseApplyAdam kernel
        return musa_ops.musa_resource_sparse_apply_adam(
            var=var.handle,
            m=m.handle,
            v=v.handle,
            beta1_power=beta_1_power,
            beta2_power=beta_2_power,
            lr=lr,
            beta1=beta_1_t,
            beta2=beta_2_t,
            epsilon=epsilon,
            grad=grad,
            indices=indices,
            use_locking=self._use_locking)

    def _finish(self, update_ops, var_list):
        """Finish the update by updating beta powers."""
        if self._beta1_power is None:
            return update_ops

        # Update beta powers for next iteration
        beta1_update = state_ops.assign(
            self._beta1_power,
            self._beta1_power * self._beta_1,
            use_locking=self._use_locking,
        )
        beta2_update = state_ops.assign(
            self._beta2_power,
            self._beta2_power * self._beta_2,
            use_locking=self._use_locking,
        )

        return update_ops + [beta1_update, beta2_update]

    def get_config(self):
        """Get optimizer configuration."""
        config = super(Adam, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self._epsilon,
        })
        return config

    def from_config(cls, config):
        """Create optimizer from configuration."""
        return cls(**config)

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

"""Monkey patch utilities for TensorFlow Adam optimizer.

This module provides functionality to patch tf.keras.optimizers.Adam to use
MUSA fused kernels for both dense and sparse gradient updates.
"""

import tensorflow as tf

# Store original methods for restoration
_original_methods = {}


def _musa_resource_apply_sparse(self, grad, var, indices, apply_state=None):
    """Use fused MusaResourceSparseApplyAdam kernel on MUSA device.

    This method replaces the default TensorFlow Adam's _resource_apply_sparse
    which uses multiple ops (assign, scatter_add, etc.) with a single fused
    kernel for better performance on MUSA GPUs.

    Args:
        self: The Adam optimizer instance.
        grad: A tensor representing the sparse gradient values.
        var: A resource variable to update.
        indices: A tensor representing the indices for sparse update.
        apply_state: A dict containing hyperparameter values.

    Returns:
        An operation that updates the variable.
    """
    from ._loader import get_musa_ops_module

    musa_ops = get_musa_ops_module()
    if musa_ops is None or not hasattr(musa_ops, 'musa_resource_sparse_apply_adam'):
        # Fallback to original implementation if MUSA op not available
        return _original_methods['Adam']['_resource_apply_sparse'](self, grad, var, indices, apply_state)

    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    # Get hyperparameters (same as dense version)
    local_step = tf.cast(self.iterations + 1, var_dtype)
    beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = tf.pow(beta_1_t, local_step)
    beta_2_power = tf.pow(beta_2_t, local_step)

    # Compute bias-corrected learning rate
    lr = coefficients['lr_t'] * (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power))
    epsilon = tf.convert_to_tensor(self.epsilon or 1e-7, var_dtype)

    # Call our custom fused kernel via the ops module
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


def _musa_resource_apply_dense(self, grad, var, apply_state=None):
    """Use fused ResourceApplyAdam kernel on MUSA device.

    This method replaces the default TensorFlow Adam's _resource_apply_dense
    to use the fused MUSA kernel.

    Args:
        self: The Adam optimizer instance.
        grad: A tensor representing the dense gradient.
        var: A resource variable to update.
        apply_state: A dict containing hyperparameter values.

    Returns:
        An operation that updates the variable.
    """
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    # Get hyperparameters
    local_step = tf.cast(self.iterations + 1, var_dtype)
    beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = tf.pow(beta_1_t, local_step)
    beta_2_power = tf.pow(beta_2_t, local_step)

    lr = coefficients['lr_t'] * (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power))
    epsilon = tf.convert_to_tensor(self.epsilon or 1e-7, var_dtype)

    # Use the fused ResourceApplyAdam operation
    # This dispatches to the MUSA kernel when on MUSA device
    return tf.raw_ops.ResourceApplyAdam(
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
        use_locking=self._use_locking)


def patch_keras_adam():
    """Patch tf.keras.optimizers.Adam to use MUSA kernels.

    After patching:
    - _resource_apply_dense uses fused ResourceApplyAdam kernel
    - _resource_apply_sparse uses fused MusaResourceSparseApplyAdam kernel

    This provides significant performance improvement for training models
    with embedding layers on MUSA GPUs.
    """
    global _original_methods

    adam_class = tf.keras.optimizers.Adam

    # Store original methods
    _original_methods['Adam'] = {
        '_resource_apply_dense': adam_class._resource_apply_dense,
        '_resource_apply_sparse': adam_class._resource_apply_sparse,
    }

    # Apply patches
    adam_class._resource_apply_dense = _musa_resource_apply_dense
    adam_class._resource_apply_sparse = _musa_resource_apply_sparse

    # Also patch NonFusedAdam if it exists
    try:
        non_fused_adam_class = tf.keras.optimizers.NonFusedAdam
        _original_methods['NonFusedAdam'] = {
            '_resource_apply_dense': non_fused_adam_class._resource_apply_dense,
            '_resource_apply_sparse': non_fused_adam_class._resource_apply_sparse,
        }
        non_fused_adam_class._resource_apply_dense = _musa_resource_apply_dense
        non_fused_adam_class._resource_apply_sparse = _musa_resource_apply_sparse
    except AttributeError:
        pass  # NonFusedAdam may not exist in all TF versions

    print("MUSA Adam optimizer patch applied successfully.")


def unpatch_keras_adam():
    """Restore original Adam optimizer methods.

    Removes the MUSA patches and restores the default TensorFlow implementation.
    """
    global _original_methods

    if 'Adam' in _original_methods:
        adam_class = tf.keras.optimizers.Adam
        original = _original_methods['Adam']
        if '_resource_apply_dense' in original:
            adam_class._resource_apply_dense = original['_resource_apply_dense']
        if '_resource_apply_sparse' in original:
            adam_class._resource_apply_sparse = original['_resource_apply_sparse']

    if 'NonFusedAdam' in _original_methods:
        try:
            non_fused_adam_class = tf.keras.optimizers.NonFusedAdam
            original = _original_methods['NonFusedAdam']
            if '_resource_apply_dense' in original:
                non_fused_adam_class._resource_apply_dense = original['_resource_apply_dense']
            if '_resource_apply_sparse' in original:
                non_fused_adam_class._resource_apply_sparse = original['_resource_apply_sparse']
        except AttributeError:
            pass

    _original_methods.clear()
    print("MUSA Adam optimizer patch removed.")


def is_adam_patched():
    """Check if Adam optimizer has been patched with MUSA kernels.

    Returns:
        bool: True if patched, False otherwise.
    """
    return 'Adam' in _original_methods

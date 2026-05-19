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

"""Optimizers tailored for MUSA mixed-precision (bf16/fp16) training.

For mixed-precision training (Keras ``mixed_bfloat16`` / ``mixed_float16``
policy or any equivalent setup) the canonical recipe is:

  * variables (``var``) stay in fp32,
  * optimizer slot variables (``m``, ``v``) stay in fp32,
  * the forward/backward pass produces a low-precision gradient,
  * the optimizer applies the bf16/fp16 gradient to the fp32 state with the
    gradient promoted to fp32 only at the moment of use.

Stock TensorFlow 2.6.1's :class:`tf.keras.optimizers.Adam` already arranges
the first three points, but it then emits an explicit ``Cast`` from bf16/fp16
to fp32 in front of ``ResourceApplyAdam`` because that op constrains
``var``/``m``/``v``/``grad`` to share a single dtype. That cast materializes a
gradient-sized fp32 buffer on device every step.

``MusaResourceApplyAdamMixed`` accepts a low-precision gradient directly and
promotes it inside the kernel, eliminating the extra memory round trip. This
module exposes two ways to reach it:

  * ``apply_adam_mixed`` -- a low-level wrapper for custom training loops.
  * ``MusaAdam`` -- a drop-in replacement for ``tf.keras.optimizers.Adam``
    that routes the dense apply path to the mixed op when it is safe to do so
    (var is fp32, gradient is bf16/fp16/fp32, not on the AMSGrad branch).

If you only ever drive Keras through ``tf.function`` / ``Model.fit``, the
Grappler-side ``MusaAdamMixedFusion`` rewrite will fold ``Cast +
ResourceApplyAdam`` automatically and you do not need ``MusaAdam`` at all.
``MusaAdam`` is mainly useful in eager-mode custom loops where the Grappler
pass does not run.
"""

import logging

from ._loader import get_musa_ops


logger = logging.getLogger(__name__)


def apply_adam_mixed(var, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                     epsilon, grad, use_locking=False, use_nesterov=False):
    """Apply one Adam step with fp32 state and a (possibly) low-precision grad.

    ``var``, ``m``, and ``v`` must be fp32 resource variables. ``grad`` may be
    bf16, fp16, or fp32. Promotion to fp32 happens inside the kernel using the
    SDK's RNE intrinsics, so per-element promotion error is bounded by 0.5 ULP
    and the Adam update itself is bit-for-bit equivalent to running Adam in
    pure fp32 on a fp32-cast copy of the same gradient.

    Args:
      var, m, v: ``tf.Variable`` of dtype fp32 (resource variables).
      beta1_power, beta2_power, lr, beta1, beta2, epsilon: fp32 scalar tensors.
        Mixed dtypes here are accepted and silently cast to fp32.
      grad: dense gradient tensor with dtype fp32, fp16, or bf16.
      use_locking: forwarded to the op attr; controls slot mutex behaviour.
      use_nesterov: forwarded to the op attr; ``MusaResourceApplyAdamMixed``
        actually honours this flag (the legacy ``MusaResourceApplyAdam`` path
        ignores it).

    Returns:
      The op result (use as a control dependency; the actual update happens
      in place on ``var``/``m``/``v``).
    """
    import tensorflow as tf  # local import: avoid forcing TF import at module load

    ops = get_musa_ops()
    if ops is None or not hasattr(ops, "musa_resource_apply_adam_mixed"):
        raise RuntimeError(
            "MusaResourceApplyAdamMixed is not registered. Did the plugin load "
            "successfully? Check that libmusa_plugin.so was rebuilt against a "
            "version of tensorflow_musa_extension that includes this op."
        )

    def _to_fp32_scalar(value):
        if isinstance(value, tf.Tensor) and value.dtype == tf.float32:
            return value
        return tf.cast(value, tf.float32)

    var_handle = var.handle if hasattr(var, "handle") else var
    m_handle = m.handle if hasattr(m, "handle") else m
    v_handle = v.handle if hasattr(v, "handle") else v

    return ops.musa_resource_apply_adam_mixed(
        var_handle,
        m_handle,
        v_handle,
        _to_fp32_scalar(beta1_power),
        _to_fp32_scalar(beta2_power),
        _to_fp32_scalar(lr),
        _to_fp32_scalar(beta1),
        _to_fp32_scalar(beta2),
        _to_fp32_scalar(epsilon),
        grad,
        use_locking=use_locking,
        use_nesterov=use_nesterov,
    )


def apply_sparse_adam_mixed(var, m, v, beta1_power, beta2_power, lr, beta1,
                            beta2, epsilon, grad, indices, use_locking=False):
    """Single-kernel sparse Adam update for fp32 state + low-precision grad.

    Stock Keras Adam's ``_resource_apply_sparse`` (legacy and new optimizer
    APIs both) decomposes the sparse update into 5-10 primitive ops per
    variable -- two dense ``assign`` (m *= beta1, v *= beta2) + two
    ``ResourceScatterAdd`` (m[indices] += ..., v[indices] += ...) + one
    dense ``assign_sub`` for var.  On a recommendation model with N sparse
    feature embeddings, that's ~5N primitive kernel launches per step,
    completely dominated by launch overhead -- a profile of such a run
    typically shows ``ResourceScatterSub_*`` / ``ResourceScatterAdd_*``
    nodes consuming the majority of device time.

    ``MusaResourceSparseApplyAdam`` is a single fused kernel that performs
    the full sparse Adam update (var/m/v updates, including bias correction
    and per-row Adam math) in one launch.  The plugin op constrains
    ``T = var.dtype = m.dtype = v.dtype = grad.dtype``; since the standard
    mixed-precision recipe keeps var/m/v in fp32, this wrapper promotes a
    bf16/fp16 ``grad`` to fp32 once before dispatch.  That single cast on
    a sparse-buffer-sized gradient is much cheaper than the dozens of
    redundant casts the decomposed path materializes per variable.

    Args:
      var, m, v: fp32 resource variables.
      beta1_power, beta2_power, lr, beta1, beta2, epsilon: fp32 scalar
        tensors (silently cast if not already fp32).
      grad: sparse gradient values (after dedup -- one row per unique index).
        Dtype may be fp32, fp16, or bf16; promoted to fp32 if needed.
      indices: 1-D int32/int64 indices into ``var``'s first dimension.
      use_locking: forwarded to the op attr; controls slot mutex behaviour.

    Returns:
      The op result (use as a control dependency; the actual update happens
      in place on ``var``/``m``/``v``).
    """
    import tensorflow as tf

    ops = get_musa_ops()
    if ops is None or not hasattr(ops, "musa_resource_sparse_apply_adam"):
        raise RuntimeError(
            "MusaResourceSparseApplyAdam is not registered. Did the plugin "
            "load successfully? Check that libmusa_plugin.so was rebuilt "
            "against a version of tensorflow_musa_extension that includes "
            "this op."
        )

    var_handle = var.handle if hasattr(var, "handle") else var
    m_handle = m.handle if hasattr(m, "handle") else m
    v_handle = v.handle if hasattr(v, "handle") else v

    # Promote grad to fp32 if needed -- plugin op requires uniform T across
    # var/m/v/grad.  This is the single Cast that replaces the decomposed
    # path's many implicit promotions.
    if grad.dtype != tf.float32:
        grad = tf.cast(grad, tf.float32)

    def _to_fp32_scalar(value):
        if isinstance(value, tf.Tensor) and value.dtype == tf.float32:
            return value
        return tf.cast(value, tf.float32)

    return ops.musa_resource_sparse_apply_adam(
        var_handle,
        m_handle,
        v_handle,
        _to_fp32_scalar(beta1_power),
        _to_fp32_scalar(beta2_power),
        _to_fp32_scalar(lr),
        _to_fp32_scalar(beta1),
        _to_fp32_scalar(beta2),
        _to_fp32_scalar(epsilon),
        grad,
        indices,
        use_locking=use_locking,
    )


def _resolve_adam_base_class():
    """Return the Keras optimizer base class we should subclass.

    TF 2.11 rewrote ``tf.keras.optimizers`` from the OptimizerV2 hierarchy
    (with ``_resource_apply_dense`` / ``_resource_apply_sparse`` hooks) to
    a new ``Optimizer`` base class whose only override point is
    ``update_step(grad, variable, learning_rate)``. The new class has no
    way to call into a custom op the way ``_resource_apply_dense`` did,
    so ``MusaAdam`` would have to be rewritten as an ``update_step``
    override that calls ``musa_resource_apply_adam_mixed`` directly.

    Pending that rewrite, the simplest path is to subclass
    ``tf.keras.optimizers.legacy.Adam``, which preserves the old hierarchy
    on TF 2.11+ specifically for backward compatibility with optimizer
    subclasses like this one. Users who want to use the new optimizer
    can call ``apply_adam_mixed(...)`` directly from their training loop.
    """
    import tensorflow as tf

    legacy = getattr(getattr(tf.keras, "optimizers", None), "legacy", None)
    if legacy is not None and hasattr(legacy, "Adam"):
        return legacy.Adam
    # TF 2.6-2.10: there was no `legacy` submodule; tf.keras.optimizers.Adam
    # *was* the OptimizerV2-based class.
    return tf.keras.optimizers.Adam


def _make_musa_adam_class():
    """Build the ``MusaAdam`` class lazily so importing this module does not
    require TensorFlow / Keras to be importable.
    """
    import tensorflow as tf

    AdamBase = _resolve_adam_base_class()

    class MusaAdam(AdamBase):
        """Keras ``Adam`` that uses ``MusaResourceApplyAdamMixed`` on MUSA
        when it is safe to do so.

        On TF 2.11+ this subclasses ``tf.keras.optimizers.legacy.Adam``
        because the new optimizer API (post-2.11) does not expose
        ``_resource_apply_dense``. On older TF (2.6-2.10) it subclasses the
        same-named ``tf.keras.optimizers.Adam``, which used the legacy
        hierarchy at that point.

        Numerics match the stock Adam to within one bf16/fp16 promotion ULP
        per element; the speedup comes from skipping the gradient ``Cast`` op
        that the stock path inserts under Keras mixed-precision policies.

        Falls back transparently to the stock implementation when:
          * the variable lives on a non-MUSA device,
          * the variable dtype is not fp32 (stay in fp32 state for proper
            mixed-precision training),
          * the gradient dtype is unsupported,
          * AMSGrad is enabled (no mixed AMSGrad op yet).
        """

        _LOWP_GRAD_DTYPES = (tf.float32, tf.bfloat16, tf.float16)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._musa_amsgrad_warned = False
            self._musa_var_dtype_warned = False

        def _resource_apply_dense(self, grad, var, apply_state=None):
            if self.amsgrad:
                if not self._musa_amsgrad_warned:
                    logger.info(
                        "MusaAdam: AMSGrad is enabled; falling back to the "
                        "stock Adam apply path. Set amsgrad=False to enable "
                        "the MUSA mixed-precision fast path."
                    )
                    self._musa_amsgrad_warned = True
                return super()._resource_apply_dense(grad, var, apply_state)

            var_device = var.device or ""
            if "MUSA" not in var_device:
                return super()._resource_apply_dense(grad, var, apply_state)

            if var.dtype.base_dtype != tf.float32:
                if not self._musa_var_dtype_warned:
                    logger.warning(
                        "MusaAdam: variable %s has dtype %s but mixed-precision "
                        "Adam requires fp32 state. Falling back to the stock "
                        "(possibly bf16/fp16) Adam path, which loses precision "
                        "between iterations. Consider keeping master weights "
                        "in fp32 and using a Keras mixed_bfloat16 / "
                        "mixed_float16 policy.",
                        var.name,
                        var.dtype,
                    )
                    self._musa_var_dtype_warned = True
                return super()._resource_apply_dense(grad, var, apply_state)

            if grad.dtype not in self._LOWP_GRAD_DTYPES:
                return super()._resource_apply_dense(grad, var, apply_state)

            var_dtype = var.dtype.base_dtype
            coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                            self._fallback_apply_state(var_device, var_dtype))
            m = self.get_slot(var, "m")
            v = self.get_slot(var, "v")

            return apply_adam_mixed(
                var=var,
                m=m,
                v=v,
                beta1_power=coefficients["beta_1_power"],
                beta2_power=coefficients["beta_2_power"],
                lr=coefficients["lr_t"],
                beta1=coefficients["beta_1_t"],
                beta2=coefficients["beta_2_t"],
                epsilon=coefficients["epsilon"],
                grad=grad,
                use_locking=self._use_locking,
                use_nesterov=False,
            )

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            """Fused sparse Adam update via MusaResourceSparseApplyAdam.

            The base class's implementation (see TF 2.15
            keras/optimizer_v2/adam.py:203) decomposes the sparse update
            into:

                m_t = state_ops.assign(m, m * beta_1)            # full dense
                m_t = scatter_add(m, indices, scaled_grad)        # sparse
                v_t = state_ops.assign(v, v * beta_2)            # full dense
                v_t = scatter_add(v, indices, scaled_v_grad)      # sparse
                var = state_ops.assign_sub(var, lr * m_t /
                                                (sqrt(v_t) + epsilon))

            For a recommendation model with many sparse feature embeddings
            this manifests in profiles as dozens of small
            ``ResourceScatterAdd`` / ``ResourceScatterSub`` /
            ``AssignSubVariableOp`` nodes that together dominate device
            time -- each launch has fixed overhead independent of the
            (small) per-batch row count.

            This override collapses the entire sparse update into a single
            ``MusaResourceSparseApplyAdam`` kernel call, mirroring what
            ``_resource_apply_dense`` already does via
            ``MusaResourceApplyAdamMixed``.

            Fallback conditions match the dense path (non-MUSA device,
            non-fp32 var, AMSGrad enabled, unsupported grad dtype); on
            fallback the base-class slow path runs unchanged.
            """
            if self.amsgrad:
                if not self._musa_amsgrad_warned:
                    logger.info(
                        "MusaAdam: AMSGrad is enabled; falling back to the "
                        "stock sparse Adam apply path. Set amsgrad=False to "
                        "enable the MUSA fused sparse Adam fast path."
                    )
                    self._musa_amsgrad_warned = True
                return super()._resource_apply_sparse(
                    grad, var, indices, apply_state)

            var_device = var.device or ""
            if "MUSA" not in var_device:
                return super()._resource_apply_sparse(
                    grad, var, indices, apply_state)

            if var.dtype.base_dtype != tf.float32:
                if not self._musa_var_dtype_warned:
                    logger.warning(
                        "MusaAdam: sparse variable %s has dtype %s but the "
                        "MUSA fused sparse Adam requires fp32 state. "
                        "Falling back to the decomposed sparse Adam path.",
                        var.name,
                        var.dtype,
                    )
                    self._musa_var_dtype_warned = True
                return super()._resource_apply_sparse(
                    grad, var, indices, apply_state)

            if grad.dtype not in self._LOWP_GRAD_DTYPES:
                return super()._resource_apply_sparse(
                    grad, var, indices, apply_state)

            var_dtype = var.dtype.base_dtype
            coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                            self._fallback_apply_state(var_device, var_dtype))
            m = self.get_slot(var, "m")
            v = self.get_slot(var, "v")

            return apply_sparse_adam_mixed(
                var=var,
                m=m,
                v=v,
                beta1_power=coefficients["beta_1_power"],
                beta2_power=coefficients["beta_2_power"],
                lr=coefficients["lr_t"],
                beta1=coefficients["beta_1_t"],
                beta2=coefficients["beta_2_t"],
                epsilon=coefficients["epsilon"],
                grad=grad,
                indices=indices,
                use_locking=self._use_locking,
            )

    return MusaAdam


# Materialize MusaAdam lazily on first attribute access so test environments
# without Keras can still import tensorflow_musa.
_MUSA_ADAM_CLASS = None


def __getattr__(name):
    if name == "MusaAdam":
        global _MUSA_ADAM_CLASS
        if _MUSA_ADAM_CLASS is None:
            _MUSA_ADAM_CLASS = _make_musa_adam_class()
        return _MUSA_ADAM_CLASS
    raise AttributeError(f"module 'tensorflow_musa._optimizers' has no "
                         f"attribute {name!r}")

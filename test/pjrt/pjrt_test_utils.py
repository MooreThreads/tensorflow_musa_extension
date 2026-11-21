# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
"""Shared utilities for PJRT mode tests."""

import os
import unittest

import numpy as np
import tensorflow_musa as _tf_musa  # noqa: F401
import tensorflow as tf


_PJRT_STATUS = _tf_musa.get_pjrt_status()
if _PJRT_STATUS.get("mode") != "pjrt" or _PJRT_STATUS.get("phase") != "ready":
    raise RuntimeError(
        "PJRT tests require tensorflow_musa to report mode=pjrt, phase=ready; "
        "run the test process with --xla. "
        f"Observed status: {_PJRT_STATUS}"
    )

tf.config.set_soft_device_placement(False)


class PjrtTestCase(tf.test.TestCase):
    """Base test class for PJRT mode tests."""

    test_session = None

    @classmethod
    def setUpClass(cls):
        super(PjrtTestCase, cls).setUpClass()
        if not tf.config.list_physical_devices("MUSA"):
            raise unittest.SkipTest("No MUSA devices found.")

    def _run_eager(self, fn, *args, **kwargs):
        with tf.device("/device:MUSA:0"):
            return fn(*args, **kwargs)

    def _run_jit(self, fn, *args, **kwargs):
        @tf.function(jit_compile=True)
        def compiled(*inputs):
            return fn(*inputs, **kwargs)
        with tf.device("/device:MUSA:0"):
            return compiled(*args)

    def _assert_on_musa(self, result, label):
        device = getattr(result, "device", "")
        self.assertIn("MUSA", device.upper(), f"{label} ran on {device!r}")

    def assertEagerAndJitClose(self, eager_result, jit_result, rtol=1e-5, atol=1e-8):
        eager_np = eager_result.numpy() if hasattr(eager_result, "numpy") else eager_result
        jit_np = jit_result.numpy() if hasattr(jit_result, "numpy") else jit_result
        self.assertAllClose(eager_np, jit_np, rtol=rtol, atol=atol)

    def _testBothEagerAndJit(self, fn, *args, rtol=1e-5, atol=1e-8):
        eager_r = self._run_eager(fn, *args)
        jit_r = self._run_jit(fn, *args)
        self._assert_on_musa(eager_r, "eager result")
        self._assert_on_musa(jit_r, "JIT result")
        self.assertEagerAndJitClose(eager_r, jit_r, rtol=rtol, atol=atol)
        return eager_r, jit_r

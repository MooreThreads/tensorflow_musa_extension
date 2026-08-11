# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
"""PJRT-compatible musa_test_utils — shadows the original in PJRT mode.

When PJRT is enabled, the original musa_test_utils would try to load
libmusa_plugin.so (which is not available).  This module re-exports
PjrtTestCase as MUSATestCase so that the standard TF op tests in
test/ops/ can run under PJRT without modification.
"""

import numpy as np
import tensorflow as tf

from pjrt.pjrt_test_utils import PjrtTestCase

# Re-export PjrtTestCase as MUSATestCase for test compatibility
MUSATestCase = PjrtTestCase


def _get_test_full_name(test_method):
    """Get the full name of a test method in format module.class.method."""
    test_class = test_method.__self__.__class__
    module_name = test_class.__module__
    module_file = module_name.split(".")[-1] if module_name else "unknown"
    class_name = test_class.__name__
    method_name = test_method.__name__
    return f"{module_file}.{class_name}.{method_name}"


def load_musa_plugin():
    """No-op in PJRT mode — native plugin is not loaded."""
    pass


def load_musa_ops():
    """No-op in PJRT mode — native ops are not available."""
    pass

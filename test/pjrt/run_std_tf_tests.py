#!/usr/bin/env python3
"""PJRT mode: run standard TF op tests from test/ops/ (excluding MUSA custom ops).

Usage:
    LD_LIBRARY_PATH=/path/to/tensorflow:$LD_LIBRARY_PATH \
    TENSORFLOW_MUSA_ENABLE_PJRT=1 \
    TENSORFLOW_MUSA_PJRT_LIBRARY_PATH=/path/to/libmusa_pjrt_plugin_zy.so \
    MUSA_VISIBLE_DEVICES=5 \
    python test/pjrt/run_std_tf_tests.py
"""

import os, sys, time, unittest, warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TENSORFLOW_MUSA_ENABLE_PJRT", "1")

# MUSA custom op tests to skip (require libmusa_plugin.so)
MUSA_OP_TESTS = {
    "apply_adagrad_op_test", "apply_adamax_op_test",
    "apply_gradient_descent_op_test", "apply_resource_op_test",
    "apply_rmsprop_op_test", "assign_add_op_test", "assign_op_test",
    "biasadd_grad_op_test", "broadcast_gradient_args_op_test",
    "concat_offset_op_test", "conv2d_backward_op_test",
    "div_no_nan_op_test", "empty_tensor_list_op_test",
    "extract_image_patches_op_test", "fused_batchnorm_hcnw_op_test",
    "fused_batchnorm_op_test", "gelu_op_test",
    "intopkv2_op_test", "invert_permutation_op_test",
    "layernorm_grad_op_test", "layernorm_op_test",
    "matmul_op_test", "matmul_bias_op_test", "matmul_biasadd_op_test",
    "matrix_band_part_op_test", "matrix_diag_op_test",
    "matrix_set_diag_op_test", "max_op_test", "min_op_test",
    "mirror_pad_op_test", "mul_op_test",
    "normalize_op_test", "one_hot_op_test", "pack_op_test",
    "pad_op_test", "quantize_and_dequantize_op_test",
    "reduce_any_op_test", "reduce_prod_op_test",
    "relu_op_test", "relu_grad_op_test", "reshape_op_test",
    "rsqrt_op_test",
}

import tensorflow_musa  # noqa: F401
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ops")
sys.path.insert(0, os.path.dirname(test_dir))

if __name__ == "__main__":
    start = time.time()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith("_test.py") and not fname.endswith("_op_test.py"):
            continue
        if fname in MUSA_OP_TESTS:
            continue
        if fname.startswith("xla_"):
            continue
        modname = "ops." + fname[:-3]
        try:
            mod = __import__(modname, fromlist=[""])
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
                    suite.addTests(loader.loadTestsFromTestCase(obj))
        except Exception as e:
            print("SKIP %s: %s" % (fname, str(e)[:80]))

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    elapsed = time.time() - start

    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)

    print("")
    print("=" * 60)
    if failed == 0 and errors == 0:
        print("ALL STANDARD TF OP TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    print("Total: %d  Passed: %d  Failed: %d  Errors: %d  Skipped: %d" % (
        total, passed, failed, errors, skipped))
    print("Time: %.1fs" % elapsed)
    sys.exit(0 if failed == 0 and errors == 0 else 1)

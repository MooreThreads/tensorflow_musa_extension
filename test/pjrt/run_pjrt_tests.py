#!/usr/bin/env python3
"""PJRT mode test runner for standard TF ops on MUSA device."""

import os, sys, time, unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
if "--xla" not in sys.argv:
    raise SystemExit("PJRT tests require --xla.")

import tensorflow_musa
import tensorflow as tf
status = tensorflow_musa.get_pjrt_status()
if status.get("mode") != "pjrt" or status.get("phase") != "ready":
    raise SystemExit(f"PJRT provider is not ready: {status}")
print("PJRT_STATUS =", status)
tf.get_logger().setLevel("ERROR")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    start = time.time()
    loader = unittest.TestLoader()
    suite = loader.discover(
        os.path.dirname(os.path.abspath(__file__)),
        pattern="ops_*.py",
    )
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    elapsed = time.time() - start

    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    failed = len(result.failures)
    errors = len(result.errors)

    print("\n" + "=" * 60)
    if failed == 0 and errors == 0:
        print("ALL PJRT TESTS PASSED")
    else:
        print("SOME PJRT TESTS FAILED")
    print("=" * 60)
    print("Total: %d  Passed: %d  Failed: %d  Errors: %d" % (total, passed, failed, errors))
    print("Time: %.1fs" % elapsed)
    sys.exit(0 if failed == 0 and errors == 0 else 1)

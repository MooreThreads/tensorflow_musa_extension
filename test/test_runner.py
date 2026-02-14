#!/usr/bin/env python3

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

import os
# Set TensorFlow logging level to reduce verbose output
# TF_CPP_MIN_LOG_LEVEL: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import unittest
import sys
import importlib
import warnings
from pathlib import Path
import tensorflow as tf
import logging
from datetime import datetime

# Also set Python logging level for TensorFlow
tf.get_logger().setLevel('ERROR')

# Filter out TensorFlow deprecation warnings about cached_session
warnings.filterwarnings("ignore", message=".*cached_session.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*Not a test.*")


class CustomTestResult(unittest.TextTestResult):
    """Custom test result class with better reporting."""

    def __init__(self, stream, descriptions, verbosity, quiet=False, detail_mode=False):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []
        self.quiet = quiet
        self.detail_mode = detail_mode

    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_results.append(('PASS', str(test), None))
        if self.detail_mode:
            print(f"✓ {test._testMethodName}: PASS")

    def addError(self, test, err):
        super().addError(test, err)
        error_msg = str(err[1])
        self.test_results.append(('ERROR', str(test), error_msg))
        if self.detail_mode:
            print(f"✗ {test._testMethodName}: ERROR")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        error_msg = str(err[1])
        self.test_results.append(('FAIL', str(test), error_msg))
        if self.detail_mode:
            print(f"✗ {test._testMethodName}: FAIL")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.test_results.append(('SKIP', str(test), reason))
        if self.detail_mode:
            print(f"~ {test._testMethodName}: SKIPPED")


class DualOutput:
    """Class to write output to both console and file."""
    def __init__(self, file_handle):
        self.terminal = sys.stdout
        self.file = file_handle

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


class CustomTestRunner(unittest.TextTestRunner):
    """Custom test runner with summary reporting."""

    def __init__(self, verbosity=2, quiet=False, detail_mode=False, log_file=None):
        super().__init__(verbosity=verbosity)
        self.quiet = quiet
        self.detail_mode = detail_mode
        self.log_file = log_file
        self.original_stdout = None
        self.dual_output = None

    def _makeResult(self):
        return CustomTestResult(self.stream, self.descriptions, self.verbosity, 
                             quiet=self.quiet, detail_mode=self.detail_mode)

    def run(self, test):
        # If log_file is specified and in detail mode, redirect stdout
        if self.log_file and self.detail_mode:
            self.original_stdout = sys.stdout
            log_file_handle = open(self.log_file, 'w', encoding='utf-8')
            self.dual_output = DualOutput(log_file_handle)
            sys.stdout = self.dual_output
            
            # Add timestamp to log file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Test run started at: {timestamp}\n")

        try:
            result = super().run(test)
            
            # Print summary only
            total_tests = result.testsRun
            passed = len([r for r in result.test_results if r[0] == 'PASS'])
            failed = len(result.failures)
            errors = len(result.errors)
            skipped = len(result.skipped)

            # Always show summary regardless of mode
            print("\n" + "="*50)
            print("TEST SUMMARY")
            print("="*50)
            print(f"Total: {total_tests}, Passed: {passed}, Failed: {failed}, Errors: {errors}, Skipped: {skipped}")

            # Always print detailed failure information
            if (failed > 0 or errors > 0):
                print("\nFAILED TESTS:")
                for test_name, full_test_str, msg in result.test_results:
                    if test_name in ['FAIL', 'ERROR']:
                        # Extract just the test method name
                        test_method = full_test_str.split(' ')[0].split('.')[-1]
                        print(f"  - {test_method}")
                        
        finally:
            # Restore original stdout
            if self.log_file and self.detail_mode:
                if self.dual_output:
                    self.dual_output.file.close()
                if self.original_stdout:
                    sys.stdout = self.original_stdout

        return result


def discover_and_run_tests(test_pattern="*_op_test.py", quiet=True, detail_mode=False, log_file=None):
    """Discover and run all test files matching the pattern."""
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob(test_pattern))

    if not test_files:
        print(f"No test files found matching pattern: {test_pattern}")
        return

    # Add test directory to Python path
    sys.path.insert(0, str(test_dir))

    # Load and run all test suites
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_file in sorted(test_files):
        module_name = test_file.stem
        try:
            module = importlib.import_module(module_name)
            module_suite = loader.loadTestsFromModule(module)
            suite.addTests(module_suite)
            if detail_mode:
                print(f"Loaded tests from: {module_name}")
        except Exception as e:
            if detail_mode:
                print(f"Failed to load {module_name}: {e}")

    if suite.countTestCases() == 0:
        print("No tests found!")
        return

    if detail_mode:
        print(f"\nRunning {suite.countTestCases()} tests...\n")

    # Default to quiet=True, detail_mode controls verbosity
    runner = CustomTestRunner(verbosity=2 if detail_mode else 0, 
                            quiet=quiet, 
                            detail_mode=detail_mode,
                            log_file=log_file)
    result = runner.run(suite)

    # Exit with error code if any tests failed
    if result.failures or result.errors:
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MUSA operator tests")
    parser.add_argument("--pattern", default="*_op_test.py",
                       help="Test file pattern (default: *_op_test.py)")
    parser.add_argument("--single", help="Run a single test file")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode - only show summary (default behavior)")
    parser.add_argument("--detail", "-d", action="store_true",
                       help="Detail mode - show individual test results and write to log file")
    parser.add_argument("--log-file", default="test_results.log",
                       help="Log file path for detail mode (default: test_results.log)")

    args = parser.parse_args()

    # Default behavior: quiet mode
    quiet_mode = True
    detail_mode = False
    
    if args.detail:
        detail_mode = True
        quiet_mode = False  # Detail mode overrides quiet
    elif args.quiet:
        quiet_mode = True
        detail_mode = False
    else:
        # Default: quiet mode
        quiet_mode = True
        detail_mode = False

    if args.single:
        # Run a single test file
        sys.path.insert(0, str(Path(__file__).parent))
        module_name = Path(args.single).stem
        try:
            module = importlib.import_module(module_name)
            suite = unittest.TestLoader().loadTestsFromModule(module)
            runner = CustomTestRunner(verbosity=2 if detail_mode else 0,
                                    quiet=quiet_mode,
                                    detail_mode=detail_mode,
                                    log_file=args.log_file if detail_mode else None)
            result = runner.run(suite)
            if result.failures or result.errors:
                sys.exit(1)
        except Exception as e:
            if detail_mode:
                print(f"Failed to run {args.single}: {e}")
            else:
                # In quiet mode, only show error if it's critical
                pass
            sys.exit(1)
    else:
        # Run all tests - default to quiet mode
        discover_and_run_tests(args.pattern, 
                             quiet=quiet_mode, 
                             detail_mode=detail_mode,
                             log_file=args.log_file if detail_mode else None)
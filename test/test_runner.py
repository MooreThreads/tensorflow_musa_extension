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

import unittest
import sys
import os
import importlib
import warnings
from pathlib import Path

# Filter out TensorFlow deprecation warnings about cached_session
warnings.filterwarnings("ignore", message=".*cached_session.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*Not a test.*")


class CustomTestResult(unittest.TextTestResult):
    """Custom test result class with better reporting."""
    
    def __init__(self, stream, descriptions, verbosity, quiet=False):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []
        self.quiet = quiet
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_results.append(('PASS', str(test), None))
        if not self.quiet:
            print(f"✓ {test._testMethodName}: PASS")
    
    def addError(self, test, err):
        super().addError(test, err)
        error_msg = str(err[1])
        self.test_results.append(('ERROR', str(test), error_msg))
        if not self.quiet:
            print(f"✗ {test._testMethodName}: ERROR")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        error_msg = str(err[1])
        self.test_results.append(('FAIL', str(test), error_msg))
        if not self.quiet:
            print(f"✗ {test._testMethodName}: FAIL")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.test_results.append(('SKIP', str(test), reason))
        if not self.quiet:
            print(f"~ {test._testMethodName}: SKIPPED")


class CustomTestRunner(unittest.TextTestRunner):
    """Custom test runner with summary reporting."""
    
    def __init__(self, verbosity=2, quiet=False):
        super().__init__(verbosity=verbosity)
        self.quiet = quiet
    
    def _makeResult(self):
        return CustomTestResult(self.stream, self.descriptions, self.verbosity, self.quiet)
    
    def run(self, test):
        result = super().run(test)
        
        # Print summary only
        total_tests = result.testsRun
        passed = len([r for r in result.test_results if r[0] == 'PASS'])
        failed = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        

        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
    
        print(f"Total: {total_tests}, Passed: {passed}, Failed: {failed}, Errors: {errors}, Skipped: {skipped}")
        
        # Don't print detailed failure information when in quiet mode
        if (failed > 0 or errors > 0):
            print("\nFAILED TESTS:")
            for test_name, full_test_str, msg in result.test_results:
                if test_name in ['FAIL', 'ERROR']:
                    # Extract just the test method name
                    test_method = full_test_str.split(' ')[0].split('.')[-1]
                    print(f"  - {test_method}")
        
        return result


def discover_and_run_tests(test_pattern="*_op_test.py", quiet=False):
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
            if not quiet:
                print(f"Loaded tests from: {module_name}")
        except Exception as e:
            if not quiet:
                print(f"Failed to load {module_name}: {e}")
    
    if suite.countTestCases() == 0:
        print("No tests found!")
        return
    
    if not quiet:
        print(f"\nRunning {suite.countTestCases()} tests...\n")
    
    runner = CustomTestRunner(verbosity=2 if not quiet else 0, quiet=quiet)
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
                       help="Quiet mode - only show summary, no detailed output")
    
    args = parser.parse_args()
    
    if args.single:
        # Run a single test file
        sys.path.insert(0, str(Path(__file__).parent))
        module_name = Path(args.single).stem
        try:
            module = importlib.import_module(module_name)
            suite = unittest.TestLoader().loadTestsFromModule(module)
            runner = CustomTestRunner(verbosity=2 if not args.quiet else 0, quiet=args.quiet)
            result = runner.run(suite)
            if result.failures or result.errors:
                sys.exit(1)
        except Exception as e:
            if not args.quiet:
                print(f"Failed to run {args.single}: {e}")
            sys.exit(1)
    else:
        # Run all tests
        discover_and_run_tests(args.pattern, quiet=args.quiet)
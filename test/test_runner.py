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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import unittest
import sys
import importlib
import warnings
from pathlib import Path
import tensorflow as tf
import time
from datetime import datetime

# Set logging levels
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", message=".*cached_session.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*Not a test.*")

# ============================================================================
# ANSI Color Codes for Beautiful Output
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Check if terminal supports colors
def supports_color():
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True

USE_COLORS = supports_color()

def color(text, color_code):
    return f"{color_code}{text}{Colors.END}" if USE_COLORS else text

def green(text): return color(text, Colors.GREEN)
def red(text): return color(text, Colors.RED)
def yellow(text): return color(text, Colors.YELLOW)
def blue(text): return color(text, Colors.BLUE)
def cyan(text): return color(text, Colors.CYAN)
def bold(text): return color(text, Colors.BOLD)

# ============================================================================
# Enhanced Progress Bar with Real-time Updates
# ============================================================================
class ProgressBar:
    def __init__(self, total, width=50):
        self.total = total
        self.current = 0
        self.width = width
        self.start_time = time.time()
        self.status_counts = {'PASS': 0, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}

    def update(self, test_name, status):
        self.current += 1
        self.status_counts[status] = self.status_counts.get(status, 0) + 1

        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)

        # Status icon and color
        status_icons = {'PASS': '✓', 'FAIL': '✗', 'ERROR': '⚠', 'SKIP': '○'}
        status_colors = {'PASS': green, 'FAIL': red, 'ERROR': red, 'SKIP': yellow}

        icon = status_icons.get(status, '?')
        color_func = status_colors.get(status, lambda x: x)

        # Calculate statistics
        elapsed = time.time() - self.start_time
        eta = (elapsed / self.current * (self.total - self.current)) if self.current > 0 else 0

        # Rate calculation
        rate = self.current / elapsed if elapsed > 0 else 0

        # Truncate test name if too long
        max_name_len = 30
        display_name = test_name[:max_name_len-3] + '...' if len(test_name) > max_name_len else test_name

        # Build progress line with stats
        stats = f"P:{self.status_counts['PASS']} F:{self.status_counts['FAIL']} E:{self.status_counts['ERROR']} S:{self.status_counts['SKIP']}"
        progress_line = f'\r  [{color_func(bar)}] {self.current}/{self.total} {color_func(icon)} {display_name} | {stats} | {rate:.1f} tests/s | ETA: {eta:.0f}s'

        sys.stdout.write(progress_line)
        sys.stdout.flush()

    def finish(self):
        elapsed = time.time() - self.start_time
        sys.stdout.write(f'\n  Completed in {elapsed:.1f}s\n')
        sys.stdout.flush()

# ============================================================================
# Custom Test Result with Enhanced Reporting
# ============================================================================
class CustomTestResult(unittest.TextTestResult):
    """Custom test result class with enhanced reporting."""

    def __init__(self, stream, descriptions, verbosity, quiet=False, detail_mode=False, total_tests=0):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []
        self.quiet = quiet
        self.detail_mode = detail_mode
        # Always show progress bar regardless of quiet mode to prevent user confusion
        self.show_progress = True  # Progress bar is always enabled
        self.progress_bar = None
        self.start_time = None
        self.total_tests = total_tests  # Initialize with total tests count

        # Initialize progress bar immediately if we have total tests
        if self.total_tests > 0 and self.show_progress:
            self.progress_bar = ProgressBar(self.total_tests)
            if self.detail_mode:  # Only show the fancy header in detail mode
                box_width = 70
                print(f"\n{bold(cyan('┌' + '─'*box_width + '┐'))}")
                title = bold('MUSA Test Suite - Running Tests')
                print(f"{bold(cyan('│'))}  {title}{' '*(box_width-2-len(title))}{bold(cyan('│'))}")
                print(f"{bold(cyan('└' + '─'*box_width + '┘'))}\n")

    def startTestRun(self):
        super().startTestRun()
        self.start_time = time.time()

    def startTest(self, test):
        super().startTest(test)
        # Don't print individual test info in quiet mode
        if not self.quiet and self.detail_mode:
            pass  # The detailed output is handled by unittest framework

    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_results.append(('PASS', str(test), None))
        if self.show_progress and self.progress_bar:
            self.progress_bar.update(test._testMethodName, 'PASS')

    def addError(self, test, err):
        super().addError(test, err)
        error_msg = str(err[1])
        self.test_results.append(('ERROR', str(test), error_msg))
        if self.show_progress and self.progress_bar:
            self.progress_bar.update(test._testMethodName, 'ERROR')

    def addFailure(self, test, err):
        super().addFailure(test, err)
        error_msg = str(err[1])
        self.test_results.append(('FAIL', str(test), error_msg))
        if self.show_progress and self.progress_bar:
            self.progress_bar.update(test._testMethodName, 'FAIL')

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.test_results.append(('SKIP', str(test), reason))
        if self.show_progress and self.progress_bar:
            self.progress_bar.update(test._testMethodName, 'SKIP')

    def stopTestRun(self):
        super().stopTestRun()
        if self.show_progress and self.progress_bar:
            self.progress_bar.finish()


# ============================================================================
# Dual Output (Console + File)
# ============================================================================
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


# ============================================================================
# PrettyTable-based Summary Generator
# ============================================================================
from prettytable import PrettyTable

def create_pretty_summary(result, elapsed_time):
    """Create a beautiful summary using prettytable."""
    total = result.testsRun
    passed = len([r for r in result.test_results if r[0] == 'PASS'])
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)

    # Calculate pass rate
    pass_rate = (passed / total * 100) if total > 0 else 0

    # Create main summary table
    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Value"]
    summary_table.align["Metric"] = "l"
    summary_table.align["Value"] = "r"

    # Add rows with colored values
    def format_count(count, status):
        if count == 0:
            return str(count)
        elif status == 'PASS':
            return green(str(count))
        elif status in ['FAIL', 'ERROR']:
            return red(str(count))
        else:
            return yellow(str(count))

    summary_table.add_row(["Total Tests", total])
    summary_table.add_row(["Passed", format_count(passed, 'PASS')])
    summary_table.add_row(["Failed", format_count(failed, 'FAIL')])
    summary_table.add_row(["Errors", format_count(errors, 'ERROR')])
    summary_table.add_row(["Skipped", format_count(skipped, 'SKIP')])
    summary_table.add_row(["Pass Rate", f"{pass_rate:.1f}%"])
    summary_table.add_row(["Execution Time", f"{elapsed_time:.2f}s"])

    # Set table style
    summary_table.border = True
    summary_table.header = True
    summary_table.padding_width = 1

    # Create status header
    if failed == 0 and errors == 0:
        status_text = green("✓ ALL TESTS PASSED")
    elif errors > 0:
        status_text = red("⚠ TESTS COMPLETED WITH ERRORS")
    else:
        status_text = red("✗ SOME TESTS FAILED")

    # Format the complete output
    output_lines = []
    output_lines.append("")
    output_lines.append(bold(cyan("=" * 72)))
    output_lines.append(bold(cyan("MUSA TEST SUMMARY".center(72))))
    output_lines.append(bold(cyan("=" * 72)))
    output_lines.append(status_text.center(72))
    output_lines.append("")
    output_lines.append(summary_table.get_string())
    output_lines.append("")

    # Add failed tests section if needed
    if failed > 0 or errors > 0:
        fail_table = PrettyTable()
        fail_table.field_names = ["Status", "Test Method"]
        fail_table.align["Status"] = "c"
        fail_table.align["Test Method"] = "l"

        for test_status, full_test_str, msg in result.test_results:
            if test_status in ['FAIL', 'ERROR']:
                test_method = full_test_str.split(' ')[0].split('.')[-1]
                status_icon = red('✗') if test_status == 'FAIL' else red('⚠')
                fail_table.add_row([status_icon, test_method])

        fail_table.border = True
        fail_table.header = True
        fail_table.padding_width = 1

        output_lines.append(bold(red("FAILED/ERROR TESTS".center(72))))
        output_lines.append("")
        output_lines.append(fail_table.get_string())
        output_lines.append("")

    return "\n".join(output_lines)


# ============================================================================
# Custom Test Runner with PrettyTable Summary
# ============================================================================
class CustomTestRunner(unittest.TextTestRunner):
    """Custom test runner with enhanced summary reporting using prettytable."""

    def __init__(self, verbosity=2, quiet=False, detail_mode=False, log_file=None):
        # Set verbosity based on modes
        if quiet and not detail_mode:
            # Quiet mode: minimal output but still show progress bar
            effective_verbosity = 0
        else:
            # Detail mode or normal mode: show individual test results
            effective_verbosity = verbosity

        super().__init__(verbosity=effective_verbosity)
        self.quiet = quiet
        self.detail_mode = detail_mode
        self.log_file = log_file
        self.original_stdout = None
        self.dual_output = None
        self.total_tests = 0  # Store total tests for result creation

    def _makeResult(self):
        # Create result with the stored total tests count
        return CustomTestResult(
            self.stream,
            self.descriptions,
            self.verbosity,
            quiet=self.quiet,
            detail_mode=self.detail_mode,
            total_tests=self.total_tests
        )

    def _printSummary(self, result, elapsed_time):
        """Print beautiful summary using prettytable."""
        summary_output = create_pretty_summary(result, elapsed_time)
        print(summary_output)

    def run(self, test):
        # If log_file is specified and in detail mode, redirect stdout
        if self.log_file and self.detail_mode:
            self.original_stdout = sys.stdout
            log_file_handle = open(self.log_file, 'w', encoding='utf-8')
            self.dual_output = DualOutput(log_file_handle)
            sys.stdout = self.dual_output

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Test run started at: {timestamp}\n")

        start_time = time.time()
        try:
            # Get total test count before running
            self.total_tests = test.countTestCases()

            # Let the parent class handle the result creation and running
            result = super().run(test)
            elapsed_time = time.time() - start_time

            # Always print summary
            self._printSummary(result, elapsed_time)

        finally:
            # Restore original stdout
            if self.log_file and self.detail_mode:
                if self.dual_output:
                    self.dual_output.file.close()
                if self.original_stdout:
                    sys.stdout = self.original_stdout

        # Exit with error code if any tests failed
        if result.failures or result.errors:
            sys.exit(1)


# ============================================================================
# Test Discovery and Execution
# ============================================================================
def discover_and_run_tests(test_pattern="*_op_test.py", quiet=True, detail_mode=False, log_file=None):
    """Discover and run all test files matching the pattern."""
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob(test_pattern))

    if not test_files:
        print(f"{red('✗')} No test files found matching pattern: {test_pattern}")
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
                print(f"  {green('✓')} Loaded tests from: {module_name}")
        except Exception as e:
            if detail_mode:
                print(f"  {red('✗')} Failed to load {module_name}: {e}")

    if suite.countTestCases() == 0:
        print(f"{red('✗')} No tests found!")
        return

    if detail_mode:
        print(f"\n  Running {suite.countTestCases()} tests...\n")

    # Run tests
    runner = CustomTestRunner(verbosity=2 if detail_mode else 0,
                            quiet=quiet,
                            detail_mode=detail_mode,
                            log_file=log_file)
    result = runner.run(suite)


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MUSA operator tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py                    # Run all tests (shows progress bar + summary)
  python test_runner.py --quiet            # Run all tests (shows progress bar + summary, minimal details)
  python test_runner.py --detail           # Run all tests with progress bar and individual results
  python test_runner.py --single matmul_op_test.py  # Run single test file
  python test_runner.py --pattern "*_grad*_op_test.py"  # Run gradient tests only
        """
    )
    parser.add_argument("--pattern", default="*_op_test.py",
                       help="Test file pattern (default: *_op_test.py)")
    parser.add_argument("--single", help="Run a single test file")
    parser.add_argument("--detail", "-d", action="store_true",
                       help="Detail mode - show progress bar and individual results")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode - show progress bar and summary only (no individual test details)")
    parser.add_argument("--log-file", default="test_results.log",
                       help="Log file path for detail mode (default: test_results.log)")

    args = parser.parse_args()

    # Determine modes
    detail_mode = args.detail
    quiet_mode = args.quiet

    # Print header in detail mode
    if detail_mode:
        print(f"\n{bold(cyan('╔' + '═'*70 + '╗'))}")
        header_text = bold('MUSA OPERATOR TEST SUITE')
        print(f"{bold(cyan('║'))}  {header_text}{' '*(68-len(header_text))}{bold(cyan('║'))}")
        print(f"{bold(cyan('╚' + '═'*70 + '╝'))}\n")

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
            if result and (result.failures or result.errors):
                sys.exit(1)
        except Exception as e:
            if detail_mode:
                print(f"{red('✗')} Failed to run {args.single}: {e}")
            sys.exit(1)
    else:
        # Run all tests
        discover_and_run_tests(args.pattern,
                             quiet=quiet_mode,
                             detail_mode=detail_mode,
                             log_file=args.log_file if detail_mode else None)

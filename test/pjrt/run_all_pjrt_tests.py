#!/usr/bin/env python3
"""Run the OpenXLA-MUSA PJRT validation layers in isolated processes.

Each child process gets a fresh TensorFlow/PJRT registration state. This
runner intentionally does not combine the control-plane and device tests in a
single interpreter because PJRT registration is process-global.
"""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PJRT_DIR = ROOT / "test" / "pjrt"

TESTS = (
    ("PJRT bootstrap contract", PJRT_DIR / "pjrt_bootstrap_test.py", ("--xla",)),
    ("PJRT boundary contract", PJRT_DIR / "pjrt_boundary_test.py", ("--xla",)),
    ("Native boundary contract", PJRT_DIR / "pjrt_boundary_test.py", ("--no-xla",)),
    ("Gradient import contract", PJRT_DIR / "pjrt_gradient_import_test.py", ("--xla",)),
    ("XLA smoke", PJRT_DIR / "pjrt_xla_smoke_test.py", ("--xla",)),
    ("PJRT ops", PJRT_DIR / "run_pjrt_tests.py", ("--xla",))
)


def main() -> int:
    if "--xla" not in sys.argv[1:]:
        raise SystemExit("PJRT tests require --xla.")

    env = os.environ.copy()
    # Each child owns a fresh process-global PJRT registry. Do not inherit
    # stale discovery settings from the shell running this aggregate test.
    for key in (
        "TENSORFLOW_MUSA_ENABLE_PJRT",
        "TENSORFLOW_MUSA_PJRT_PROVIDER_MODULE",
        "TENSORFLOW_MUSA_PJRT_LIBRARY_PATH",
        "TENSORFLOW_MUSA_PJRT_BRIDGE_PATH",
        "TF_PLUGGABLE_DEVICE_LIBRARY_PATH",
        "PJRT_NAMES_AND_LIBRARY_PATHS",
        "NEXT_PLUGGABLE_DEVICE_USE_C_API",
        "MUSA_TFNPD_IS_PLUGGABLE_DEVICE",
        "MUSA_TFNPD_COMPILATION_DEVICE_NAME",
        "MTCC20X_TOOLS_DIR",
        "TF_XLA_FLAGS",
    ):
        env.pop(key, None)
    python = sys.executable
    for label, test_path, extra_args in TESTS:
        command = [python, str(test_path), *extra_args]
        child_env = env.copy()
        print(f"\n===== {label}: {' '.join(command)} =====", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=child_env)
        if result.returncode != 0:
            print(f"[FAIL] {label} exited with {result.returncode}")
            return result.returncode
        print(f"[PASS] {label}")

    print("\nALL PJRT VALIDATION TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

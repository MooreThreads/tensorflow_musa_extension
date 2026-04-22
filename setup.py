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

"""Setup script for tensorflow_musa package."""

import os
import shutil
import subprocess
import sys
from setuptools import setup, Command
from wheel.bdist_wheel import bdist_wheel


# Package metadata
PACKAGE_NAME = "tensorflow_musa"  # pip install name
SOURCE_DIR = "python"             # source code directory
VERSION = "0.1.0"
DESCRIPTION = "High-performance TensorFlow extension for Moore Threads MUSA GPUs"
AUTHOR = "TensorFlow MUSA Authors"
LICENSE = "Apache 2.0"

# Build configuration
#
# Starting with commit C2 the MUSA backend is shipped as three cooperating
# shared objects (see `CMakeLists.txt` for the split rationale):
#
#   * libmusa_core.so     — allocator / EventPool / telemetry singletons
#   * libmusa_plugin.so   — TensorFlow PluggableDevice entry point
#   * _C.<extsuffix>.so   — Python extension (tensorflow_musa._C)
#
# All three must land inside the installed package directory so that the
# plugin's `$ORIGIN` RPATH resolves `libmusa_core.so` and the Python
# extension imports as `tensorflow_musa._C`.
PLUGIN_LIBRARY = "libmusa_plugin.so"
CORE_LIBRARY = "libmusa_core.so"
BUILD_DIR = "build"


def _python_ext_suffix() -> str:
    """Return the CPython EXT_SUFFIX (e.g. '.cpython-38-x86_64-linux-gnu.so')."""
    import sysconfig

    ext = sysconfig.get_config_var("EXT_SUFFIX")
    return ext if ext else ".so"


PYEXT_LIBRARY = "_C" + _python_ext_suffix()

# Files that must be copied from build/ into the wheel payload (all live
# alongside each other under `site-packages/tensorflow_musa/`).
PACKAGED_ARTIFACTS = [PLUGIN_LIBRARY, CORE_LIBRARY, PYEXT_LIBRARY]

# Supported TensorFlow version range.
#
# The plugin talks to TF exclusively through the PluggableDevice C ABI
# (SP_Platform / SP_StreamExecutor), which TF has kept append-only from the
# 2.5 series onward. In practice 2.6.x -- 2.16.x all share the SE_MAJOR=0 ABI
# that we depend on, so we validate a range instead of a single pin. The
# extra `ALLOWED_EXACT_VERSIONS` knob lets us ship a wheel that has been
# end-to-end tested on a known-good set while still building on others.
MIN_TF_VERSION = "2.6"
MAX_TF_VERSION_EXCLUSIVE = "2.17"
RECOMMENDED_TF_VERSION = "2.6.1"


def _parse_version(ver: str):
    parts = []
    for component in ver.split("+", 1)[0].split("."):
        # Drop any non-numeric suffix like "rc0" so comparisons stay numeric.
        digits = ""
        for ch in component:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def check_tensorflow_version():
    """Verify that TensorFlow is installed within the supported range.

    Returns:
        tuple: (is_installed, version_string or None)

    Raises:
        SystemExit: If TensorFlow is installed but the version is not within
            [MIN_TF_VERSION, MAX_TF_VERSION_EXCLUSIVE).
    """
    try:
        import tensorflow as tf
        version = tf.__version__

        v = _parse_version(version)
        v_min = _parse_version(MIN_TF_VERSION)
        v_max_exc = _parse_version(MAX_TF_VERSION_EXCLUSIVE)

        if v < v_min or v >= v_max_exc:
            print("ERROR: TensorFlow version out of supported range!")
            print(f"  Supported: >= {MIN_TF_VERSION}, < {MAX_TF_VERSION_EXCLUSIVE}")
            print(f"  Installed: {version}")
            print(f"  Recommended (fully tested): {RECOMMENDED_TF_VERSION}")
            sys.exit(1)

        if version != RECOMMENDED_TF_VERSION:
            print(
                f"NOTE: TensorFlow {version} is within the supported range; "
                f"the primary test matrix uses {RECOMMENDED_TF_VERSION}."
            )
        else:
            print(f"TensorFlow {version} found - OK")
        return True, version
    except ImportError:
        print("WARNING: TensorFlow not installed.")
        print(
            f"  Supported range: >= {MIN_TF_VERSION}, < {MAX_TF_VERSION_EXCLUSIVE} "
            f"(recommended: {RECOMMENDED_TF_VERSION})"
        )
        print(f"  Please install: pip install tensorflow=={RECOMMENDED_TF_VERSION}")
        return False, None


class BuildPluginCommand(Command):
    """Build the MUSA plugin shared library using CMake."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Check TensorFlow version before building
        check_tensorflow_version()

        project_root = os.path.abspath(os.path.dirname(__file__))
        build_dir = os.path.join(project_root, BUILD_DIR)

        # Create build directory if it doesn't exist
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        # Run CMake configuration
        cmake_cmd = [
            "cmake",
            "..",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DMUSA_KERNEL_DEBUG=OFF",
        ]

        print(f"Running CMake configuration: {cmake_cmd}")
        result = subprocess.run(cmake_cmd, cwd=build_dir, check=False)
        if result.returncode != 0:
            print("CMake configuration failed. Please ensure MUSA SDK and TensorFlow are installed.")
            sys.exit(1)

        # Run make to build the library
        make_cmd = ["make", f"-j{os.cpu_count()}"]
        print(f"Running make: {make_cmd}")
        result = subprocess.run(make_cmd, cwd=build_dir, check=False)
        if result.returncode != 0:
            print("Make failed.")
            sys.exit(1)

        # Verify every artifact produced by the split build was built and
        # copy each one next to the Python package sources.
        for artifact in PACKAGED_ARTIFACTS:
            src = os.path.join(build_dir, artifact)
            if not os.path.exists(src):
                print(f"Error: {artifact} not found after build at {src}.")
                sys.exit(1)
            dst = os.path.join(project_root, SOURCE_DIR, artifact)
            shutil.copy2(src, dst)
            print(f"  packaged: {dst}")


class BdistWheelCommand(bdist_wheel):
    """Custom bdist_wheel that builds plugin and excludes test directory."""

    def run(self):
        # Check TensorFlow version first
        check_tensorflow_version()

        # Always rebuild the plugin for wheel packaging so the wheel
        # contains a library matching the current source tree.
        project_root = os.path.abspath(os.path.dirname(__file__))
        BuildPluginCommand(self.distribution).run()

        # Force only the tensorflow_musa package (source is in python directory)
        self.distribution.packages = ["tensorflow_musa"]
        self.distribution.package_data = {PACKAGE_NAME: list(PACKAGED_ARTIFACTS)}
        self.distribution.py_modules = None
        # Map tensorflow_musa package name to python source directory
        self.distribution.package_dir = {"tensorflow_musa": SOURCE_DIR}

        # Clean build/lib to only contain tensorflow_musa
        build_lib = os.path.join(project_root, "build", "lib")
        if os.path.exists(build_lib):
            # Remove test directory from build/lib
            test_dir = os.path.join(build_lib, "test")
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            # Remove musa_ext directory
            musa_ext_dir = os.path.join(build_lib, "musa_ext")
            if os.path.exists(musa_ext_dir):
                shutil.rmtree(musa_ext_dir)
            # Remove docs directory
            docs_dir = os.path.join(build_lib, "docs")
            if os.path.exists(docs_dir):
                shutil.rmtree(docs_dir)

        super().run()


# Read long description from README
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return DESCRIPTION


# Check TensorFlow at setup.py load time (before any build commands)
# This ensures version mismatch is detected early
check_tensorflow_version()


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    license=LICENSE,
    # Map package name (tensorflow_musa) to source directory (python)
    package_dir={"tensorflow_musa": SOURCE_DIR},
    # Package name (pip install tensorflow_musa)
    packages=["tensorflow_musa"],
    package_data={
        PACKAGE_NAME: list(PACKAGED_ARTIFACTS),
    },
    python_requires=">=3.7",
    # NOTE: tensorflow is NOT listed in install_requires to prevent pip from
    # downloading it during wheel build. Users must install a supported TF
    # version (>= 2.6, < 2.17; recommended 2.6.1) manually before installing
    # tensorflow_musa. See README.md for installation instructions.
    install_requires=[
        "numpy>=1.19.0",
    ],
    cmdclass={
        "bdist_wheel": BdistWheelCommand,
        "build_plugin": BuildPluginCommand,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="tensorflow musa gpu moore-threads deep-learning",
)

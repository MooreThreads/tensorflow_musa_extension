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
PLUGIN_LIBRARY = "libmusa_plugin.so"
BUILD_DIR = "build"

# Required TensorFlow version
REQUIRED_TF_VERSION = "2.6.1"


def check_tensorflow_version():
    """Check if TensorFlow is installed with the required version.

    Returns:
        tuple: (is_installed, version_string or None)

    Raises:
        SystemExit: If TensorFlow is installed but version doesn't match.
    """
    try:
        import tensorflow as tf
        version = tf.__version__

        if version != REQUIRED_TF_VERSION:
            print(f"ERROR: TensorFlow version mismatch!")
            print(f"  Required: {REQUIRED_TF_VERSION}")
            print(f"  Installed: {version}")
            print(f"  Please install the correct version: pip install tensorflow=={REQUIRED_TF_VERSION}")
            sys.exit(1)

        print(f"TensorFlow {version} found - OK")
        return True, version
    except ImportError:
        print(f"WARNING: TensorFlow not installed.")
        print(f"  Required version: {REQUIRED_TF_VERSION}")
        print(f"  Please install: pip install tensorflow=={REQUIRED_TF_VERSION}")
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

        # Verify the library was built
        plugin_path = os.path.join(build_dir, PLUGIN_LIBRARY)
        if not os.path.exists(plugin_path):
            print(f"Error: {PLUGIN_LIBRARY} not found after build.")
            sys.exit(1)

        # Copy to package directory (source dir is python, but package name is tensorflow_musa)
        package_lib_path = os.path.join(project_root, SOURCE_DIR, PLUGIN_LIBRARY)
        shutil.copy2(plugin_path, package_lib_path)
        print(f"Successfully built and copied to: {package_lib_path}")


class BdistWheelCommand(bdist_wheel):
    """Custom bdist_wheel that builds plugin and excludes test directory."""

    def run(self):
        # Check TensorFlow version first
        check_tensorflow_version()

        # Always rebuild the plugin for wheel packaging so the wheel
        # contains a library matching the current source tree.
        project_root = os.path.abspath(os.path.dirname(__file__))
        BuildPluginCommand(self.distribution).run()

        # Force only tensorflow_musa packages (source is in python directory)
        self.distribution.packages = ["tensorflow_musa", "tensorflow_musa.optimizer"]
        self.distribution.package_data = {PACKAGE_NAME: [PLUGIN_LIBRARY]}
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
    # Package names (pip install tensorflow_musa)
    packages=["tensorflow_musa", "tensorflow_musa.optimizer"],
    package_data={
        PACKAGE_NAME: [PLUGIN_LIBRARY],
    },
    python_requires=">=3.7",
    # NOTE: tensorflow is NOT listed in install_requires to prevent pip from
    # downloading it during wheel build. Users must install tensorflow==2.6.1
    # manually before installing tensorflow_musa.
    # See README.md for installation instructions.
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

"""
Package initialization for MUSA test utilities.
This file ensures that the test package can be imported from any location
and automatically sets up the correct path to find the MUSA plugin.
"""

import os
import sys

# Get the directory of this __init__.py file
_test_package_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_test_package_dir)

# Add project root to Python path if not already present
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Also ensure the parent directory is in path (for cases where tensorflow_musa_extension is a subpackage)
_parent_dir = os.path.dirname(_project_root)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import commonly used test utilities
from .musa_test_utils import MUSATestCase, load_musa_plugin

# Make them available when importing the test package
__all__ = ['MUSATestCase', 'load_musa_plugin']

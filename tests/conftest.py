"""Pytest configuration for DOE-Toolkit tests."""

import sys
from pathlib import Path

# Get absolute path to project root
project_root = Path(__file__).parent.parent.resolve()

# Add project root to Python path at the beginning
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

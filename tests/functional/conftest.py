# -*- coding: utf-8 -*-
"""
Pytest configuration for functional tests.

Adds the tests/functional directory to sys.path so that
cold_night_profile can be imported by name in test files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

"""Re-export baseline feature engineering functions from code/src/utils.py.

This thin wrapper isolates the import path so that other modules do not need
to reference the baseline directly.
"""

import importlib
import sys
import os

# Ensure code/src is on sys.path (it normally is when pipeline runs)
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils import engineer_features, engineer_features_39, engineer_features_158plus39

__all__ = ['engineer_features', 'engineer_features_39', 'engineer_features_158plus39']

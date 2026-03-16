"""Test configuration helpers.

Ensures the project root is on ``sys.path`` so imports like ``import src``
work even when tests are run from within the ``tests`` directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


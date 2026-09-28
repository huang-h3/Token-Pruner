"""Put an uninstalled lmms-eval checkout on ``sys.path`` when LMMS_EVAL_ROOT is set."""

import os
import sys
from pathlib import Path


_root = os.environ.get("LMMS_EVAL_ROOT")
if _root:
    _entry = str(Path(_root).expanduser())
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

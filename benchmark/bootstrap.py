"""Import bootstrap for entry points that are also run as scripts."""

import sys

try:
    from .paths import PROJECT_ROOT
except ImportError:  # Running the script by path.
    from paths import PROJECT_ROOT


for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

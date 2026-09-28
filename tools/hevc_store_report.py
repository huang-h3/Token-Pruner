#!/usr/bin/env python3
"""Report the encoding recipes an HEVC artifact store covers.

    python tools/hevc_store_report.py STORE_DIR
"""

from collections import Counter
import json
from pathlib import Path
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit(__doc__)
    store = Path(argv[0]).expanduser()
    recipes = Counter()
    total_bytes = 0
    for metadata_path in store.glob("artifacts/*/*.json"):
        metadata = json.loads(metadata_path.read_text())
        identity = metadata.get("identity") or {}
        recipe = identity.get("recipe") or {}
        recipes[(
            identity.get("encode_scope"),
            recipe.get("effective_gop_size"),
            metadata.get("actual_encoder"),
        )] += 1
        total_bytes += int((metadata.get("validation") or {}).get("size_bytes", 0))
    print(f"{sum(recipes.values())} artifacts, {total_bytes / 1024 ** 3:.2f} GiB\n")
    print("by (scope, gop, encoder):")
    for recipe, count in recipes.most_common():
        print(f"  {count:6d}  {recipe}")


if __name__ == "__main__":
    main()

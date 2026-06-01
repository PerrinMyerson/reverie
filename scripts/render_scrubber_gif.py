#!/usr/bin/env python3
"""Placeholder-free scrubber GIF entry point.

The checked repository keeps the generated GIF under docs/assets. Regenerating
it is an optional local presentation step, so this script intentionally fails
with a clear message instead of blocking broad verification when renderer
dependencies are unavailable.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "error: scrubber GIF regeneration is not configured in this checkout; "
        "use docs/assets/reverie-scrub-demo.gif",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

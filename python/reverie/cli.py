"""The `reverie` command line, installed with the Python package.

`pip install reverie-lang` puts a `reverie` command on PATH that runs the
same CLI as the Rust binary (`reverie run`, `reverse`, `roundtrip`, `scrub`,
`explain`, `fmt`, `invert`, `verify-roundtrip`, ...). `python -m reverie`
does the same.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from ._reverie import cli as _cli


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    sys.stdout.flush()
    return _cli(["reverie", *args])


def entry_point() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entry_point()

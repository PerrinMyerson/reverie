"""Reverie: a statically typed reversible language, from Python.

>>> import reverie
>>> fib = reverie.loads('''
... from i == 0 do
...   a += b; a <=> b; i += 1
... loop skip until i == n
... ''')
>>> fib.run(n=7, i=0, a=0, b=1)
{'a': 13, 'b': 21, 'i': 7, 'n': 7}
>>> fib.reverse(n=7, i=7, a=13, b=21)
{'a': 0, 'b': 1, 'i': 0, 'n': 7}
"""

from ._types import (
    CheckError,
    ExecutionError,
    ParseError,
    ReverieError,
    SeedError,
    Stack,
)
from ._reverie import Frame, Program, __version__, load, loads, verify

__all__ = [
    "CheckError",
    "ExecutionError",
    "Frame",
    "ParseError",
    "Program",
    "ReverieError",
    "SeedError",
    "Stack",
    "__version__",
    "load",
    "loads",
    "verify",
]

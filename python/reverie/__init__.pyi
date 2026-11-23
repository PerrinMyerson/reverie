"""Type stubs for the ``reverie`` package (hand-written; keep in sync with
``crates/reverie-py/src/lib.rs``)."""

import os
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union, overload

__version__: str

#: A Reverie value as seen from Python: ``int``, ``bool``, a (nested) ``list``
#: of values for arrays, or a :class:`Stack`. Tuples and NumPy arrays are
#: accepted as array inputs; outputs are always lists.
_Value = Union[int, bool, List["_Value"], "Stack"]
_Store = Dict[str, _Value]
_StoreInput = Mapping[str, Any]
_PathLike = Union[str, "os.PathLike[str]"]

class Stack:
    """An immutable integer stack, top first. ``Stack()`` is ``nil``."""

    def __init__(self, items: Iterable[int] = ...) -> None: ...
    @property
    def items(self) -> Tuple[int, ...]: ...
    @property
    def top(self) -> int: ...
    def push(self, value: int) -> Stack: ...
    def pop(self) -> Tuple[int, Stack]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: slice) -> Tuple[int, ...]: ...
    def __bool__(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class ReverieError(Exception):
    message: str
    rendered: str
    line: Optional[int]
    column: Optional[int]
    span: Optional[Tuple[int, int]]
    def __init__(
        self,
        message: str,
        *,
        rendered: Optional[str] = ...,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        span: Optional[Tuple[int, int]] = ...,
    ) -> None: ...

class CheckError(ReverieError): ...
class ParseError(CheckError): ...
class SeedError(ReverieError, ValueError): ...
class ExecutionError(ReverieError): ...

class Frame:
    """One step of :meth:`Program.timeline`."""

    @property
    def step(self) -> int: ...
    @property
    def label(self) -> str: ...
    @property
    def span(self) -> Optional[Tuple[int, int]]: ...
    @property
    def line(self) -> Optional[int]: ...
    @property
    def column(self) -> Optional[int]: ...
    @property
    def store(self) -> _Store: ...

class Execution:
    """The full result of :meth:`Program.execute`."""

    @property
    def store(self) -> _Store: ...
    @property
    def inputs(self) -> List[_Value]: ...
    @property
    def outputs(self) -> List[_Value]: ...
    @property
    def observations(self) -> List[str]: ...

class Program:
    """A parsed, statically checked Reverie program."""

    @property
    def source(self) -> str: ...
    @property
    def path(self) -> Optional[str]: ...
    @property
    def engine(self) -> str: ...
    @property
    def legacy_janus(self) -> bool: ...
    @property
    def types(self) -> Dict[str, str]: ...
    def execute(
        self,
        store: Optional[_StoreInput] = None,
        /,
        *,
        inputs: Optional[Iterable[Any]] = None,
        outputs: Optional[Iterable[Any]] = None,
        backward: bool = False,
    ) -> Execution:
        """Run with I/O tapes; returns store, tapes and show/printf output."""
    def inverse(self) -> Program:
        """The inverse program; running it forward runs this one backward."""
    def explain(self, *, ml: bool = False) -> Dict[str, Any]:
        """The `reverie explain --json` report."""
    def run(self, store: Optional[_StoreInput] = None, /, **vars: Any) -> _Store:
        """Run forward and return the final store."""
    def reverse(self, store: Optional[_StoreInput] = None, /, **vars: Any) -> _Store:
        """Run backward and return the restored store."""
    def roundtrip(self, store: Optional[_StoreInput] = None, /, **vars: Any) -> Dict[str, Any]:
        """Forward then backward; returns the CLI's ``--proof-output`` JSON."""
    def timeline(self, store: Optional[_StoreInput] = None, /, **vars: Any) -> List[Frame]:
        """Every step of a forward run, entry state first."""
    def inverse_source(self) -> str: ...
    def formatted_source(self) -> str: ...

def load(
    path: _PathLike,
    *,
    engine: str = "slot",
    types: Optional[Mapping[str, str]] = None,
    legacy_janus: bool = False,
) -> Program:
    """Parse and check a ``.rev`` file. Raises :class:`CheckError`."""

def loads(
    source: str,
    *,
    engine: str = "slot",
    types: Optional[Mapping[str, str]] = None,
    legacy_janus: bool = False,
) -> Program:
    """Parse and check source text. Raises :class:`CheckError`."""

def verify(
    proof: Union[Mapping[str, Any], _PathLike],
    *,
    source: Union[Program, str, None] = None,
) -> Dict[str, Any]:
    """Check a roundtrip proof like ``reverie verify-roundtrip --json``."""

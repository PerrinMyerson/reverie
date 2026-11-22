"""Pure-Python value and exception types shared with the native extension."""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Tuple, overload

__all__ = [
    "Stack",
    "ReverieError",
    "CheckError",
    "ParseError",
    "SeedError",
    "ExecutionError",
]


class Stack:
    """An immutable Reverie integer stack.

    Items are listed top first, matching ``push``/``pop`` in Reverie: after
    ``push(x, s)`` the pushed value is ``s[0]``. ``Stack()`` is ``nil``.

    Stacks compare equal only to other stacks with the same items, so stores
    returned by :meth:`reverie.Program.run` compare cleanly with ``==``.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[int] = ()) -> None:
        values = tuple(items)
        for value in values:
            if isinstance(value, bool) or not hasattr(value, "__index__"):
                raise TypeError(f"Stack items must be ints, got {type(value).__name__}")
        self._items: Tuple[int, ...] = tuple(int(value) for value in values)

    @property
    def items(self) -> Tuple[int, ...]:
        """The stack's items, top first."""
        return self._items

    @property
    def top(self) -> int:
        """The top item. Raises ``IndexError`` on an empty stack."""
        if not self._items:
            raise IndexError("top of empty Stack")
        return self._items[0]

    def push(self, value: int) -> "Stack":
        """Return a new stack with ``value`` on top."""
        return Stack((value, *self._items))

    def pop(self) -> Tuple[int, "Stack"]:
        """Return ``(top, rest)``. Raises ``IndexError`` on an empty stack."""
        return self.top, Stack(self._items[1:])

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[int]:
        return iter(self._items)

    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: slice) -> Tuple[int, ...]: ...
    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self._items[index]

    def __bool__(self) -> bool:
        return bool(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Stack):
            return self._items == other._items
        return NotImplemented

    def __hash__(self) -> int:
        return hash((Stack, self._items))

    def __repr__(self) -> str:
        if not self._items:
            return "Stack()"
        return f"Stack({list(self._items)!r})"


class ReverieError(Exception):
    """Base class for every error raised by the ``reverie`` package.

    Attributes:
        message: The one-line error message.
        rendered: The full diagnostic as the CLI prints it (without color).
        line, column: 1-based source position, when the error has one.
        span: ``(start, end)`` byte offsets into the source, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        rendered: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
        span: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.rendered = rendered if rendered is not None else f"Error: {message}\n"
        self.line = line
        self.column = column
        self.span = span

    def __str__(self) -> str:
        if self.line is not None and self.column is not None:
            return f"{self.message} (line {self.line}, column {self.column})"
        return self.message


class CheckError(ReverieError):
    """The program failed static checking (``reverie check``)."""


class ParseError(CheckError):
    """The source could not be parsed. A subclass of :class:`CheckError`."""


class SeedError(ReverieError, ValueError):
    """A store value or name is invalid for this program."""


class ExecutionError(ReverieError):
    """Execution failed at runtime (an assertion, a failed ``delocal``, ...)."""

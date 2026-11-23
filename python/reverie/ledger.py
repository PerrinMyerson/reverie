"""Witness ledgers: exact accounting of the information a program discards.

A reversible program keeps, in its witness variables, exactly what an
ordinary program would have thrown away. The ledger measures that witness:

* ``values``: how many integers it holds;
* ``nonzero``: how many of them carry anything;
* ``coded_bits``: the length of a concrete prefix-free code for it
  (zig-zag + Elias-gamma per value). This is an *achievable* storage cost, so
  it upper-bounds the information the computation erased;
* ``landauer_joules``: ``coded_bits * k_B * T * ln 2``, the minimum heat an
  irreversible machine would dissipate erasing that much, at temperature ``T``.

These are operational measures (what you must actually store), not entropy
estimates, so they need no assumptions about the data distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

from ._types import Stack

__all__ = [
    "BOLTZMANN",
    "WitnessEntry",
    "Ledger",
    "code_length_bits",
    "flatten",
    "ledger",
    "landauer_joules",
]

#: Boltzmann constant, J/K.
BOLTZMANN = 1.380649e-23


def landauer_joules(bits: float, temperature_kelvin: float = 300.0) -> float:
    """Landauer's bound: minimum energy to erase ``bits`` bits at ``T``."""
    return bits * BOLTZMANN * temperature_kelvin * math.log(2)


def _zigzag(value: int) -> int:
    return 2 * value if value >= 0 else -2 * value - 1


def code_length_bits(value: int) -> int:
    """Bits to store ``value`` with zig-zag + Elias-gamma coding (1 bit for 0)."""
    z = _zigzag(int(value)) + 1
    return 2 * (z.bit_length() - 1) + 1


def flatten(value: Any) -> Iterator[int]:
    """Every integer in a Reverie value or store (bools count as 0/1)."""
    if isinstance(value, bool):
        yield int(value)
    elif isinstance(value, int):
        yield value
    elif isinstance(value, Stack):
        yield from value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from flatten(item)
    else:
        raise TypeError(f"not a Reverie value: {value!r}")


@dataclass
class WitnessEntry:
    name: str
    values: int
    nonzero: int
    coded_bits: int

    @property
    def raw_bits(self) -> int:
        """Uncompressed size: 64 bits per value."""
        return 64 * self.values


@dataclass
class Ledger:
    entries: List[WitnessEntry] = field(default_factory=list)
    temperature_kelvin: float = 300.0

    @property
    def values(self) -> int:
        return sum(entry.values for entry in self.entries)

    @property
    def coded_bits(self) -> int:
        return sum(entry.coded_bits for entry in self.entries)

    @property
    def raw_bits(self) -> int:
        return sum(entry.raw_bits for entry in self.entries)

    @property
    def landauer_joules(self) -> float:
        return landauer_joules(self.coded_bits, self.temperature_kelvin)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "coded_bits": self.coded_bits,
            "raw_bits": self.raw_bits,
            "landauer_joules": self.landauer_joules,
            "temperature_kelvin": self.temperature_kelvin,
            "entries": [
                {
                    "name": entry.name,
                    "values": entry.values,
                    "nonzero": entry.nonzero,
                    "coded_bits": entry.coded_bits,
                    "raw_bits": entry.raw_bits,
                }
                for entry in self.entries
            ],
        }


def ledger(
    store: Mapping[str, Any],
    names: Iterable[str],
    *,
    temperature_kelvin: float = 300.0,
) -> Ledger:
    """Account for the witness variables ``names`` in a final store."""
    result = Ledger(temperature_kelvin=temperature_kelvin)
    for name in names:
        values = list(flatten(store[name]))
        result.entries.append(
            WitnessEntry(
                name=name,
                values=len(values),
                nonzero=sum(1 for value in values if value != 0),
                coded_bits=sum(code_length_bits(value) for value in values),
            )
        )
    return result


def per_row_bits(rows: Sequence[Any]) -> List[int]:
    """Coded bits of each row of a tape (e.g. one training step each)."""
    return [sum(code_length_bits(value) for value in flatten(row)) for row in rows]

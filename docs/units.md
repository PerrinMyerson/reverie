# Units Of Measure

Integer types can carry compile-time units such as `int<m>`, `int<s>`, and
`int<m/s>`. Unit annotations are erased at runtime, but the checker enforces
dimensional consistency before execution.

Multiplication combines units, division subtracts exponents, and addition or
subtraction requires matching units. Unit exponent syntax such as `m^2` is
accepted in type annotations and integer literals.

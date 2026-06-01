# Arrays

Arrays are fixed-size runtime values. They are useful for demos where a small
flat store is not quite enough, but dynamic allocation would be too much
language for v1.

## Syntax

Array literals use square brackets:

```rev
[10, 20, 30]
[]
```

Indexed reads use an integer expression:

```rev
xs[0]
xs[i + 1]
matrix[i][j]
```

Repeated indexes address nested fixed-size arrays, so Jana-style declarations
such as `int matrix[3][3]` create a two-dimensional value.
Concrete declaration lengths must be positive; use an empty literal `[]` only
for a value whose type or local initializer supplies its shape.
When a global or source declaration has a literal initializer, Reverie checks
that the literal shape matches the declared dimensions before execution.
Nested array literals must be rectangular. A value such as `[[1, 2], [3]]` is
rejected by `check` because the second row has a different length.

Array lengths are read with the Janus-style `size(xs)` expression:

```rev
size(xs)
```

Array types can be written on locals and procedure parameters:

```rev
local xs: array<int> = [10, 20, 30]
  xs[1] += 5
delocal xs = [10, 25, 30]
```

Type-first locals may spell concrete suffix lengths. Those lengths are checked
against literal initializers and delocal assertions, and matching concrete
`local`/`delocal` suffixes must agree:

```rev
local int xs[3] = [10, 20, 30]
  xs[1] += 5
delocal int xs[3] = [10, 25, 30]
```

Units compose through array element types:

```rev
local distances: array<int<m>> = [1<m>, 2<m>]
  distances[1] += 3<m>
delocal distances = [1<m>, 5<m>]
```

See `examples/array_units.rev` for a runnable version.

Top-level arrays are seeded through the CLI:

```sh
cargo run -p reverie-cli -- run examples/array.rev \
  --var 'xs=[10,20,30]' --var delta=5
```

Nested arrays use nested literals:

```sh
cargo run -p reverie-cli -- run examples/skip.rev \
  --var 'matrix=[[1,2],[3,4]]' --type 'matrix=array<array<int>>'
```

The final store is:

```text
{delta = 5, xs = [30, 25, 10]}
```

`examples/array_oob.rev` is the negative companion. It parses and checks, then
fails at runtime with a source label on the bad index:

```sh
cargo run -p reverie-cli -- run examples/array_oob.rev --var 'xs=[1,2,3]'
```

## Reversible Updates

Array elements are assignable places, so the primitive reversible operations
work on them:

```rev
xs[1] += delta
xs[2] -= 3
xs[0] ^= mask
flags[0] ^= true
xs[0] <=> xs[2]
```

Integer array elements support `+=`, `-=`, and dimensionless integer `^=`.
Boolean array elements support boolean `^=`, so an indexed flag can be toggled
by `true` or left unchanged by `false`. See `examples/bool_flags.rev` for a
runnable boolean-array update.

The same no-destruction rule applies, but it is cell-sensitive. For
`xs[i] += e`, the index expression must not mention `xs`, because the inverse
must be able to find the same element after the value changes. The right-hand
side may read another cell from the same array when the checker can prove the
locations differ, such as `xs[0] += xs[1]`. The checker rejects exact same-cell
reads such as `xs[0] += xs[0]`, and it rejects runtime-dependent same-root
reads such as `xs[i] += xs[j]` because those locations may alias.

For swaps, index expressions must not mention either mutated root. For example,
`xs[i] <=> i` is rejected: after the swap, `i` may hold a different value, so
the inverse might look at a different element.

## Runtime Rules

Array length is fixed after creation. Updating or swapping an element preserves
the length. Indexes are zero-based signed `i64` values at the expression level
and must be non-negative and in bounds. Constant negative indexes such as
`xs[-1]` or `xs[1 - 2]` are rejected during checking. Constant out-of-bounds
indexes such as `xs[3]` or `xs[1 + 2]` are also rejected during checking when
declarations, local suffixes, or array literals make the fixed shape known.
Calls propagate known argument shapes through procedure bodies and nested
procedure calls for the same constant-index check. Runtime-dependent indexes
and indexes into externally seeded arrays without type/shape information
remain runtime checks.

The interpreter stores arrays as normal values, so whole arrays can be swapped,
passed through by-reference procedures, compared with `==`/`!=`, and watched in
the scrubber timeline. `size(xs)` returns the fixed length as a dimensionless
`int`.

The reversibility oracle also generates local array programs with computed
`delocal` assertions, then checks that forward execution and backward execution
both return to the empty store. This catches mistakes in local deletion,
indexed updates, swaps, and inversion together.

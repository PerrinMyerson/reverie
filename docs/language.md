# Reverie Language Reference

This document describes the language implemented by the current interpreter
and records the intended direction for later reversible features.
For parser-facing syntax, see `docs/grammar.md`; for command-line usage, see
`docs/cli.md`.

## Current Status

The current implementation is the Phase 6 reversible core plus the v1 array
fill-out:

- `skip`
- statement sequencing with `;`
- standalone runtime assertions with `assert`
- reversible updates: `+=`, `-=`, `^=`, plus `++`/`--` sugar
- swaps with `<=>`
- reversible conditionals with exit assertions
- reversible loops with entry and exit assertions
- Jana-style bounded `iterate ... to ... end` loops
- integer, boolean, and array expressions
- fixed-size arrays with indexed reads, `size(xs)` / `len(xs)`, and reversible
  element updates
- Janus-style integer stacks with `nil`, `push`, `pop`, `empty`, `top`, and
  `size`
- externally seeded variables through the CLI
- source-level globals initialized to zero
- backward execution through mechanical inversion
- reversible tape I/O with `read` and `write`
- Janus/Jana-style `show(x)` and `printf(...)` observations for compatibility
  demos
- procedures with by-reference `call`/`uncall`
- reversible local blocks with `local`/`delocal`
- units of measure on integer literals, local variables, and procedure params
- refinement predicates on local variables and procedure params
- time-travel scrubbing through recorded execution timelines

## Values

`int` is a signed `i64`. Arithmetic uses two's-complement wrapping behavior.
This is deliberate: overflow must remain deterministic and reversible-friendly.

```sh
cargo run -p reverie-cli -- run examples/wrapping.rev --var x=9223372036854775807
cargo run -p reverie-cli -- reverse examples/wrapping.rev --var x=-9223372036854775808
```

```text
{x = -9223372036854775808}
{x = 9223372036854775807}
```
Integer literals can carry units, such as `100<m>` or `10<m/s>`.

`bool` has the values `true` and `false`. Boolean locals, globals, procedure
parameters, and CLI seeds can use `bool` annotations.

Arrays are fixed-size values written with square brackets:

```rev
[10, 20, 30]
[]
```
Nested array literals must be rectangular, so `[[1, 2], [3]]` is rejected
before runtime.

Array element reads use zero-based indexes:

```rev
xs[0]
xs[i + 1]
matrix[i][j]
```

When an array element is being mutated, reads from the same array root must be
provably disjoint. For example, `xs[0] += xs[1]` is accepted, but
`xs[i] += xs[j]` is rejected because `i` and `j` may be equal at runtime.
The same conservative alias rule applies to by-reference procedure arguments:
`call f(xs[0], xs[1])` is accepted, while `call f(xs[0], xs[i])` is rejected
unless the indexes are statically known to differ.

Array lengths use the Janus-style `size(xs)` expression or the familiar
`len(xs)` alias. Both return a dimensionless `int`:

```rev
assert size(xs) == 3
assert len(xs) == 3
```

Stacks hold signed `i64` values. `nil` is the empty stack. `push(x, s)` moves
the integer value in `x` onto the top of `s` and clears `x`; `pop(x, s)`
requires `x == 0`, removes the top value from `s`, and restores it into `x`.
These two statements are mechanical inverses. `empty(s)` / `is_empty(s)`
returns a `bool`, `top(s)` / `peek(s)` returns the top integer, and `size(s)`
returns the stack depth.

```rev
local s: stack = nil
  local x: int = 0
    x += 1;
    push(x, s);
    pop(x, s);
    x -= 1
  delocal x = 0
delocal s = nil
```

Top-level values can be seeded from the CLI:

```sh
cargo run -p reverie-cli -- run examples/fib.rev --var n=7 --var i=0 --var a=0 --var b=1
cargo run -p reverie-cli -- run examples/array.rev --var 'xs=[10,20,30]' --var delta=5
```

Seeded variables can also carry static type/unit annotations:

```sh
cargo run -p reverie-cli -- run examples/top_level_units.rev \
  --var speed=0 --type 'speed=int<m/s>' \
  --var distance=100 --type 'distance=int<m>' \
  --var time=10 --type 'time=int<s>'
```

Backward execution seeds the final store instead:

```sh
cargo run -p reverie-cli -- reverse examples/fib.rev --var n=7 --var i=7 --var a=13 --var b=21
```

`local`/`delocal` adds reversible lexical creation and destruction inside a
program. The current interpreter uses a flat store internally, so a local name
may not shadow an existing variable.

## Globals

Source-level globals are initialized to zero unless the CLI seeds a value with
`--var`.

```rev
global x;
global xs[3];
```

`global x;` creates an integer value `0`. `global xs[3];` creates
`[0, 0, 0]`. Typed globals use the same declaration-shape rule as source
declarations, so both `global flags[3]: bool;` and
`global bool flags[3];` create `[false, false, false]`. Declared globals are
visible inside parameterless procedures, which supports the classic Janus
global-store style:

```rev
global x;

proc main() {
  x += 1
}
```

Globals and source declarations allocate storage, so array lengths must be
explicit there. Unsized `[]` suffixes are accepted only for by-reference
procedure parameters and local values whose initializer supplies the concrete
array shape. Literal initializers must match declared array dimensions, so
`global xs[3] = [1, 2]` is rejected by `check` before runtime. Array type
annotations on globals or source declarations still need a storage length, so
write `global xs[3]: int` or `int xs[3]`, not `global xs: array<int>`.

Older Janus files often omit the `global` keyword and put bare declarations
before the first procedure:

```janus
n x1 x2
psiR[128] psiI[128]

procedure main
  n += 1
```

When the prefix before the first procedure contains only declarations of that
form, Reverie treats them as zero-initialized globals. Files without
procedures keep the normal top-level statement grammar, so `x += 1` is still a
program body rather than a declaration.

If a source file omits the top-level body, Reverie uses `call main()` as the
entry point. For legacy Janus programs that define `main_fwd` but not `main`,
the default entry is `call main_fwd()`. Older demo files that expose `test1`
or `test` without `main` use those as last-resort entry procedures.

## Tape I/O

`read` and `write` are Janus-style I/O operations backed by explicit tapes, so
they stay reversible.

```rev
write x
read x
```

`write x` appends the current value of `x` to the output tape. `read x`
consumes one value from the input tape, stores it in `x`, and appends the old
value of `x` to output. The mechanical inverse uses `unwrite` and `unread`.

`show(x)` and `printf(...)` are Janus/Jana compatibility observations. They
record human-readable output without mutating the store and without using the
reversible tape. `show(x)` emits a line such as `x = 5` or `s = <3, 2, 1]`;
`show(x, y)` emits one line per named value. `printf(...)` supports `%d`
placeholders for dimensionless integer expressions and `%%` for a literal
percent sign:

```rev
printf("%d %d\n", n, x1)
```

```sh
cargo run -p reverie-cli -- run examples/io.rev --input 7
cargo run -p reverie-cli -- reverse examples/io.rev \
  --var x=7 --output 0 --output 0 --output 7
cargo run -p reverie-cli -- scrub examples/io.rev --input 7 --dump
```

## Expressions

Supported expression forms:

```rev
42
-42
true
false
x
xs[0]
[1, 2, 3]
(x + 1)
```

Supported operators, from tighter to looser precedence:

```text
-        numeric negation
! ~      logical not
* */ / % \  multiplication, fixed-point multiplication, division, remainder
+ -
<< >>    bit shifts
< <= > >=
== = != #
&        bitwise and
^ !      bitwise xor
|        bitwise or
&&
||
```

Constant zero divisors are rejected during checking for `/` and `%`. Divisors
computed from variables still remain runtime checks.

Arithmetic and shifts require integers. `a */ b` is the legacy Janus fixed-point
multiply used by the Schrödinger example; it evaluates `(a * b) >> 31` using a
widened intermediate. Bitwise `&`, `^`, and `|` operate on dimensionless
integers; when both operands are booleans they operate as boolean and/xor/or.
Unary `-` preserves the operand's unit and uses the same wrapping `i64`
semantics as binary arithmetic. Comparisons require integers except `==` and
`!=`, which compare full values. Logical operators, including unary `!`,
require booleans.

The symbolic aliases mirror Janus where they do not change the reversible
meaning: `=` equals `==`, `#` equals `!=`, `\` equals `%`, infix `!` equals
`^`, and `~` equals unary `!`. In statement position, Janus-style `x != e` is
accepted as an alias for Reverie's `x ^= e`, and `x : y` is accepted as an
alias for `x <=> y`.

The CLI's `--legacy-janus` mode keeps the same precedence table while enabling
older Janus surface rules such as semicolon comments and case-insensitive
identifiers.

## Statements

`skip` does nothing:

```rev
skip
```

Statements are sequenced with semicolons:

```rev
skip;
skip
```

`assert` checks a predicate and leaves the store unchanged. `assert_eq(a, b)`
and `assert_ne(a, b)` are readable aliases for equality and inequality
assertions:

```rev
assert x == 1
assert_eq(x, 1)
assert_ne(x, 0)
```

Because it keeps all information, an assertion is its own inverse. A false
assertion traps at runtime with a source span.

Reversible updates mutate an existing integer or boolean variable:

```rev
x += 1
x -= delta
x ^= mask
flag ^= true
x++
x--
++x
--x
```

`+=` and `-=` require integers with matching units. `^=` requires either
matching booleans or dimensionless integers. Boolean xor update toggles by
`true` and leaves the flag unchanged by `false`, and the same statement is its
own inverse. The right-hand side must not mention the updated variable.
`x += x` is rejected because it destroys the old value of `x`.
Prefix or postfix `++` and `--` are syntax sugar for `+= 1` and `-= 1`; `fmt`
prints the canonical update form.

The left-hand side can also be an array element:

```rev
xs[1] += delta
xs[0] ^= mask
flags[0] ^= true
xs[1]++
```

Constant negative indexes such as `xs[-1]` or `xs[1 - 2]` are rejected during
checking. Constant out-of-bounds indexes such as `xs[3]` or `xs[1 + 2]` are
also rejected during checking when declarations, local suffixes, or array
literals make the fixed shape known. Runtime-dependent indexes are checked at
runtime because their value depends on the current reversible state.

For `xs[i] += e`, the index expression may not mention `xs`. The right-hand
side may read other cells from `xs`, but the checker rejects exact same-cell
reads when both index paths are constant, such as `xs[0] += xs[0]`. If the
target and read use the same array root and any index path is runtime-dependent,
the checker rejects the update unless the constant indexes prove the cells are
distinct.
This accepts Janus/Jana matrix code such as
`LDU[i][j] -= LDU[i][k]` only when the cells are statically known to be
distinct; otherwise use a temporary or witness cell that makes the dependency
explicit.

Swaps exchange two existing variables:

```rev
x <=> y
swap(x, y)
```

They can also swap array elements:

```rev
xs[0] <=> xs[2]
swap(xs[0], xs[2])
```

Swap index expressions may not mention either mutated root, so `xs[i] <=> i`
is rejected.

Jana-style `iterate` loops are bounded loops with local integer counters:

```rev
iterate int i = 0 to n - 1
  x += i
end

iterate int i = n - 1 by -1 to 0
  x += i
end
```

The bounds are inclusive. Explicit zero steps such as `by 0` are rejected.
Empty ranges execute zero iterations. The iteration counter is introduced for
the body and destroyed at `end`; runtime checks catch programs that mutate the
counter inside the body.

Procedures are top-level definitions with by-reference parameters:

```rev
proc bump(x: int) {
  x += 1
}

call bump(n)
uncall bump(n)
```

For Janus compatibility, parameterless global-store procedures may also use
the `procedure` keyword and parenless calls:

```rev
procedure main {
  call bump;
  uncall bump
}
```

`call` runs the procedure body forward. `uncall` runs the same procedure body
backward by applying the derived inverse. Arguments may be variables or array
elements:

```rev
call bump(xs[0])
call bump(matrix[i][j])
```

For element arguments, the runtime freezes the element location at call entry
and copies the parameter back to that same location on return.
When a procedure is called with a known-shape array, constant indexes inside the
callee and procedures it calls are checked against that argument shape at the
call site.

V1 uses an explicit no-alias by-reference model: duplicate arguments such as
`call f(x, x)` are rejected. Constant duplicate element arguments such as
`call f(xs[1 + 1], xs[2])` are also rejected during checking. Potentially
duplicate same-root element arguments such as `call f(xs[i], xs[j])` or
`call f(matrix[i][j], matrix[k][l])` are rejected unless the constant index
paths prove the cells are distinct. This keeps procedure calls deterministic
under the same no-information-loss rule as updates.
`examples/element_args.rev` shows an accepted constant element call, and
`examples/alias_rejection.rev` is the negative companion.

Local variables are created and destroyed reversibly:

```rev
local t: int = 0
  t += x;
  x += 1
delocal t = x - 1
```

Forward execution evaluates the initializer, introduces `t`, runs the body,
asserts `t == x - 1`, and then removes `t`. The initializer and delocal
assertion may not mention the local variable itself.
Locals also follow the v1 no-shadow rule: a live binding name cannot be reused
by an inner `local`. `examples/local_shadow_rejection.rev` shows the diagnostic.

Array locals can carry annotations such as `array<int>` and `array<bool>`:

```rev
local xs: array<int> = [10, 20, 30]
  xs[1] += 5
delocal xs = [10, 25, 30]
```

The same local can be written in Janus/Jana type-first form:

```rev
local int xs[3] = [10, 20, 30]
  xs[1] += 5
delocal int xs[3] where size(xs) == 3 = [10, 25, 30]
```

The `delocal` annotation is checked against the local type in either native
style (`delocal xs: array<int> where size(xs) == 3 = ...`) or type-first style.
A `delocal` refinement is a final deletion-time assertion, so it can document
and enforce the condition expected at the end of the local block.
Concrete suffix lengths on type-first locals are also checked against literal
initializers and delocal assertions. If both `local` and `delocal` spell a
concrete length for the same dimension, the lengths must match.

Units may be attached to array element types, such as `array<int<m>>`.

`size(xs)` is a normal expression and returns the fixed outer length of the
array or stack as a dimensionless integer.

Units are compile-time annotations on integer types:

```rev
proc pace(distance: int<m>, time: int<s>, speed: int<m/s>) {
  speed += distance / time
}
```

The checker rejects dimension mismatches before runtime. Units are erased by
the interpreter.

Refinements are runtime-checked boolean predicates:

```rev
proc dec(n: int where n >= 0) {
  n -= 1
}

local n: int where n >= 0 = 3
  call dec(n)
delocal n = 2
```

The checker ensures each refinement has boolean type. The interpreter checks
local refinements after initialization and before `delocal`, and procedure
parameter refinements before and after `call`/`uncall`.

## Conditionals

Reversible conditionals carry both an entry condition and an exit assertion:

```rev
if c1 then
  S1
else
  S2
fi c2
```

For Janus compatibility, `then` and `else` clauses may be omitted. An omitted
clause is parsed as `skip`. Conditions are normally boolean expressions; for
legacy Janus bit-mask code, dimensionless integer conditions are also accepted
with nonzero meaning true.

Forward behavior:

- Evaluate `c1`.
- If `c1` is true, run `S1`, then require `c2` to be true.
- If `c1` is false, run `S2`, then require `c2` to be false.

The trailing assertion is what will let the inverse interpreter know which
branch was taken without storing a hidden branch log.

## Loops

Reversible loops are written:

```rev
from c1 do
  S1
loop
  S2
until c2
```

For Janus compatibility, the `do` clause or `loop` clause may be omitted. An
omitted clause is parsed as `skip`.

Forward behavior:

- Require `c1` to be true on first entry.
- Run `S1`.
- If `c2` is true, exit.
- Otherwise run `S2`, require `c1` to be false, and repeat from `S1`.

These checks encode the Janus-style loop discipline needed for deterministic
backward execution.

`examples/loop_assertion_failure.rev` is the negative companion: it keeps the
entry condition true after the loop step, so `reverie run` reports a loop
re-entry assertion failure.

## Backward Execution

Every implemented statement has a derived inverse. The CLI command
`reverie reverse <file>` runs that inverse from the supplied state, and
`reverie invert <file>` prints the derived inverse as source.

Forward:

```sh
cargo run -p reverie-cli -- run examples/fib.rev --var n=7 --var i=0 --var a=0 --var b=1
```

```text
{a = 13, b = 21, i = 7, n = 7}
```

Backward:

```sh
cargo run -p reverie-cli -- reverse examples/fib.rev --var n=7 --var i=7 --var a=13 --var b=21
```

```text
{a = 0, b = 1, i = 0, n = 7}
```

Derived inverse source:

```sh
cargo run -p reverie-cli -- invert examples/fib.rev
```

Procedure/local example:

```sh
cargo run -p reverie-cli -- run examples/proc.rev --var n=4
cargo run -p reverie-cli -- reverse examples/proc.rev --var n=5
```

```text
{n = 5}
{n = 4}
```

Array example:

```sh
cargo run -p reverie-cli -- run examples/array.rev --var 'xs=[10,20,30]' --var delta=5
cargo run -p reverie-cli -- reverse examples/array.rev --var 'xs=[30,25,10]' --var delta=5
```

```text
{delta = 5, xs = [30, 25, 10]}
{delta = 5, xs = [10, 20, 30]}
```

Units example:

```sh
cargo run -p reverie-cli -- check examples/units.rev
cargo run -p reverie-cli -- check examples/unit_mismatch.rev
```

The first command succeeds; the second reports a unit mismatch.

Refinement example:

```sh
cargo run -p reverie-cli -- run examples/refinement.rev
cargo run -p reverie-cli -- run examples/refinement_violation.rev
cargo run -p reverie-cli -- run examples/refinement_units_loop.rev
cargo run -p reverie-cli -- run examples/refinement_units_loop_violation.rev
```

The first and third commands succeed. The second fails because `n >= 0`
becomes false; the fourth fails because a unitful `distance >= 0<m>` predicate
becomes false inside a loop.

Scrubber example:

```sh
cargo run -p reverie-cli -- scrub examples/fib.rev --var n=7 --var i=0 --var a=0 --var b=1
cargo run -p reverie-cli -- scrub examples/fib.rev --var n=2 --var i=0 --var a=0 --var b=1 --dump
```

The first command opens the interactive TUI; the second prints the same
timeline in a plain terminal.

## Example

`examples/fib.rev` computes a reversible Fibonacci pair transform:

```rev
if n != 0 then
  from i == 0 do
    a += b;
    a <=> b;
    i += 1
  loop
    skip
  until i == n
else
  skip
fi n != 0
```

With `n = 7, i = 0, a = 0, b = 1`, the final store is:

```text
{a = 13, b = 21, i = 7, n = 7}
```

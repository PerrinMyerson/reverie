# Reverie Examples

Run commands from the repository root with:

```sh
cargo run -p reverie-cli -- <command>
```

If you have installed the binary, replace that prefix with `reverie`.

## Suggested Tour

1. Start with `skip.rev` to see the smallest valid program and the canonical
   empty store.
2. Run `fib.rev` forward and backward to see the reversible Fibonacci pair
   transform.
3. Print `fib.rev` with `invert` to inspect the mechanically derived inverse.
4. Run `negation.rev`, `janus_operators.rev`, `janus_optional_control.rev`,
   `janus_procedure_syntax.rev`, and `assert.rev` to see Janus-style syntax and
   standalone runtime assertions.
5. Run `injectivized_max.rev` to see how an extra witness bit makes `max`
   reversible.
6. Use `scrub --dump` on `fib.rev` to see the same timeline the TUI renders.
7. Try `proc.rev`, `element_args.rev`, and `procedure_call_loop.rev` to see
   by-reference procedures, reversible locals, and procedure-heavy loops.
8. Try `globals.rev` for Janus-style zero-initialized source globals.
9. Try `increment.rev` for prefix/postfix reversible update sugar, then
   `io.rev` for reversible Janus-style tape I/O.
10. Try `bool_toggle.rev`, `bool_flags.rev`, `array.rev`, `size.rev`, `stack.rev`,
   `janus_stack_reverse.rev`,
   `janus_sort.rev`, `matrix_transpose.rev`, `bit_reversal.rev`,
   `perm_to_code.rev`,
   `janus_automata.rev`, `janus_turing.rev`, `rle_compression.rev`,
   `janus_root.rev`, and `janus_factor.rev` to see data structures and numeric
   operations scale up into classic Janus-style reversible algorithms.
11. Try `units.rev` and `refinement.rev` to tour the v1 safety
   layers.
12. Run the negative examples when you want to see the diagnostics.

## Positive Examples

### `skip.rev`

The smallest Reverie program.

```sh
cargo run -p reverie-cli -- check examples/skip.rev
cargo run -p reverie-cli -- run examples/skip.rev
```

Expected run output:

```text
{}
```

### `fib.rev`

The Phase 1 reversible Fibonacci pair transform. The program expects externally
seeded variables because top-level declarations wait for `local`/`delocal`.

```sh
cargo run -p reverie-cli -- run examples/fib.rev \
  --var n=7 --var i=0 --var a=0 --var b=1
```

Expected output:

```text
{a = 13, b = 21, i = 7, n = 7}
```

Run the same source backward from the final store:

```sh
cargo run -p reverie-cli -- reverse examples/fib.rev \
  --var n=7 --var i=7 --var a=13 --var b=21
```

Expected output:

```text
{a = 0, b = 1, i = 0, n = 7}
```

Inspect the derived inverse source:

```sh
cargo run -p reverie-cli -- invert examples/fib.rev
```

Try a plain-text scrubber timeline:

```sh
cargo run -p reverie-cli -- scrub examples/fib.rev \
  --var n=2 --var i=0 --var a=0 --var b=1 \
  --watch a --watch i --dump
```

### `wrapping.rev`

Shows signed `i64` two's-complement wrapping arithmetic.

```sh
cargo run -p reverie-cli -- run examples/wrapping.rev \
  --var x=9223372036854775807
cargo run -p reverie-cli -- reverse examples/wrapping.rev \
  --var x=-9223372036854775808
```

Expected outputs:

```text
{x = -9223372036854775808}
{x = 9223372036854775807}
```

### `negation.rev`

Shows Janus-style unary numeric negation. Unary `-` is read-only and uses
wrapping `i64` semantics.

```sh
cargo run -p reverie-cli -- run examples/negation.rev
cargo run -p reverie-cli -- reverse examples/negation.rev
```

Expected outputs:

```text
{}
{}
```

### `janus_operators.rev`

Shows accepted Janus-style operator aliases: `=` equality, `#` not-equals, `~`
logical not, bitwise/boolean `&` and `|`, `<<`/`>>` shifts, `\` remainder,
`*/` fixed-point multiply, infix `!` xor, `!=` xor-update, and `:` swap.

```sh
cargo run -p reverie-cli -- run examples/janus_operators.rev
cargo run -p reverie-cli -- reverse examples/janus_operators.rev
```

Expected outputs:

```text
{}
{}
```

### `janus_optional_control.rev`

Shows Janus-style optional control-flow clauses. Omitted `else`, `do`, and
`loop` clauses parse as `skip`.

```sh
cargo run -p reverie-cli -- run examples/janus_optional_control.rev
cargo run -p reverie-cli -- reverse examples/janus_optional_control.rev
```

Expected outputs:

```text
{}
{}
```

### `janus_procedure_syntax.rev`

Shows the Janus-style `procedure` keyword and parameterless `call name` /
`uncall name` syntax for global-store procedures.

```sh
cargo run -p reverie-cli -- run examples/janus_procedure_syntax.rev
cargo run -p reverie-cli -- reverse examples/janus_procedure_syntax.rev \
  --var x=0
```

Expected outputs:

```text
{x = 0}
{x = 0}
```

### `assert.rev`

Shows standalone assertions, readable `assert_eq` / `assert_ne` helpers, and
unary logical not. Assertions check predicates without changing the store, so
they are their own inverse.

```sh
cargo run -p reverie-cli -- run examples/assert.rev \
  --var x=0 --var target=2
cargo run -p reverie-cli -- reverse examples/assert.rev \
  --var x=0 --var target=2
```

Expected outputs:

```text
{target = 2, x = 0}
{target = 2, x = 0}
```

### `bool_toggle.rev`

Shows reversible boolean xor update. `flag ^= true` toggles a boolean flag, and
running the same source backward toggles it back.

```sh
cargo run -p reverie-cli -- run examples/bool_toggle.rev \
  --var flag=false --type flag=bool
cargo run -p reverie-cli -- reverse examples/bool_toggle.rev \
  --var flag=true --type flag=bool
```

Expected outputs:

```text
{flag = true}
{flag = false}
```

### `bool_flags.rev`

Shows reversible indexed updates over `array<bool>`. Boolean `^=` toggles by
`true` and leaves a flag unchanged by `false`, including when the target is an
array element.

```sh
cargo run -p reverie-cli -- run examples/bool_flags.rev \
  --var 'flags=[false,true,false]' --type 'flags=array<bool>'
cargo run -p reverie-cli -- reverse examples/bool_flags.rev \
  --var 'flags=[true,true,true]' --type 'flags=array<bool>'
```

Expected outputs:

```text
{flags = [true, true, true]}
{flags = [false, true, false]}
```

### `injectivized_max.rev`

Shows how a non-injective computation becomes reversible by carrying the
missing branch fact as data.

```sh
cargo run -p reverie-cli -- run examples/injectivized_max.rev \
  --var x=9 --var y=4 --var swapped=0
cargo run -p reverie-cli -- run examples/injectivized_max.rev \
  --var x=4 --var y=9 --var swapped=0
```

Expected outputs:

```text
{swapped = 0, x = 9, y = 4}
{swapped = 1, x = 9, y = 4}
```

Run either final state backward:

```sh
cargo run -p reverie-cli -- reverse examples/injectivized_max.rev \
  --var x=9 --var y=4 --var swapped=0
cargo run -p reverie-cli -- reverse examples/injectivized_max.rev \
  --var x=9 --var y=4 --var swapped=1
```

Expected outputs:

```text
{swapped = 0, x = 9, y = 4}
{swapped = 0, x = 4, y = 9}
```

### `proc.rev`

Shows a by-reference procedure plus a reversible local block.

```sh
cargo run -p reverie-cli -- run examples/proc.rev --var n=4
cargo run -p reverie-cli -- reverse examples/proc.rev --var n=5
```

Expected outputs:

```text
{n = 5}
{n = 4}
```

### `element_args.rev`

Passes statically distinct array elements as by-reference procedure arguments.
Jana rejects this simple `call bump(xs[0], xs[1])` shape, while Reverie checks
that the cells cannot alias and runs it directly.

```sh
cargo run -p reverie-cli -- run examples/element_args.rev --var 'xs=[0,1]'
cargo run -p reverie-cli -- reverse examples/element_args.rev --var 'xs=[1,1]'
```

Expected outputs:

```text
{xs = [1, 1]}
{xs = [0, 1]}
```

### `increment.rev`

Shows prefix/postfix `++` and `--` sugar over the existing reversible `+= 1`
and `-= 1` updates. It works for both variables and indexed array elements.

```sh
cargo run -p reverie-cli -- run examples/increment.rev
cargo run -p reverie-cli -- reverse examples/increment.rev \
  --var x=1 --var 'xs=[-1,1]'
```

Expected outputs:

```text
{x = 1, xs = [-1, 1]}
{x = 0, xs = [0, 0]}
```

### `procedure_call_loop.rev`

Runs a by-reference procedure followed by its matching `uncall` inside a
bounded loop. The final value change is just the counter, making it a compact
procedure dispatch example and a benchmark fixture.

```sh
cargo run -p reverie-cli -- run examples/procedure_call_loop.rev \
  --var n=5 --var i=0 --var x=0 --var y=1
cargo run -p reverie-cli -- reverse examples/procedure_call_loop.rev \
  --var n=5 --var i=5 --var x=0 --var y=1
```

Expected outputs:

```text
{i = 5, n = 5, x = 0, y = 1}
{i = 0, n = 5, x = 0, y = 1}
```

### `globals.rev`

Shows Janus-style source declarations. Declared globals are initialized to zero
or false when the CLI does not seed them, type-first globals use the same
spelling as Jana-style declarations, and parameterless procedures can mutate
them.

```sh
cargo run -p reverie-cli -- run examples/globals.rev
cargo run -p reverie-cli -- reverse examples/globals.rev \
  --var flag=true --var x=1 --var 'xs=[1,2,1]'
```

Expected outputs:

```text
{flag = true, x = 1, xs = [1, 2, 1]}
{flag = false, x = 0, xs = [0, 0, 0]}
```

### `io.rev`

Shows reversible tape I/O. `write` appends to the output tape. `read` consumes
one input tape value, stores it in the target, and writes the target's old
value to output.

```sh
cargo run -p reverie-cli -- run examples/io.rev --input 7
cargo run -p reverie-cli -- reverse examples/io.rev \
  --var x=7 --output 0 --output 0 --output 7
```

Expected outputs:

```text
{x = 7}
output: [0, 0, 7]
{x = 0}
input: [7]
```

### `array.rev`

Shows fixed-size arrays, indexed reversible updates, sized type-first local
arrays, and `swap(xs[0], xs[2])` as a readable alias for element swaps.

```sh
cargo run -p reverie-cli -- run examples/array.rev \
  --var 'xs=[10,20,30]' --var delta=5
cargo run -p reverie-cli -- reverse examples/array.rev \
  --var 'xs=[30,25,10]' --var delta=5
```

Expected outputs:

```text
{delta = 5, xs = [30, 25, 10]}
{delta = 5, xs = [10, 20, 30]}
```

### `size.rev`

Shows fixed array lengths through the readable `len(xs)` alias. `size(xs)`
remains accepted for Janus-style sources. Both read the length and leave the
store unchanged.

```sh
cargo run -p reverie-cli -- run examples/size.rev
cargo run -p reverie-cli -- reverse examples/size.rev
```

Expected outputs:

```text
{}
{}
```

### `stack.rev`

Shows Janus-style reversible integer stacks with readable `is_empty(s)` and
`peek(s)` aliases for stack inspection. `push(x, s)` moves `x` onto the stack
and clears it; `pop(x, s)` requires `x == 0`, restores the top stack value into
`x`, and removes it from the stack.

```sh
cargo run -p reverie-cli -- run examples/stack.rev
cargo run -p reverie-cli -- reverse examples/stack.rev
```

Expected outputs:

```text
{}
{}
```

### `janus_stack_reverse.rev`

Translates the core of Jana's stack-operations example with unbraced procedure
bodies, semicolonless statement streams, type-first procedure parameters,
source declarations, `show(s)`, and `local`/`delocal` declarations. It
recursively reverses a stack, checks the new top, uncalls the reverse
procedure, and clears the stack back to `nil`.

```sh
cargo run -p reverie-cli -- run examples/janus_stack_reverse.rev
cargo run -p reverie-cli -- reverse examples/janus_stack_reverse.rev
```

Expected outputs:

```text
s = <5, 4, 3, 2, 1]
{s = nil}
s = <5, 4, 3, 2, 1]
{s = nil}
```

### `janus_sort.rev`

Translates the classic Janus reversible bubble-sort example. The `list` array
is sorted while the `perm` array records enough information to run the sort
backward.

```sh
cargo run -p reverie-cli -- run examples/janus_sort.rev \
  --var n=5 --var i=0 --var j=0 \
  --var 'list=[5,3,4,1,2]' --var 'perm=[0,0,0,0,0]'
cargo run -p reverie-cli -- reverse examples/janus_sort.rev \
  --var n=5 --var i=0 --var j=0 \
  --var 'list=[1,2,3,4,5]' --var 'perm=[3,4,1,2,0]'
```

Expected outputs:

```text
{i = 0, j = 0, list = [1, 2, 3, 4, 5], n = 5, perm = [3, 4, 1, 2, 0]}
{i = 0, j = 0, list = [5, 3, 4, 1, 2], n = 5, perm = [0, 0, 0, 0, 0]}
```

### `bit_reversal.rev`

Applies an eight-element bit-reversal permutation using a witness permutation
table. The operation swaps each transposition exactly once, so running the same
source backward restores the original array.

```sh
cargo run -p reverie-cli -- run examples/bit_reversal.rev \
  --var n=8 \
  --var 'xs=[10,20,30,40,50,60,70,80]' \
  --var 'perm=[0,4,2,6,1,5,3,7]'
cargo run -p reverie-cli -- reverse examples/bit_reversal.rev \
  --var n=8 \
  --var 'xs=[10,50,30,70,20,60,40,80]' \
  --var 'perm=[0,4,2,6,1,5,3,7]'
```

Expected outputs:

```text
{n = 8, perm = [0, 4, 2, 6, 1, 5, 3, 7], xs = [10, 50, 30, 70, 20, 60, 40, 80]}
{n = 8, perm = [0, 4, 2, 6, 1, 5, 3, 7], xs = [10, 20, 30, 40, 50, 60, 70, 80]}
```

### `matrix_transpose.rev`

Transposes a 3x3 matrix with nested Janus-style `iterate` loops and reversible
off-diagonal swaps. Running the same source backward restores the original
matrix because transpose is its own inverse.

```sh
cargo run -p reverie-cli -- run examples/matrix_transpose.rev \
  --var n=3 \
  --var 'm=[[1,2,3],[4,5,6],[7,8,9]]'
cargo run -p reverie-cli -- reverse examples/matrix_transpose.rev \
  --var n=3 \
  --var 'm=[[1,4,7],[2,5,8],[3,6,9]]'
```

Expected outputs:

```text
{m = [[1, 4, 7], [2, 5, 8], [3, 6, 9]], n = 3}
{m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], n = 3}
```

### `perm_to_code.rev`

Translates Jana's permutation-to-code example. The program converts a
permutation into the factorial-code style representation used by the original
Janus example, and the same source reverses the code back to the permutation.

```sh
cargo run -p reverie-cli -- run examples/perm_to_code.rev \
  --var 'x=[2,0,3,1,5,4]'
cargo run -p reverie-cli -- reverse examples/perm_to_code.rev \
  --var 'x=[0,0,2,1,4,4]'
```

Expected outputs:

```text
{x = [0, 0, 2, 1, 4, 4]}
{x = [2, 0, 3, 1, 5, 4]}
```

### `janus_automata.rev`

Adapts Jana's historic one-dimensional reversible cellular automata example.
It uses legacy bare globals, global-store procedures, indexed array updates,
nested loops, bit operations, and reversible tape output.

```sh
cargo run -p reverie-cli -- run examples/janus_automata.rev
```

Expected output ends with:

```text
output: [100, 100, 0, 0, 100, 100]
```

### `janus_turing.rev`

Adapts Jana's historic reversible Turing machine simulation. It keeps the
classic global-store style and makes the selected-rule guard explicit so the
program satisfies Reverie's strict reversible branch assertions.

```sh
cargo run -p reverie-cli -- run examples/janus_turing.rev
```

Expected output includes the final halting state:

```text
q = 6
```

### `rle_compression.rev`

Builds a run-length encoding table from repeated input values. Since ordinary
compression discards information, the original `data` array remains in the
store as the witness that lets `reverie reverse` clear the compressed
`symbols` and `counts` tables exactly.

```sh
cargo run -p reverie-cli -- run examples/rle_compression.rev \
  --var n=8 --var run=0 \
  --var 'data=[2,2,2,5,5,1,1,1]' \
  --var 'symbols=[0,0,0,0,0,0,0,0]' \
  --var 'counts=[0,0,0,0,0,0,0,0]'
cargo run -p reverie-cli -- reverse examples/rle_compression.rev \
  --var n=8 --var run=3 \
  --var 'data=[2,2,2,5,5,1,1,1]' \
  --var 'symbols=[2,5,1,0,0,0,0,0]' \
  --var 'counts=[3,2,3,0,0,0,0,0]'
```

Expected outputs:

```text
{counts = [3, 2, 3, 0, 0, 0, 0, 0], data = [2, 2, 2, 5, 5, 1, 1, 1], n = 8, run = 3, symbols = [2, 5, 1, 0, 0, 0, 0, 0]}
{counts = [0, 0, 0, 0, 0, 0, 0, 0], data = [2, 2, 2, 5, 5, 1, 1, 1], n = 8, run = 0, symbols = [0, 0, 0, 0, 0, 0, 0, 0]}
```

### `janus_root.rev`

Translates the original Janus integer square-root example. Starting from
`num = N`, it computes `root = floor(sqrt(N))` and leaves the remainder in
`num`; the temporary globals return to zero.

```sh
cargo run -p reverie-cli -- run examples/janus_root.rev --var num=27
cargo run -p reverie-cli -- reverse examples/janus_root.rev \
  --var num=2 --var root=5 --var z=0 --var bit=0
```

Expected outputs:

```text
{bit = 0, num = 2, root = 5, z = 0}
{bit = 0, num = 27, root = 0, z = 0}
```

### `janus_factor.rev`

Translates the original Janus factorization example. Starting from `num > 1`,
it moves prime factors into `fact[1..]`, clears `num`, and returns the
temporaries to zero.

```sh
cargo run -p reverie-cli -- run examples/janus_factor.rev --var num=84
cargo run -p reverie-cli -- reverse examples/janus_factor.rev \
  --var num=0 --var try=0 --var z=0 --var i=0 \
  --var 'fact=[0,2,2,3,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'
```

Expected outputs:

```text
{fact = [0, 2, 2, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], i = 0, num = 0, try = 0, z = 0}
{fact = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], i = 0, num = 84, try = 0, z = 0}
```

### `units.rev`

Checks dimensions statically and erases them before runtime.

```sh
cargo run -p reverie-cli -- check examples/units.rev
cargo run -p reverie-cli -- run examples/units.rev
```

Expected run output:

```text
{}
```

`examples/units_exponents.rev` covers source-level exponents such as `m/s^2`.
`examples/array_units.rev` shows arrays whose elements carry units.

### `top_level_units.rev`

Annotates externally seeded variables with CLI type declarations.

```sh
cargo run -p reverie-cli -- run examples/top_level_units.rev \
  --var speed=0 --type 'speed=int<m/s>' \
  --var distance=100 --type 'distance=int<m>' \
  --var time=10 --type 'time=int<s>'
```

Expected output:

```text
{distance = 100, speed = 10, time = 10}
```

### `refinement.rev`

Checks `where` predicates when locals and procedure parameters are entered and
exited.

```sh
cargo run -p reverie-cli -- run examples/refinement.rev
cargo run -p reverie-cli -- reverse examples/refinement.rev
```

Expected outputs:

```text
{}
{}
```

`refinement_units_loop.rev` combines refinements, units, and a reversible loop:

```sh
cargo run -p reverie-cli -- run examples/refinement_units_loop.rev
cargo run -p reverie-cli -- reverse examples/refinement_units_loop.rev
```

## Negative Examples

These programs are intentionally invalid or intentionally fail at runtime. They
are useful when checking that diagnostics point at the right source.

### Static Rejections

```sh
cargo run -p reverie-cli -- check examples/alias_rejection.rev
cargo run -p reverie-cli -- check examples/irreversible_update.rev
cargo run -p reverie-cli -- check examples/local_shadow_rejection.rev
cargo run -p reverie-cli -- check examples/unit_mismatch.rev
cargo run -p reverie-cli -- check examples/top_level_unit_mismatch.rev \
  --type 'speed=int<m/s>' --type 'distance=int<m>'
cargo run -p reverie-cli -- check examples/static_array_bounds.rev
```

What they show:

- `alias_rejection.rev`: duplicate by-reference call arguments are rejected.
- `irreversible_update.rev`: `x += x` is rejected because it destroys the old
  value of `x`.
- `local_shadow_rejection.rev`: v1 locals may not shadow live names.
- `unit_mismatch.rev`: local unit annotations catch dimensional mistakes.
- `top_level_unit_mismatch.rev`: CLI `--type` annotations also participate in
  unit checking.
- `static_array_bounds.rev`: known array shapes are propagated through nested
  procedure calls so constant out-of-bounds indexes fail during checking.

### Runtime Failures

```sh
cargo run -p reverie-cli -- run examples/loop_assertion_failure.rev \
  --var i=0 --var n=2
cargo run -p reverie-cli -- run examples/if_assertion_failure.rev \
  --var x=0 --var y=0
cargo run -p reverie-cli -- run examples/assert_failure.rev --var x=0
cargo run -p reverie-cli -- run examples/delocal_assertion_failure.rev
cargo run -p reverie-cli -- run examples/naive_max_no_witness.rev \
  --var x=9 --var y=4
cargo run -p reverie-cli -- run examples/proc_runtime_error.rev --var n=1 --var z=0
cargo run -p reverie-cli -- run examples/array_oob.rev --var 'xs=[1,2,3]'
cargo run -p reverie-cli -- run examples/refinement_violation.rev
cargo run -p reverie-cli -- run examples/refinement_units_loop_violation.rev
```

What they show:

- `loop_assertion_failure.rev`: reversible loop re-entry assertions are checked.
- `if_assertion_failure.rev`: conditional exit assertions are checked.
- `assert_failure.rev`: standalone assertion helpers are checked.
- `delocal_assertion_failure.rev`: local destruction assertions are checked.
- `naive_max_no_witness.rev`: witness-free compare-swap fails instead of
  silently becoming ambiguous.
- `proc_runtime_error.rev`: procedure errors report both body and call-site
  context.
- `array_oob.rev`: bad array indexes report a runtime source span.
- `refinement_violation.rev`: refinements are enforced dynamically.
- `refinement_units_loop_violation.rev`: unitful refinements are still runtime
  predicates after static unit checking.

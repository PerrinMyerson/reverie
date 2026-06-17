# Janus Compatibility

Reverie is Janus-inspired, not a claim to have invented reversible imperative
semantics. The compatibility goal is stronger and more useful: support the same
core class of programs as Janus, keep the inversion laws clear, and use Rust to
make execution and tooling faster.

Primary baseline sources:

- PIRC describes Janus86 as a core language with reversible updates, branches,
  and loops, plus later experimental interpreters:
  <https://topps.diku.dk/pirc/?id=janus>
- The original Lutz/Derby Janus note documents declarations, arrays,
  procedures, reversible control flow, read/write, and sample programs:
  <https://revcomp.info/legacy/mpf/rc/janus.html>

## Capability Matrix

| Janus capability | Reverie status |
| --- | --- |
| signed integer arithmetic | supported as signed `i64` with wrapping semantics |
| legacy fixed-point `*/` expression | supported as signed Q31 multiply `(a * b) >> 31` |
| reversible `+=`, `-=`, xor update | supported as `+=`, `-=`, `^=` |
| prefix/postfix increment/decrement convenience | supported as Reverie sugar `++` / `--`, canonicalized to `+= 1` / `-= 1` |
| boolean xor update | supported as `flag ^= true` |
| variable swap | supported as `<=>` |
| unary numeric negation | supported as unary `-` |
| logical negation | supported as unary `!` |
| Janus operator spellings | accepted for equality, not-equality, logical connectives, remainder, bitwise and/or/xor, shifts, xor-update, and swap aliases |
| fixed-size integer arrays | supported as fixed runtime arrays, including nested arrays |
| array size expression | supported as `size(xs)` |
| indexed updates and swaps | supported, including multidimensional indexes |
| integer stacks | supported with `nil`, `push`, `pop`, `empty`, `top`, and `size` |
| reversible `if ... fi` assertions | supported |
| integer bit-mask conditions | supported as dimensionless int conditions with nonzero truth |
| optional Janus `then` / `else` clauses | supported as omitted-clause `skip` |
| reversible `from ... until` loops | supported |
| optional Janus `do` / `loop` clauses | supported as omitted-clause `skip` |
| Jana `iterate ... to ... by ... end` loops | supported for local signed integer counters |
| standalone `assert` statement | supported |
| `call` / `uncall` procedures | supported, with explicit by-reference arguments |
| procedure arguments as lvalues | supported for variables and array elements |
| Janus `procedure` keyword and parenless parameterless calls | supported |
| unbraced Janus/Jana procedure bodies | supported |
| semicolonless Janus/Jana statement streams | supported |
| legacy semicolon comments and case-insensitive identifiers | supported with `--legacy-janus` |
| legacy expression parsing | `--legacy-janus` keeps Reverie's documented precedence while enabling legacy comments/case rules |
| Jana-style self-dependent update aliases | supported with `--legacy-janus` for upstream compatibility |
| Janus/Jana type-first procedure params | supported, including `stack s`, `int n`, and `int xs[]` |
| Janus/Jana source declarations in `main` | supported for scalar ints, multidimensional int arrays, and stacks |
| local variables | supported with `local` / `delocal`, including checked native and type-first annotations plus deletion-time `where` refinements |
| source-level global declarations initialized to zero or false | supported with `global x;`, `global xs[n];`, and type-first `global bool flag;` |
| legacy bare top-level globals | supported before the first procedure, e.g. `n x1 x2` and `psiR[128]` |
| parameterless global-store procedures | supported for declared globals |
| `main` entry convention | supported when the top-level body is omitted |
| legacy `main_fwd` entry convention | supported when `main` is absent |
| legacy demo `test1` / `test` entry convention | supported as a fallback when `main` and `main_fwd` are absent |
| Janus `read` / `write` | supported with explicit reversible input/output tapes |
| Jana `show(x)` / `show(x, y)` observations | supported for named values, separate from reversible tapes |
| Jana `printf(...)` observations | supported for `%d` integer placeholders and `%%` percent escapes |
| runtime trace / inspection | supported through `scrub` and timeline dumps |
| Janus bubble-sort example | supported as `examples/janus_sort.rev` |
| reversible bit-reversal permutation | supported as `examples/bit_reversal.rev` and checked against Jana fixtures |
| Janus permutation-to-code example | supported as `examples/perm_to_code.rev` and direct upstream `perm-to-code.ja` |
| Janus integer square-root example | supported as `examples/janus_root.rev` |
| Janus factorization example | supported as `examples/janus_factor.rev` |
| named Janus implementation benchmark | supported against `kirkedal/Jana-JanusInterp` via `scripts/bench_jana_vs_reverie.py` |
| Janus-style wave-equation benchmark | supported as modern-Jana fixtures in `benchmarks/jana/schroedinger_n16_t100*.ja` |
| Janus-style Turing machine simulation | supported as `examples/janus_turing.rev` with explicit rule-selection guards |

## Current Janus Corpus

`examples/globals.rev` demonstrates source-level globals and a parameterless
procedure mutating the global store. `examples/io.rev` demonstrates reversible
tape-backed `read` and `write`. `examples/negation.rev`,
`examples/janus_operators.rev`, `examples/janus_optional_control.rev`,
`examples/janus_procedure_syntax.rev`, `examples/size.rev`, and
`examples/stack.rev` demonstrate
Janus-style unary negation, operator spellings, optional control-flow clauses,
procedure spellings, fixed-array length expressions, and reversible integer
stack operations. `examples/janus_stack_reverse.rev` translates the recursive
stack reversal core from Jana's stack-operations example and uses Jana-style
unbraced procedure bodies, semicolonless statement streams, type-first
procedure parameters, source declarations, and `local`/`delocal` declarations.
The original `target/jana-baseline/examples/fib.ja`,
`target/jana-baseline/examples/sqrt.ja`,
`target/jana-baseline/examples/stack-operations.ja`,
`target/jana-baseline/examples/factor.ja`,
`target/jana-baseline/examples/perm-to-code.ja`,
`target/jana-baseline/examples/run-length-enc.ja`,
`target/jana-baseline/examples/run-length-enc-stack.ja`,
`target/jana-baseline/examples/matrixmult_v1.0.ja`, and
`target/jana-baseline/examples/test2.ja` now check and run when the Jana
checkout is present. `matrixmult_v1.0.ja` uses `--legacy-janus` because it
contains Jana-style self-dependent array updates; strict Reverie mode still
rejects those aliases. `perm-to-code.ja` and the run-length examples broaden
the direct upstream corpus beyond numerics; `matrixmult_v1.0.ja` exercises
nested arrays, `iterate`, and matrix-style lvalue arguments. `test2.ja`
exercises array-element procedure arguments such as
`call test_rev(test_array[0])`, but Jana itself exits 0 while printing a
local/delocal diagnostic for that translated call, so it remains audit coverage
rather than a checked performance workload. `examples/element_args.rev`
documents the Reverie-native version: statically distinct element arguments
such as `call bump(xs[0], xs[1])` are accepted after no-alias checking even
though the Jana baseline rejects that direct syntax.
`examples/janus_sort.rev` translates the original Janus bubble-sort showcase
into Reverie. It sorts `list` while
carrying a `perm` witness array, which is what makes the sorting operation
injective and therefore reversible. `examples/bit_reversal.rev` covers a
table-driven bit-reversal permutation where each transposition is applied
exactly once. `examples/perm_to_code.rev` translates the original
permutation-to-code example. `examples/janus_root.rev` translates the original
integer square-root example, and `examples/janus_factor.rev` translates the
original factorization example.
`examples/janus_automata.rev` adapts Jana's historic one-dimensional
reversible cellular automata example, covering indexed array updates, nested
loops, bit operations, and reversible tape output.
`examples/janus_turing.rev` adapts Jana's historic reversible Turing machine
simulation. The upstream source uses a guarded-rule idiom where several rules
share the same current state; the Reverie example makes that guard explicit so
the program obeys the strict reversible `if` entry/exit assertion law.

The external benchmark corpus also includes a modern-Jana version of the
classic 50-element reverse-order bubble sort in
`benchmarks/jana/sort_n50_reverse.ja`, checked against Reverie's
`examples/janus_sort.rev`; bit-reversal permutation fixtures in
`benchmarks/jana/bit_reversal_n8*.ja`, checked against Reverie's
`examples/bit_reversal.rev`; a recursive stack cleanup fixture in
`benchmarks/jana/stack_reverse_cleanup.ja`, checked against Reverie's
`examples/janus_stack_reverse.rev`; and
`benchmarks/jana/schroedinger_n16_t100.ja` and its reverse fixture, a compact
integer wave-stencil derived from Jana's legacy
`basicExamples/schroedinger.janus`. The legacy source uses a fixed-point
operator that the current Jana binary no longer parses, so the checked fixture
uses small integer coefficients and ordinary multiplication while preserving
the same reversible two-pass update structure.

Forward:

```sh
cargo run -p reverie-cli -- run examples/janus_sort.rev \
  --var n=5 --var i=0 --var j=0 \
  --var 'list=[5,3,4,1,2]' --var 'perm=[0,0,0,0,0]'
```

Expected output:

```text
{i = 0, j = 0, list = [1, 2, 3, 4, 5], n = 5, perm = [3, 4, 1, 2, 0]}
```

Backward:

```sh
cargo run -p reverie-cli -- reverse examples/janus_sort.rev \
  --var n=5 --var i=0 --var j=0 \
  --var 'list=[1,2,3,4,5]' --var 'perm=[3,4,1,2,0]'
```

Expected output:

```text
{i = 0, j = 0, list = [5, 3, 4, 1, 2], n = 5, perm = [0, 0, 0, 0, 0]}
```

## Remaining Work To Match Janus

The systematic language-feature audit against the Lutz/Derby note lives in
[`docs/janus-audit.md`](janus-audit.md). The older `matrixmult.ja` also runs in
Reverie, but Jana itself prints local/delocal diagnostics for that file, so
`matrixmult_v1.0.ja` is the cleaner benchmark oracle.

Performance claims should be made against named implementations, not against
Janus-the-language. The current internal benchmark compares Reverie's reference
tree walker against its slot-compiled Rust engine. The external benchmark
harness compares Reverie against the public Haskell Jana interpreter and checks
expected outputs before timing.

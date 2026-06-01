# CLI Reference

The `reverie` binary is the main interface for checking, running, reversing,
inverting, explaining, and scrubbing programs.

During development, run it through Cargo:

```sh
cargo run -p reverie-cli -- <command>
```

If the binary is installed, replace the prefix with `reverie`.

## Commands

```text
reverie check [OPTIONS] <FILE>
reverie run [OPTIONS] <FILE>
reverie reverse [OPTIONS] <FILE>
reverie invert [OPTIONS] <FILE>
reverie explain [OPTIONS] <FILE>
reverie fmt [OPTIONS] <FILE>
reverie scrub [OPTIONS] <FILE>
```

## `check`

Parses a source file and runs static checks without executing it.

```sh
cargo run -p reverie-cli -- check examples/units.rev
```

Options:

```text
--type NAME=TYPE
--legacy-janus
```

Use `--type` to annotate externally seeded variables for static type and unit
checking:

```sh
cargo run -p reverie-cli -- check examples/top_level_unit_mismatch.rev \
  --type 'speed=int<m/s>' --type 'distance=int<m>'
```

Duplicate `--type` names are rejected. With `--legacy-janus`, this check runs
after case-insensitive name normalization. Extra `--type` names are also
rejected when the external store is known, which catches typos such as
`--type nn=int` when the program expects `n`.

## `run`

Runs a source file forward from the supplied initial store.

```sh
cargo run -p reverie-cli -- run examples/fib.rev \
  --var n=7 --var i=0 --var a=0 --var b=1
```

Expected output:

```text
{a = 13, b = 21, i = 7, n = 7}
```

Options:

```text
--var NAME=VALUE
--type NAME=TYPE
--engine tree|slot
--input VALUE
--output VALUE
--legacy-janus
```

`run` defaults to `--engine slot`, the compiled slot-indexed interpreter. Use
`--engine tree` to force the reference AST interpreter.

Use `--legacy-janus` for older Janus files that treat every semicolon as a
line comment marker and identifiers as case-insensitive. Seed and type names
are normalized the same way before checking or execution.

If the program contains Jana-style `show(x)` or `printf(...)` statements, those
observation lines are printed before the final store. They are separate from
the reversible `read`/`write` tapes. `printf` output preserves embedded newline
escapes, so `printf("%d\n", x)` prints exactly one newline before the store.

Use `--input` to seed values consumed by `read`. Use `--output` only when
running inverse I/O operations directly.

Duplicate `--var` names are rejected so command lines cannot silently override
an earlier seed.
Seed values also provide static type information for plain ints, booleans,
stacks, and non-empty homogeneous arrays, so `--var flag=true` is enough for
the checker to reject integer arithmetic on `flag`. Use `--type` when a seed
needs units or an empty array's element type is otherwise ambiguous.
Heterogeneous array seeds and tape values, such as `--var xs=[1,true]` or
`--input [1,true]`, are rejected before execution because Reverie arrays are
homogeneous.
Ragged nested arrays, such as `--var matrix=[[1,2],[3]]`, are also rejected so
CLI seeds and tape values obey the same rectangular fixed-size array model as
source literals.
Before execution starts, `run`, `reverse`, and `scrub` also check that every
external store name reported by `explain` has a matching `--var` seed. If one
is missing, the command fails with a seed-template hint instead of waiting for
a runtime undefined-variable error. Extra `--var` names are also rejected when
the external store is known, which catches typos such as `nn=7` when the
program expects `n`.

## `reverse`

Runs the mechanically derived inverse from the supplied final store.

```sh
cargo run -p reverie-cli -- reverse examples/fib.rev \
  --var n=7 --var i=7 --var a=13 --var b=21
```

Expected output:

```text
{a = 0, b = 1, i = 0, n = 7}
```

Options:

```text
--var NAME=VALUE
--type NAME=TYPE
--engine tree|slot
--input VALUE
--output VALUE
--legacy-janus
```

`reverse` also defaults to `--engine slot`.

Use `--output` to seed the output tape produced by the forward run. When
reversing `read`, the consumed values are restored to the input tape.

## `invert`

Prints the mechanically derived inverse as parser-compatible Reverie source.

```sh
cargo run -p reverie-cli -- invert examples/fib.rev
```

Use this when you want to inspect what `reverse` will execute.

Options:

```text
--type NAME=TYPE
--legacy-janus
```

## `fmt`

Parses a source file and prints canonical parser-compatible Reverie source.
This is useful for reviewing generated inverses, normalizing Janus-style input,
or checking that examples stay readable.

```sh
cargo run -p reverie-cli -- fmt examples/fib.rev
```

Options:

```text
--check
--write
--legacy-janus
```

Use `--check` to fail when the file is not already in canonical form:

```sh
cargo run -p reverie-cli -- fmt --check examples/fib.rev
```

Use `--write` to rewrite the file in place:

```sh
cargo run -p reverie-cli -- fmt --write examples/fib.rev
```

`--check` and `--write` are mutually exclusive.
`fmt` preserves the parsed program, not comments or original whitespace.

## `explain`

Parses, checks, and summarizes the reversible constructs used by a source file.
This is a quick way to understand whether a program relies on loops, indexed
mutation, procedures, tape I/O, stacks, or other language features before you
run it.

```sh
cargo run -p reverie-cli -- explain examples/fib.rev
```

Example output:

```text
file: examples/fib.rev
status: reversible program checks
globals: 0
procedures: 0
statements: 8
expressions: 14
features:
- reversible conditionals
- reversible loops
- reversible updates
- swaps
safety checks:
- no additional indexed or arithmetic runtime checks
external store:
- a: inferred at runtime; add --type a=TYPE for units
- b: inferred at runtime; add --type b=TYPE for units
- i: inferred at runtime; add --type i=TYPE for units
- n: inferred at runtime; add --type n=TYPE for units
run template: reverie run examples/fib.rev --var a=0 --var b=0 --var i=0 --var n=0
declared store:
- none
inverse: reverie invert examples/fib.rev
```

Use `--json` for a machine-readable summary with the same counts, feature
names, safety checks, per-safety-check counts, store templates, and command
templates:

```sh
cargo run -p reverie-cli -- explain --json examples/fib.rev
```

When you provide `--type` annotations, `explain` uses them to choose seed
templates. Array templates are shell-quoted once, including nested arrays such
as `--type 'm=array<array<int>>'` producing `--var m='[[0]]'`.
The generated run template also repeats matching `--type` annotations, so a
copy-pasted command preserves unit checks such as `--type 'speed=int<m/s>'`.
Template paths are also shell-quoted when needed, so files under directories
with spaces can still be copied directly into a terminal.
For programs with globals or source declarations, `explain` also prints a
`declared store` section. Those names default from the program, but the
declared override template shows valid optional `--var` values for changing
the initial store.

Options:

```text
--type NAME=TYPE
--json
--legacy-janus
```

## `scrub`

Builds a forward timeline and opens the interactive scrubber TUI.

```sh
cargo run -p reverie-cli -- scrub examples/fib.rev \
  --var n=7 --var i=0 --var a=0 --var b=1 \
  --watch a --watch b --watch i
```

Keys:

```text
left/right or h/l: scrub
home/end: jump to ends
digits+enter: jump to step
q/esc: quit
```

Use `--dump` for non-interactive output:

```sh
cargo run -p reverie-cli -- scrub examples/fib.rev \
  --var n=2 --var i=0 --var a=0 --var b=1 \
  --watch a --watch i --dump
```

Options:

```text
--var NAME=VALUE
--type NAME=TYPE
--watch NAME
--dump
--input VALUE
--output VALUE
--legacy-janus
```

Use `--input` and `--output` with `scrub` the same way as `run` when a
timeline includes reversible tape I/O.
Duplicate `--watch` names are rejected, and each watched name must appear in
at least one timeline frame. This catches typos such as `--watch nn` when the
program stores `n`.

## Seed Values

`--var` values currently support signed `i64` integers, booleans, nested
arrays, and integer stacks:

```sh
--var n=7
--var x=-1
--var flag=true
--var 'xs=[10,20,30]'
--var 'flags=[true,false,true]'
--var 'matrix=[[1,2],[3,4]]'
--var s=nil
--var 's=stack[3,2,1]'
```

Arrays may contain whitespace:

```sh
--var 'xs=[10, 20, 30]'
```

The same value grammar is used by `--input` and `--output`, including the
homogeneous-array check.

The CLI validates that every seed name is a Reverie identifier.

## Type Annotations

`--type` annotations use the same type grammar as source annotations:

```sh
--type 'distance=int<m>'
--type 'time=int<s>'
--type 'speed=int<m/s>'
--type 'flag=bool'
--type 'flags=array<bool>'
--type 'xs=array<int<m>>'
--type 's=stack'
```

`--type` does not create a runtime value. It only tells the checker how to
interpret an externally seeded variable.

## Exit Behavior

Successful commands exit with status 0. Syntax, static check, and span-carrying
runtime errors render `ariadne` diagnostics and exit non-zero. Runtime failures
that cannot be attached to source still print a concise CLI error.

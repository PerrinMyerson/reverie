# Diagnostics

Reverie reports syntax, static semantic, and span-carrying runtime errors with
`ariadne` source snippets. The goal is that a failed `check`, `run`,
`reverse`, or `scrub` command points at the source text that made the program
invalid or caused execution to stop.

## Syntax Errors

Lexer and parser diagnostics come from `reverie-syntax`. They include the
expected token set when the parser can recover enough context:

```sh
cargo run -p reverie-cli -- check bad.rev
```

```text
Error: expected `+=`, `-=`, `<=>`, `[` or `^=`, found end of input
```

## Core Errors

`reverie-core` diagnostics use the span carried by the AST. These cover:

- irreversible updates, such as `x += x` or `xs[0] += xs[0]`
- invalid indexed places, such as `xs[i] <=> i`
- procedure arity and duplicate by-reference arguments
- mismatched `local`/`delocal` names
- local shadowing
- unit mismatches
- non-boolean refinement predicates
- constant division or remainder by zero
- constant array indexes that are negative or out of bounds for a known shape

For example:

```sh
cargo run -p reverie-cli -- check examples/unit_mismatch.rev
```

reports the mismatched expression and labels the source with:

```text
expected unit `m*s^-1`
found unit `m`
```

When both sides of a comparison are in source, type and unit mismatches carry
two labels: one for the expected side and one for the value that did not fit.
Other semantic errors still use a single `semantic check failed here` label.

## Runtime Errors

Runtime diagnostics come from `reverie-interp`. They cover dynamic facts that
the checker intentionally does not prove yet:

- undefined variables
- division or remainder by zero when the divisor is not a known constant zero
- failed branch and loop assertions
- failed standalone assertions
- failed `delocal` assertions
- failed runtime refinements
- runtime-dependent array indexes that are negative or out of bounds, plus constant
  out-of-bounds indexes when the checker does not know the array shape

These errors are deterministic and preserve the reversible semantics, but they
now carry the most specific source span available. For example:

```sh
cargo run -p reverie-cli -- run examples/array_oob.rev --var 'xs=[1,2,3]'
```

reports the out-of-bounds index and labels it with:

```text
runtime failed here
```

Loop assertion failures are reported the same way:

```sh
cargo run -p reverie-cli -- run examples/loop_assertion_failure.rev --var i=0 --var n=2
```

```text
loop re-entry assertion expected false, found true
runtime failed here
```

Some errors that arise before a source statement is executing, such as
duplicate procedure definitions in a manually constructed AST, still fall back
to concise CLI messages.

When a runtime error happens inside a procedure, Reverie reports the failing
inner expression and labels the call site that entered the procedure. For
example:

```sh
cargo run -p reverie-cli -- run examples/proc_runtime_error.rev --var n=1 --var z=0
```

includes:

```text
runtime failed here
while calling `boom` here
```

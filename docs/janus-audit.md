# Janus Feature Audit

This audit maps the original Lutz/Derby Janus86 language note and the PIRC
Janus summary onto Reverie's current implementation. It is meant to keep the
compatibility claim concrete: Reverie should either support a Janus feature,
support it through an explicit modernized spelling, or document why it is out
of scope.

Primary references:

- Lutz and Derby, "Janus: A Time-Reversible Language":
  <https://revcomp.info/legacy/mpf/rc/janus.html>
- PIRC Janus overview:
  <https://topps.diku.dk/pirc/?id=janus>

## Summary

Reverie covers the Janus86 core identified by PIRC: reversible updates,
branches, loops, procedures, and integer stores. It also covers the major
constructs in the Lutz/Derby grammar: global declarations, arrays, procedure
definitions, `call`/`uncall`, reversible `if` and `from` control flow,
`read`/`write`, reversible mutation, swaps, expressions, unary negation, and
logical negation.

Reverie deliberately extends the baseline with units, refinements, stack
values, prefix/postfix `++`/`--` update sugar, a scrubber, better diagnostics, direct
`reverse`, and a checked benchmark harness. The main intentional compatibility
differences are:

- Reverie keeps its modern syntax as the primary form while accepting many
  Janus/Jana spellings, including case-insensitive keywords. The optional
  `--legacy-janus` front end also treats all semicolons as comments and
  normalizes user identifiers case-insensitively.
- Reverie uses stable signed `i64` wrapping semantics instead of the host-size
  integer model of older interpreters.
- Reverie exposes reversible tape I/O through CLI `--input` / `--output`
  rather than interactive terminal prompts.
- Reverie enforces a v1 no-alias/no-shadow model for clearer diagnostics and
  simpler runtime state.

## Audit Matrix

| Janus86 feature | Reverie status | Evidence |
| --- | --- | --- |
| Program-level global declarations | Supported | `global x;`, legacy bare globals, `examples/globals.rev`, direct upstream `fib.ja` / `sqrt.ja` parsing |
| Scalar integer variables initialized to zero | Supported | global initialization tests, `examples/globals.rev` |
| Fixed-size integer arrays | Supported | `examples/array.rev`, `examples/array_units.rev`, direct upstream `matrixmult_v1.0.ja` |
| Zero-based array indexing | Supported | parser/interpreter array tests, `array_oob.rev` |
| Identifier-as-element-zero convention | Supported for Janus/Jana source declarations | direct upstream `sqrt.ja`, `factor.ja`, and `fib.ja` |
| `procedure` definitions | Supported | `examples/janus_procedure_syntax.rev`, direct upstream Jana examples |
| Forward procedure calls | Supported | `call`, parser/core/interpreter/CLI tests |
| Reverse procedure calls | Supported | `uncall`, `reverse`, procedure round-trip tests |
| Forward references to procedures | Supported | procedure lookup after parsing all declarations |
| Recursive procedures | Supported | `examples/janus_stack_reverse.rev` |
| Reversible updates `+=`, `-=` | Supported | grammar, interpreter, examples |
| Janus xor update `!=` | Supported as `^=` and accepted Janus spelling | `examples/janus_operators.rev` |
| Janus swap `:` | Supported as `<=>` and accepted Janus spelling | `examples/janus_operators.rev`, sort examples |
| Update RHS cannot read target | Statically and dynamically checked | core alias tests, `examples/irreversible_update.rev` |
| Swap indexes cannot mention mutated roots | Checked | core swap alias tests |
| Reversible `if ... fi` assertions | Supported | `examples/janus_optional_control.rev`, negative `if_assertion_failure.rev` |
| Optional `then` / `else` clauses | Supported as `skip` | parser tests, `examples/janus_optional_control.rev` |
| Reversible `from ... until` loops | Supported | Fibonacci example, loop round-trip tests |
| Optional `do` / `loop` clauses | Supported as `skip` | parser tests, `examples/janus_optional_control.rev` |
| Loop entry/re-entry assertions | Runtime checked | `examples/loop_assertion_failure.rev` |
| `read` statement | Supported through reversible input tape | `examples/io.rev`, CLI `--input` |
| `write` statement | Supported through reversible output tape | `examples/io.rev`, CLI `--output` |
| Interactive Janus runtime commands | Replaced by CLI subcommands | `check`, `run`, `reverse`, `invert`, `scrub` |
| Symbol inspection / tracing | Replaced and extended | `scrub --dump`, TUI timeline, watch support |
| Decimal integer constants | Supported | lexer/parser tests |
| Signed arithmetic | Supported as `i64` wrapping arithmetic | wrapping tests and examples |
| Binary arithmetic and comparisons | Supported | grammar docs, parser/interpreter tests |
| Logical and bitwise operators | Supported with Janus aliases | `examples/janus_operators.rev` |
| Unary numeric negation | Supported | `examples/negation.rev` |
| Unary logical negation | Supported | `examples/assert.rev` |
| Legacy expression precedence | Uses the same documented precedence in default and `--legacy-janus` modes, matching the upstream example corpus | parser and CLI tests, `docs/grammar.md` |
| Attached semicolon comments | Supported for legacy forms such as `;comment text` | lexer/parser tests, `docs/grammar.md` |
| Case-insensitive keywords | Supported | lexer/parser tests, `docs/grammar.md` |
| Case-insensitive identifiers | Supported with `--legacy-janus`; default Reverie identifiers remain case-sensitive | parser and CLI tests, `docs/grammar.md` |

## Upstream Corpus Coverage

The checked external corpus includes both direct upstream Jana files and
Reverie translations:

- direct upstream `fib.ja`
- direct upstream `sqrt.ja`
- direct upstream `stack-operations.ja`
- direct upstream `factor.ja`
- direct upstream `perm-to-code.ja`
- direct upstream `run-length-enc.ja`
- direct upstream `run-length-enc-stack.ja`
- direct upstream `matrixmult_v1.0.ja`
- translated Janus square root, factorization, sort, stack reversal, and
  cellular automata, plus a strict reversible Turing-machine translation
- table-driven bit-reversal permutation
- translated permutation-to-code conversion
- witness-preserving run-length compression in forward, reverse, and round-trip
  benchmark fixtures
- modern-Jana forward/reverse fixtures for Fibonacci, square root,
  factorization, bit reversal, permutation-to-code, sorting, a wave-kernel
  benchmark, and a Turing-machine benchmark

The benchmark harness verifies expected output before timing. The standard
smoke gate is:

```sh
scripts/verify_janus_performance.py
```

For a broader non-timing inventory of upstream Jana examples, run:

```sh
python3 scripts/audit_jana_examples.py \
  --json-output benchmarks/results/jana-example-audit.json
```

Validate the resulting audit artifact with:

```sh
python3 scripts/check_jana_audit.py \
  benchmarks/results/jana-example-audit.json \
  --min-examples 21 \
  --exact-paths-file benchmarks/jana-example-corpus.txt \
  --expect-timeout 10.0 \
  --expect-jana-bin-suffix target/jana-baseline/bin/janus \
  --expect-reverie-bin-suffix target/release/reverie \
  --min-status reverie:ok:20 \
  --min-status reverie_legacy_janus:ok:20 \
  --expect-total jana:ok:8 \
  --expect-total jana:ok_with_diagnostics:2 \
  --expect-total jana:failed:11 \
  --expect-total reverie:ok:20 \
  --expect-total reverie:failed:1 \
  --expect-total reverie_legacy_janus:ok:20 \
  --expect-total reverie_legacy_janus:failed:1 \
  --expect-status reverie:basicExamples/turing.janus:failed \
  --expect-status reverie_legacy_janus:basicExamples/turing.janus:failed
```

The current audit covers 21 upstream files: the current `janus -m64` frontend
exits successfully for 10 of them, with 8 clean `ok` runs and 2
`ok_with_diagnostics` runs (`examples/matrixmult.ja` and `examples/test2.ja`)
that print local/delocal diagnostics while exiting with status 0. Reverie runs
20 files in default mode, and Reverie `--legacy-janus` also runs 20. The one
Reverie failure is the upstream `basicExamples/turing.janus`, which reaches an
exit assertion failure at runtime because its guarded-rule idiom is looser than
Reverie's strict branch assertion law. Reverie covers that algorithm through
`examples/janus_turing.rev` and `benchmarks/jana/turing_binary_inc*.ja`, where
the selected-rule guard is explicit. The audit includes `examples/test2.ja`
because it exercises array-element procedure arguments, but it is not promoted
into the checked benchmark corpus while Jana itself reports that translated
temporary local cannot be deleted cleanly.

## Remaining Compatibility Work

The remaining Janus-compatibility work is mostly about exact legacy surface
syntax rather than reversible-language capability:

- Keep default parsing modern while using `--legacy-janus` for older sources
  where every semicolon begins a comment and user identifiers are
  case-insensitive.
- Keep expression precedence modern in both default and `--legacy-janus` modes
  so legacy surface parsing does not misparse upstream guards such as
  `j = n - 2` or `i = 0 && j = 0`.
- Use `scripts/audit_jana_examples.py` to promote additional upstream examples
  into checked benchmark or CLI regression coverage after each gap is explained.
- Keep adding direct upstream or historically documented examples when they
  exercise a distinct feature.

These are compatibility polish items; they are not blockers for the current
v1 reversible semantics or for the checked benchmark corpus.

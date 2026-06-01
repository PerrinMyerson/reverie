# Reversibility and Oracle Testing

Reverie's center of gravity is the rule that every primitive statement must be
invertible. The current implementation derives mechanical inverses for the
statement core, array element operations, procedure calls, and local blocks,
then uses that inverse for backward execution. Units of measure sit above this
machinery and are erased before runtime. Refinements are runtime assertions
attached to bindings.

## Inversion Rules

The current inverse is pure AST transformation:

```text
invert(skip)        = skip
invert(assert c)    = assert c
invert(x += e)      = x -= e
invert(x -= e)      = x += e
invert(x ^= e)      = x ^= e
invert(x <=> y)     = x <=> y
invert(xs[i] += e)  = xs[i] -= e
invert(xs[i] <=> y) = xs[i] <=> y
invert(call f(xs))  = uncall f(xs)
invert(uncall f(xs)) = call f(xs)
invert(S1; S2)      = invert(S2); invert(S1)
```

Conditionals swap entry and exit assertions:

```text
invert(if c1 then S1 else S2 fi c2)
  = if c2 then invert(S1) else invert(S2) fi c1
```

Loops do the same:

```text
invert(from c1 do S1 loop S2 until c2)
  = from c2 do invert(S1) loop invert(S2) until c1
```

Local blocks swap their initializer and deletion assertion:

```text
invert(local x = e S delocal x = e')
  = local x = e' invert(S) delocal x = e
```

Expressions are not inverted; they are duplicated into the inverse statement.
For scalar updates, the core checker rejects expressions that mention the
target. For indexed places such as `xs[i]`, the index expression also must not
mention `xs`, and the checker rejects right-hand side reads from the same root
unless constant indexes prove the read and write cells are distinct. Reads of
other cells, such as `xs[0] += xs[1]`, are allowed when the locations are
statically proven different. Swaps apply the same
idea to both mutated roots: `xs[i] <=> i` is rejected because the inverse could
otherwise look at a different element. The checker also rejects local
initializers and delocal assertions that mention the local variable itself.

Procedures use a no-alias call model in v1: duplicate by-reference arguments
and potentially duplicate same-root element arguments are rejected rather than
merged into shared cells. Constant indexes can prove two element arguments are
distinct, such as `call f(xs[0], xs[1])`; dynamic same-root pairs such as
`call f(xs[i], xs[j])` are rejected before runtime. Locals likewise use a
no-shadow model. These restrictions keep the runtime store flat and make the
mechanical inverse easy to audit.

Unit annotations do not change inversion. For example, `x += 1<m>` still
inverts to `x -= 1<m>`, and the runtime sees only the integer value after the
unit checker has accepted the program.

Refinements also do not change the structural inverse. A refined local keeps
the same predicate when inverted; only the initializer/delocal expressions and
body direction change.

## Injectivization

Some useful operations are not reversible until you keep the information they
would otherwise discard. A sorting `max/min` pair is the teaching example:
after sorting, `x = 9, y = 4` could have come from either original order.
`examples/injectivized_max.rev` stores that missing branch fact in a `swapped`
witness:

```rev
if x < y then
  x <=> y;
  swapped += 1
else
  skip
fi swapped != 0
```

The witness doubles as the conditional exit assertion, so the inverse can pick
the correct branch without a hidden history log. See `docs/injectivization.md`
for the full walkthrough.

## Backward Execution

Backward execution is intentionally small:

```text
execute_backward(program, state) = execute(invert(program), state)
```

That keeps the interpreter honest. There is one forward execution semantics,
and reverse execution reuses it through the inverse AST.

For example:

```sh
cargo run -p reverie-cli -- reverse examples/fib.rev \
  --var n=7 --var i=7 --var a=13 --var b=21
```

prints:

```text
{a = 0, b = 1, i = 0, n = 7}
```

To inspect the derived inverse instead of running it, use:

```sh
cargo run -p reverie-cli -- invert examples/fib.rev
```

The command prints parser-compatible Reverie source, so the output can be
checked, run, or used as a debugging artifact.

## Oracles

The broader evaluation checklist lives in `docs/evaluation.md`.

The test suite checks three reversibility laws:

```text
invert(invert(P)) == P
```

and:

```text
execute(P, S) = S'
execute_backward(P, S') = S
```

and:

```text
execute(uncall f(args), S) == execute(invert(body_of_f), S)
```

The first law is fuzzed over generated ASTs that include the current statement
forms, including procedure calls and local blocks. The second law is fuzzed
over generated straight-line reversible programs, integer stores, fixed-size
integer and boolean array stores with indexed updates/swaps, simple generated
procedure calls, dynamically valid conditionals, and finite Janus-style loops.
`examples/fib.rev`
covers the readable Fibonacci conditional/loop path, `examples/proc.rev` covers
`call`/`uncall` and `local`/`delocal`, `examples/injectivized_max.rev` covers a
branch-witness injectivization pattern, and `examples/array.rev` covers a
readable fixed-size array scenario. `examples/negation.rev` and
`examples/janus_operators.rev` cover Janus-style unary negation and operator
aliases, `examples/janus_optional_control.rev` covers omitted Janus
control-flow clauses, `examples/janus_procedure_syntax.rev` covers Janus-style
procedure spellings, and `examples/size.rev` covers array length reads. The
`examples/stack.rev` covers reversible `push`/`pop` transfer through an
integer stack, and `examples/janus_stack_reverse.rev` covers recursive
procedure use over stack parameters. The array oracle also generates boolean
array xor/swap programs and `local`/`delocal` programs whose deletion
assertions are computed from the generated update sequence. The uncall fidelity
law is fuzzed across generated one-step procedure bodies using `+=`, `-=`, and
`^=`.

Future phases should extend these generators whenever the language grows.

## Scrubbing

The scrubber builds a forward timeline of labeled states and lets the user move
the cursor backward and forward through those states. This is separate from the
semantic inverse used by `reverie reverse`, but both rely on the same
deterministic reversible core.

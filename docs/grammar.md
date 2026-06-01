# Grammar Reference

This page is the compact parser-facing reference for the implemented v1
language. The prose reference in `docs/language.md` explains the semantics;
this file focuses on accepted surface syntax.

## Lexical Rules

Whitespace is ignored. Line comments start with `//` and continue to the end of
the line. Block comments are written `/* ... */`, matching Jana examples.
For legacy Janus sources, an attached semicolon comment such as `;comment text`
also continues to the end of the line. A semicolon followed by whitespace
remains a Reverie statement separator, as in `x += 1; y += 1`.
When the CLI is run with `--legacy-janus`, every semicolon starts a comment to
the end of the line, matching the older Janus surface rule.

Identifiers match:

```text
[A-Za-z_][A-Za-z0-9_]*
```

The following words are reserved and are recognized case-insensitively for
legacy Janus compatibility. User-defined identifiers remain case-sensitive.
When the CLI is run with `--legacy-janus`, user-defined identifiers are also
normalized case-insensitively.

```text
array assert bool by call delocal do else empty end false fi from if int iterate local loop nil pop printf proc procedure push show size skip stack then to top true uncall until where
```

Integer literals are decimal, non-negative source tokens that must fit in
signed `i64`. Negative source values are written with unary `-`, such as
`-1`. CLI seed values may also be negative, such as `--var x=-1`.

String literals are currently accepted for `printf(...)` format strings. They
support `\n`, `\r`, `\t`, `\"`, `\\`, and `\0` escapes.

## Program Shape

```ebnf
program        ::= global-decl* procedure* statement? EOF
               | legacy-global-decl+ procedure+ EOF

global-decl    ::= "global" ident ("[" int-literal "]")*
                   (":" type)? ("=" expr)? ";"?
                 | "global" type ident ("[" int-literal "]")*
                   ("=" expr)? ";"?
legacy-global-decl
               ::= ident ("[" int-literal "]")*
procedure      ::= ("proc" | "procedure") ident ("(" parameter-list? ")")?
                   ("{" statement "}" | statement)
parameter-list ::= parameter ("," parameter)* ","?
parameter      ::= ident type-annotation?
                 | type ident array-suffix* ("where" expr)?
```

Globals must appear before procedures and the top-level body. A scalar global
is written `global x;`; an array global is written `global xs[10];`. Typed
globals can use either Reverie annotation syntax, such as
`global flags[3]: bool;`, or Janus/Jana-style type-first syntax, such as
`global bool flags[3];`. Matrix globals use repeated dimensions, such as
`global m[3][3];`. For old Janus files, a procedure-prefixed program may also
start with bare zero-initialized globals such as `n x1 x2` or
`psiR[128] psiI[128]`; Reverie normalizes those to explicit `global`
declarations before parsing procedures.
Procedures must appear before the top-level body. Procedure parameters are
by-reference names, not values. If the body is omitted, Reverie uses
`call main()` as the entry point, or `call main_fwd()` for legacy Janus files
that define `main_fwd` but not `main`. As a final legacy-demo fallback,
procedure-only files may default to `test1` or `test`.

## Statements

```ebnf
statement      ::= statement-item (";"* statement-item)* ";"*

statement-item ::= "skip"
                 | "assert" expr
                 | ("assert_eq" | "assert_ne") "(" expr "," expr ","? ")"
                 | place update-op expr
                 | place ("++" | "--")
                 | ("++" | "--") place
                 | place "<=>" place
                 | "swap" "(" place "," place ","? ")"
                 | "push" "(" place "," ident ")"
                 | "pop" "(" place "," ident ")"
                 | "call" ident ("(" argument-list? ")")?
                 | "uncall" ident ("(" argument-list? ")")?
                 | "read" place
                 | "unread" place
                 | "write" place
                 | "unwrite" place
                 | "show" "(" ident ")"
                 | "printf" "(" string-literal printf-args? ")"
                 | declaration
                 | if-statement
                 | loop-statement
                 | iterate-statement
                 | local-statement

update-op      ::= "+=" | "-=" | "^="
argument-list  ::= place ("," place)* ","?
printf-args    ::= "," expr ("," expr)* ","?
declaration    ::= type ident ("[" int-literal "]")* ("=" expr)?
```

Semicolons may separate statements, and a trailing semicolon is allowed. They
are optional between unambiguous adjacent statement items, so Jana-style
newline-separated bodies parse even though whitespace itself is ignored.

Leading declarations in parameterless `main()` are hoisted into the program
store, matching Jana examples such as `int n = 50`, `int xs[3] = {1, 2, 3}`,
and `stack s`. A declaration outside hoisted `main()` initializes the name if
it is absent and otherwise asserts that the current value equals the declared
initializer/default. Concrete storage declarations and globals require explicit
positive integer lengths for array suffixes; unsized `[]` is reserved for
by-reference procedure parameters and local values whose initializer supplies
the shape.
Array type annotations such as `array<int>` are accepted for parameters and
locals, but globals and source declarations must still spell concrete storage
dimensions with suffixes such as `[3]`.

Reversible conditionals:

```ebnf
if-statement   ::= "if" expr ("then" statement)? ("else" statement)? "fi" expr
```

Omitted `then` or `else` clauses are parsed as `skip`, matching Janus's
nullable statement clauses.

Reversible loops:

```ebnf
loop-statement ::= "from" expr ("do" statement)? ("loop" statement)? "until" expr
```

Omitted `do` or `loop` clauses are parsed as `skip`.

Janus/Jana bounded iteration:

```ebnf
iterate-statement ::= "iterate" "int" ident "=" expr ("by" expr)? "to" expr
                      statement
                      "end"
```

`iterate` uses inclusive bounds. The default step is `1`; descending loops use
`by -1`. Explicit zero steps such as `by 0` are rejected. Empty ranges, such as
`iterate int i = 0 to -1`, execute zero iterations. The loop variable is local
to the iteration body.

Reversible local blocks:

```ebnf
local-statement ::= "local" ident type-annotation? "=" expr
                    statement
                    "delocal" ident type-annotation? "=" expr
                  | "local" type ident array-suffix* ("where" expr)? "=" expr
                    statement
                    "delocal" type ident array-suffix* ("where" expr)? "=" expr
```

The `delocal` identifier must match the `local` identifier. The checker also
rejects locals that shadow a live name. The second form accepts Janus/Jana-style
type-first declarations such as `local int x = 0 ... delocal int x = 0`.
Sized suffixes on type-first local arrays are checked when the initializer or
delocal assertion is an array literal, and contradictory concrete sizes such as
`local int xs[3] ... delocal int xs[2]` are rejected. Concrete local suffixes
must also be positive, so `local int xs[0]` is rejected by `check`. Unsized
suffixes remain available for locals whose initializer supplies the shape. `delocal`
annotations are checked against the local type, and an optional `where`
refinement is asserted immediately before the local is destroyed. Both
Reverie-style `delocal x: int where x > 0 = 1` and type-first
`delocal int x where x > 0 = 1` forms are accepted.

## Places

```ebnf
place          ::= ident ("[" expr "]")*
```

A place is assignable. It can be a variable such as `x` or an array element
such as `xs[0]`. Repeated indexes address nested arrays, such as `m[i][j]`.
The checker rejects potentially aliasing same-root element mutations and
by-reference arguments unless a constant index difference proves the elements
are distinct.

## Types and Refinements

```ebnf
type-annotation ::= ":" type ("where" expr)?
array-suffix    ::= "[" "]"

type            ::= "int" unit-suffix?
                  | "bool"
                  | "array" "<" type ">"
                  | "stack"
```

Refinements currently require a type annotation because `where` is part of
`type-annotation` for Reverie-style `name: type` bindings. Type-first
Janus/Jana-style parameters and locals may write the refinement after the
identifier, for example `int n where n >= 0`. Empty `[]` suffixes after a
type-first parameter or local identifier are parsed as array wrappers, so
`int xs[]` is equivalent to `xs: array<int>` in those positions.

Examples:

```rev
x: int
distance: int<m>
speed: int<m/s>
flag: bool
xs: array<int>
flags: array<bool>
distances: array<int<m>>
s: stack
n: int where n >= 0
int fact[]
stack s
```

## Units

```ebnf
unit-suffix ::= "<" unit ">"
unit        ::= unit-factor (("*" | "/") unit-factor)*
unit-factor ::= ident ("^" int-literal)?
```

Unit exponents are non-negative source integers. Division negates the following
factor internally, so `m/s^2` represents `m*s^-2`.

## Expressions

```ebnf
expr          ::= or-expr
or-expr       ::= and-expr ("||" and-expr)*
and-expr      ::= bit-or-expr ("&&" bit-or-expr)*
bit-or-expr   ::= bit-xor-expr ("|" bit-xor-expr)*
bit-xor-expr  ::= bit-and-expr (("^" | "!") bit-and-expr)*
bit-and-expr  ::= equality-expr ("&" equality-expr)*
equality-expr ::= comparison-expr (("==" | "=" | "!=" | "#") comparison-expr)*
comparison-expr
              ::= shift-expr (("<" | "<=" | ">" | ">=") shift-expr)*
shift-expr    ::= sum-expr (("<<" | ">>") sum-expr)*
sum-expr      ::= product-expr (("+" | "-") product-expr)*
product-expr  ::= unary-expr (("*" | "*/" | "/" | "%" | "\\") unary-expr)*
unary-expr    ::= ("-" | "!" | "~")* atom
atom          ::= int-literal unit-suffix?
                | "true"
                | "false"
                | "nil"
                | ("empty" | "is_empty") "(" ident ")"
                | ("top" | "peek") "(" ident ")"
                | ("size" | "len") "(" ident ")"
                | ident ("[" expr "]")*
                | "[" expr-list? "]"
                | "{" expr-list? "}"
                | "(" expr ")"
expr-list     ::= expr ("," expr)* ","?
```

Binary operators are left-associative within each precedence level. Prefix
unary operators bind tightest. From tightest to loosest precedence:

```text
- ! ~
* */ / % \\
+ -
<< >>
< <= > >=
== = != #
&
^ !
|
&&
||
```

`--legacy-janus` keeps this precedence table while enabling older Janus surface
rules such as case-insensitive identifiers and semicolon comments. Upstream
Janus and Jana examples rely on comparisons binding looser than arithmetic and
logical `&&`.

## Reversibility Restrictions

The grammar admits some programs that the core checker rejects. Important
static restrictions include:

- `x++`, `++x`, `x--`, and `--x` parse as `x += 1` or `x -= 1`.
- `x += e`, `x -= e`, and `x ^= e` require `e` not to mention `x`.
  Janus-style `x != e` is accepted as an alias for `x ^= e`.
- `x += e` and `x -= e` require integer operands with compatible units.
  `x ^= e` requires matching boolean operands or dimensionless integer
  operands.
- `xs[i] += e` requires `i` not to mention `xs`. The right-hand side may read
  other cells of `xs` only when constant indexes prove the read and write cells
  are distinct.
- Swap index expressions may not mention either mutated root.
- `push(x, s)` and `pop(x, s)` require `x` to be an int place and `s` to be a
  stack; `pop` also requires the target place to hold zero at runtime.
- `printf(...)` accepts `%d` integer placeholders and `%%` literal percent
  signs. The checker rejects unsupported format specifiers, argument-count
  mismatches, and non-dimensionless arguments.
- `call f(x, x)` is rejected by the v1 no-alias procedure model, as are
  constant duplicate element arguments such as `call f(xs[1 + 1], xs[2])`.
- `call f(xs[i], i)` is rejected because an argument index mentions another
  mutable argument. Potentially duplicate same-root element arguments such as
  `call f(xs[i], xs[j])` or `call f(m[i][j], m[k][l])` are rejected unless
  constant indexes prove the locations are distinct.
- `local x ...` is rejected if `x` is already live.
- Unit and type mismatches are rejected before runtime.
- `!e` requires `e` to be a bool expression.
- `if`, `from`, `until`, and `assert` conditions accept bool expressions and,
  for legacy Janus compatibility, dimensionless int expressions where nonzero
  means true.
- Refinement predicates must type-check as `bool`.

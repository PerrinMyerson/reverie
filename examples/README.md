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
   `janus_sort.rev`, `matrix_transpose.rev`, `tensor_linear.rev`,
   `invertible_coupling.rev`, `triangular_residual.rev`, `reversible_preprocess.rev`,
   `reversible_normalize.rev`, `reversible_clamp.rev`, `reversible_pack.rev`,
   `reversible_inference_trace.rev`,
   `mnist_identify.rev`, `mnist_reversible_step.rev`,
   `mnist_witness_tape.rev`, `mnist_witness_tape_loop.rev`,
   `mnist_mlp_witness.rev`, `bit_reversal.rev`, `perm_to_code.rev`,
   `janus_automata.rev`, `janus_turing.rev`, `rle_compression.rev`, `janus_root.rev`, and
   `janus_factor.rev` to see data structures and numeric operations scale up
   into classic Janus-style reversible algorithms.
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

### `tensor_linear.rev`

Shows array-backed tensor annotations and reversible tensor accumulation. The
integer path computes a 2x3 by 3x2 matrix product; the Q31 path uses
`matmul_q31`, which shares the fixed-point multiply semantics of `*/`.

```sh
cargo run -p reverie-cli -- run examples/tensor_linear.rev
cargo run -p reverie-cli -- reverse examples/tensor_linear.rev \
  --var 'y=[[58,64],[139,154]]' \
  --var 'qy=[[1073741824]]'
```

Expected outputs include:

```text
y = [[58, 64], [139, 154]]
qy = [[1073741824]]
y = [[0, 0], [0, 0]]
qy = [[0]]
```

### `invertible_coupling.rev`

Shows a RevNet-style additive coupling block over two 4-wide Q31 activation
halves. `right` is updated from `left`, then `left` is updated from the new
`right`; the inverse runs those reversible updates backward without a witness
tape for the layer itself.

```sh
cargo run -p reverie-cli -- run examples/invertible_coupling.rev
cargo run -p reverie-cli -- reverse examples/invertible_coupling.rev \
  --var 'left=[2684354560,-805306368,1073741824,-805306368]' \
  --var 'right=[2147483648,536870912,0,-536870912]'
```

Expected outputs include:

```text
left = [2684354560, -805306368, 1073741824, -805306368]
right = [2147483648, 536870912, 0, -536870912]
left = [2147483648, -1073741824, 536870912, -536870912]
right = [0, 1073741824, -536870912, 536870912]
```

### `triangular_residual.rev`

Shows an invertible triangular residual block over one 4-wide Q31 activation.
Earlier coordinates accumulate residual terms from later coordinates only. The
inverse restores later coordinates first, then subtracts those same residuals,
so the layer itself needs no witness tape.

```sh
cargo run -p reverie-cli -- run examples/triangular_residual.rev
cargo run -p reverie-cli -- reverse examples/triangular_residual.rev \
  --var 'x=[1610612736,-805306368,536870912,-671088640]'
```

Expected outputs include:

```text
x = [1610612736, -805306368, 536870912, -671088640]
x = [2147483648, -1073741824, 536870912, -536870912]
```

### `reversible_preprocess.rev`

Shows a reversible preprocessing block for small deterministic ML inputs. The
raw Q31 sample stays intact while a separate `features` buffer accumulates a
centered and permuted view. The inverse clears `features` exactly without a
witness tape because the raw sample, mean vector, and swaps are still
available.

```sh
cargo run -p reverie-cli -- run examples/reversible_preprocess.rev
cargo run -p reverie-cli -- reverse examples/reversible_preprocess.rev \
  --var 'features=[1073741824,-1073741824,1610612736,536870912]'
```

Expected outputs include:

```text
features = [1073741824, -1073741824, 1610612736, 536870912]
features = [0, 0, 0, 0]
```

### `reversible_normalize.rev`

Shows Q31 feature normalization as a replayable preprocessing step.
`normalize_q31(raw, mean, inv_scale)` computes `(raw - mean) */ inv_scale`
elementwise into a separate `features` tensor. The inverse clears `features`
without touching the raw sample or normalization parameters.

```sh
cargo run -p reverie-cli -- run examples/reversible_normalize.rev
cargo run -p reverie-cli -- reverse examples/reversible_normalize.rev \
  --var 'features=[536870912,1073741824,536870912,1073741824]'
```

Expected outputs include:

```text
features = [536870912, 1073741824, 536870912, 1073741824]
features = [0, 0, 0, 0]
```

### `reversible_clamp.rev`

Shows how to make Q31 clipping auditable. `clamp_q31` produces the clipped
feature view, while `residual` records `raw - clipped` as a witness. That keeps
the saturated signal exact: backward execution clears the clipped view and
residual while preserving the raw sample.

```sh
cargo run -p reverie-cli -- run examples/reversible_clamp.rev
cargo run -p reverie-cli -- reverse examples/reversible_clamp.rev \
  --var 'clipped=[-1073741824,-268435456,536870912,1073741824]' \
  --var 'residual=[-1073741824,0,0,1073741824]'
```

Expected outputs include:

```text
clipped = [-1073741824, -268435456, 536870912, 1073741824]
residual = [-1073741824, 0, 0, 1073741824]
clipped = [0, 0, 0, 0]
residual = [0, 0, 0, 0]
```

### `reversible_pack.rev`

Shows compact feature packing for bit/token-style inputs. `pack_bits` converts
an 8-wide bit tensor into a scalar payload, and `unpack_bits` records the
decoded tensor as a witness. Backward execution clears the compact and decoded
views while preserving the exact source flags.

```sh
cargo run -p reverie-cli -- run examples/reversible_pack.rev
cargo run -p reverie-cli -- reverse examples/reversible_pack.rev \
  --var packed=77 \
  --var 'unpacked=[1,0,1,1,0,0,1,0]'
```

Expected outputs include:

```text
packed = 77
unpacked = [1, 0, 1, 1, 0, 0, 1, 0]
packed = 0
unpacked = [0, 0, 0, 0, 0, 0, 0, 0]
```

### `reversible_inference_trace.rev`

Combines reversible preprocessing with a tiny Q31 linear classifier. The raw
sample is centered and permuted into `features`, `vecmat_q31` records class
logits, while `top_k_indices`, `top_k_values`, `argmax`, `runner_up`,
`top2_margin`, `rank_of`, `argmax_eq`, and `top_k_contains` record the
deterministic ranked classes, ranked logits, prediction, runner-up class,
confidence margin, true-label rank, top-1 label check, and top-k label check. The
inverse clears `features`, `logits`, `top_classes`, `top_logit_values`,
`prediction`, `runner_up_class`, `margin`, `label_rank`, `correct`, and
`top2_correct` while preserving the raw sample and model.

```sh
cargo run -p reverie-cli -- run examples/reversible_inference_trace.rev
cargo run -p reverie-cli -- reverse examples/reversible_inference_trace.rev \
  --var 'features=[1073741824,-1073741824,1610612736,536870912]' \
  --var 'logits=[1073741824,1610612736,536870912]' \
  --var 'top_classes=[1,0,2]' \
  --var 'top_logit_values=[1610612736,1073741824,536870912]' \
  --var prediction=1 \
  --var runner_up_class=0 \
  --var margin=536870912 \
  --var label_rank=1 \
  --var correct=1 \
  --var top2_correct=1
```

Expected outputs include:

```text
logits = [1073741824, 1610612736, 536870912]
top_classes = [1, 0, 2]
top_logit_values = [1610612736, 1073741824, 536870912]
prediction = 1
runner_up_class = 0
margin = 536870912
label_rank = 1
correct = 1
top2_correct = 1
logits = [0, 0, 0]
top_classes = [0, 0, 0]
top_logit_values = [0, 0, 0]
runner_up_class = 0
margin = 0
label_rank = 0
top2_correct = 0
features = [0, 0, 0, 0]
```

### `mnist_identify.rev`

Checks a full MNIST-shaped reversible Q31 linear classifier inference step. It
keeps `logits`, `prediction`, and `correct` in the final state as witnesses so
the inverse can uncompute the identification exactly.

```sh
cargo run -p reverie-cli -- check examples/mnist_identify.rev
```

### `mnist_reversible_step.rev`

Checks a full MNIST-shaped reversible Q31 linear classifier/training step. The
program uses `image: tensor<int, 784>`, `weights: tensor<int, 784, 10>`,
`bias: tensor<int, 10>`, `logits/error: witness<tensor<int, 10>>`,
`vecmat_q31` for inference, `argmax` and `argmax_eq` for
identification/accuracy, and `outer_q31` plus `scale_q31` for a reversible
single-sample update. `logits` and `error` remain as witnesses so the inverse
can restore mutated parameters before uncomputing the forward pass.

```sh
cargo run -p reverie-cli -- check examples/mnist_reversible_step.rev
cargo run -p reverie-cli --bin reverie-mnist-linear -- --self-test
```

### `mnist_witness_tape.rev`

Shows the same reversible MNIST-shaped training idea with the witness trace
stored as first-class `witness<tensor<...>>` state. The example keeps two
samples' `logits_tape`, `error_tape`, `prediction_tape`, and `correct_tape`
entries inside the program state, so the inverse can restore the previous
model from the final model plus those tensor witnesses.
`reverie explain --json` reports the static witness proof cost for this state
as `witness_metrics`, while `run --json` and `reverse --json` emit
`witness_proof` fingerprints for the concrete witness values.

```sh
cargo run -p reverie-cli -- check examples/mnist_witness_tape.rev
cargo run -p reverie-cli -- explain examples/mnist_witness_tape.rev --json
```

### `mnist_witness_tape_loop.rev`

Shows the batched form of the same trace pattern. An `iterate int sample = 0 to
len(labels) - 1` loop indexes `images`, `labels`, and each
`witness<tensor<...>>` by `sample`, so the source follows the dataset tensor
length instead of hand-writing one block per sample.
For the two-sample demo, `explain --json` reports 44 shaped witness `int`
cells, or 352 logical payload bytes.

```sh
cargo run -p reverie-cli -- check examples/mnist_witness_tape_loop.rev
```

### `mnist_mlp_witness.rev`

Extends the auditable ML pattern from a linear classifier to a tiny MLP-shaped
kernel. The program keeps hidden preactivations, Q31 ReLU masks, hidden
activations, output errors, hidden backprop values, and hidden deltas in
first-class witness tensor tapes before updating `w1/b1` and `w2/b2`. Its
sample loop is also bounded by `len(labels) - 1`, which keeps the activation
and gradient witnesses aligned with the dataset shape.

```sh
cargo run -p reverie-cli -- check examples/mnist_mlp_witness.rev
```

Use `--vars-json PATH` with `run`, `reverse`, or `scrub --dump` to override
large tensor globals such as `images`, `w1`, `b1`, `w2`, and `b2` from exact
integer JSON when experimenting with externally trained Q31 weights.
`scripts/export_q31_mlp_vars.py` converts common PyTorch-style `fc1`/`fc2`
state dictionaries or `.npz` members into that seed shape:

```sh
python3 scripts/export_q31_mlp_vars.py \
  --input target/mlp-state-dict.json \
  --output target/mlp-vars.json \
  --input-scale float
cargo run -p reverie-cli -- run examples/mnist_mlp_witness.rev \
  --vars-json target/mlp-vars.json \
  --json > target/mlp-run.json
python3 scripts/check_q31_mlp_witness.py \
  --vars-json target/mlp-vars.json \
  --run-output-json target/mlp-run.json \
  --json > target/mlp-check-report.json
python3 scripts/summarize_mnist_ml_profile.py target/mlp-check-report.json \
  --output target/mlp-check-report.md
```

The checker emits the `deterministic_q31_mlp_witness_replay` proof-cost shape
in JSON mode, including witness bytes, trace bytes, replay bytes, recompute
steps, recomputed update bytes, and the checked `witness_proof` fingerprint.
The profile summary renderer turns that same report into a compact Markdown
proof-cost table.

With local IDX files, the host runner drives those Reverie programs over real
MNIST data:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --train-images data/train-images-idx3-ubyte \
  --train-labels data/train-labels-idx1-ubyte \
  --test-images data/t10k-images-idx3-ubyte \
  --test-labels data/t10k-labels-idx1-ubyte \
  --epochs 1 \
  --reverse-check
```

`--reverse-check` keeps a compact witness tape of sample indexes, logits,
errors, predictions, and labels, reports the trace payload size, then runs the
Reverie inverse over every training step to prove the final weights restore to
the initial model.

Add `--json` to emit the same run as a machine-readable audit artifact with
train/eval throughput, witness-trace bytes, reverse-check cost, loaded dataset
payload, model payload, estimated payload, peak RSS when the platform exposes
it, and a `proof` section that makes the replay tradeoff explicit: model
bytes, saved sample bytes, witness bytes, full replay payload bytes, per-step
replay bytes, and forward/inverse recompute steps:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- --self-test --json
```

Use `--audit-output PATH` when you want a replayable proof bundle instead of
only metrics. The bundle stores the final model plus per-step witnesses,
including the image bytes used by each step, along with SHA-256 fingerprints of
the source programs, final model, witness trace, report, unsigned payload, and
stable computation identity. `--verify-audit PATH` recomputes those
fingerprints, replays the saved trace forward to prove the witnesses and final
model agree with fresh Reverie execution, then replays backward to prove the
initial model is restored. The bundle also carries the proof-cost summary, and
the verifier recomputes it, so the artifact records the storage/recompute
shape of its own replay proof without trusting the stored numbers.
`--export-model AUDIT --model-output PATH` replays that audit bundle forward
and backward, then writes the final Q31 weights as a signed standalone model
bundle with source/proof provenance and model payload bytes. `--verify-model
PATH` later checks the model shape, model payload bytes, fingerprints, source
audit provenance, and embedded training proof-cost summary without replaying
the full training trace.
`--import-model-json PATH --model-output PATH` wraps an externally trained
Q31 linear model in the same signed model-bundle format. The source JSON can be
either `{ "weights": ..., "bias": ... }` or `{ "model": { "weights": ...,
"bias": ... } }`; the verifier records `provenance_kind: "external_import"`
and, when the source file is available, checks its hash and model contents
instead of claiming a Reverie training proof.
`--scan-audit PATH --audit-limit N` summarizes
correctness, witness mismatches, trace bytes, largest updates, and the
lowest-margin training step. It emits per-label summaries, top confusion
pairs, and separate ranked views for suspicious steps, low-margin close calls,
and large updates so you can find the training steps worth inspecting.
Optional scan gates such as `--audit-max-witness-mismatches 0`,
`--audit-min-margin 0`, `--audit-min-accuracy 95`,
`--audit-max-weight-delta N`, and `--audit-require-label-coverage` turn those
metrics into a pass/fail replay policy for CI or repeatable experiments.
`--inspect-audit PATH --audit-step N` reconstructs the selected step's
before/after model window by reversing from the saved final model, then shows
the witnesses, top logits, winner-vs-runner-up margin, and update deltas for
that step. Its JSON report includes a `debug_contract` proving the witness,
model window, observed deltas, and explanatory state line up for stepping
backward from the model update. Add `--step-output PATH` to save that one
update as a standalone replay bundle whose verifier reruns the step forward
and backward without the full trace. Step bundles carry their own signed proof
object for model bytes,
sample bytes, witness bytes, derived update bytes, replay bytes, and
forward/inverse recompute steps; `--verify-step` recomputes that proof before
accepting the artifact. `--inspect-inference PATH
--audit-step N` runs the saved final model against the selected sample through
`mnist_identify.rev`; add `--inference-output PATH` to save that inference as a
standalone replay bundle whose verifier reruns forward and backward inference
without the training bundle. Inference reports include deterministic Q31
attribution for the predicted class: winning digit, runner-up, margin, bias,
reconstructed logit, reconstructed margin, the largest signed active-pixel
contributions for the winning logit, and the largest signed active-pixel
contributions for the winner-vs-runner-up margin. The standalone verifier
recomputes those contributions from the saved model and sample. The inference
bundle reports the memory/recompute tradeoff directly:
model bytes, sample bytes, witness/result bytes, zero trace bytes, replay
payload bytes, runtime state bytes, and the forward/inverse recompute count.
It also carries a signed `proof` object for the deterministic Q31 inference
claim; `--verify-inference` rebuilds that proof from the saved model, sample,
result, and fresh Reverie forward/backward execution before accepting the
artifact, and when referenced model, training-audit, sample, or evaluation
artifacts are available it also checks the embedded inputs against those signed
sources. Inference reports also carry an `explanation_contract` with the claim
`q31_inference_prediction_explanation`; it ties the prediction to the logits,
checks the label result, reconstructs the winning logit and margin from
attribution rows, and records that reverse replay restores the initial
model/sample state. Verification reports extend the same contract with proof,
result, and source-input checks.
`--inspect-model-inference MODEL --sample-audit AUDIT --audit-step N` runs the
same deterministic inference from a signed model bundle while borrowing the
image and label from an audited sample. Use `--sample-json PATH` instead when
the input is a new labeled sample object with `image_u8` and `label`; the
resulting inference bundle embeds the sample and records the sample JSON
fingerprint. Add `--standalone-rev-output PATH` to any inspected inference
path to emit a self-contained `.rev` classifier with the selected Q31 image,
weights, bias, label, and expected prediction embedded as global tensor
literals. That file can then be checked, run, or roundtripped with the normal
`reverie` CLI, so the final classification step is plain Reverie source rather
than a dedicated MNIST runner. `--export-samples AUDIT --samples-output PATH`
exports audited
images and labels into the same sample-set JSON shape accepted by
`--evaluate-model`, and `--verify-samples PATH` checks the sample-set
fingerprint, shape, and signed lineage proof. Exported samples retain
`audit_step` and `source_sample_index`, and evaluation rows preserve that
lineage with a per-sample fingerprint. When the referenced source audit file is
available, `--verify-samples` also checks the exported pixels, labels, and
source sample indexes against the named audit steps. `--evaluate-model MODEL --samples-json PATH` runs the
signed model over a JSON array or `samples` object of labeled inputs, reports
deterministic accuracy and lowest-margin rows, and records aggregate
sample/witness proof bytes. `--scan-evaluation PATH` ranks incorrect and
lowest-margin evaluation rows while keeping that sample lineage visible.
`--inspect-evaluation PATH --evaluation-row N` expands one signed row into a
full recomputed inference audit and can save it as a standalone replay bundle
with `--inference-output PATH`; `--verify-inference` checks that replay bundle
against the referenced evaluation row when the evaluation bundle is available.
Evaluation gates such as `--eval-min-accuracy 99`,
`--eval-min-margin 0`, `--eval-max-incorrect 0`, and
`--eval-require-label-coverage` turn that report into a pass/fail deployment
policy. Add `--evaluation-output PATH` to save the model, embedded samples,
rows, proof-cost fields, gate policy, and gate result as a signed evaluation
bundle; `--verify-evaluation PATH` reruns every sample and reapplies the stored
gate. When the referenced model bundle and samples file are available, it also
checks that the embedded model and samples match those signed source artifacts.
`--compare-artifacts PATH...` verifies full training, standalone model,
extracted step, inference, and model evaluation proof artifacts, then compares
their actual file bytes against logical payload bytes and recompute counts. In
JSON mode, the `ml_profile` block aggregates model/sample/witness/trace/update
payload bytes, total forward/inverse recompute steps, total recompute steps,
and trace-to-model plus witness-to-model payload ratios:

```sh
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --self-test --audit-output target/mnist-self-test-replay-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-audit target/mnist-self-test-replay-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --export-model target/mnist-self-test-replay-bundle.json \
  --model-output target/mnist-self-test-model-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --import-model-json target/q31-linear-model.json \
  --model-output target/imported-q31-linear-model-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-model target/mnist-self-test-model-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --scan-audit target/mnist-self-test-replay-bundle.json --audit-limit 5
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-audit target/mnist-self-test-replay-bundle.json --audit-step 0
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-audit target/mnist-self-test-replay-bundle.json \
  --audit-step 0 \
  --step-output target/mnist-self-test-step-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-step target/mnist-self-test-step-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-inference target/mnist-self-test-replay-bundle.json --audit-step 0
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-inference target/mnist-self-test-replay-bundle.json \
  --audit-step 0 \
  --inference-output target/mnist-self-test-inference-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-inference target/mnist-self-test-inference-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-model-inference target/mnist-self-test-model-bundle.json \
  --sample-audit target/mnist-self-test-replay-bundle.json \
  --audit-step 0 \
  --inference-output target/mnist-self-test-model-inference-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-model-inference target/mnist-self-test-model-bundle.json \
  --sample-audit target/mnist-self-test-replay-bundle.json \
  --audit-step 0 \
  --standalone-rev-output target/mnist-standalone-classifier.rev
cargo run -p reverie-cli -- run target/mnist-standalone-classifier.rev --json
cargo run -p reverie-cli -- roundtrip target/mnist-standalone-classifier.rev --json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-model-inference target/mnist-self-test-model-bundle.json \
  --sample-json target/mnist-sample.json \
  --inference-output target/mnist-sample-inference-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --export-samples target/mnist-self-test-replay-bundle.json \
  --samples-output target/mnist-samples.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-samples target/mnist-samples.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --evaluate-model target/mnist-self-test-model-bundle.json \
  --samples-json target/mnist-samples.json \
  --eval-min-accuracy 99 \
  --eval-min-margin 0 \
  --eval-max-incorrect 0 \
  --evaluation-output target/mnist-evaluation-bundle.json \
  --json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --verify-evaluation target/mnist-evaluation-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --scan-evaluation target/mnist-evaluation-bundle.json \
  --evaluation-limit 5
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --inspect-evaluation target/mnist-evaluation-bundle.json \
  --evaluation-row 0 \
  --inference-output target/mnist-evaluation-row-inference-bundle.json
cargo run -p reverie-cli --bin reverie-mnist-linear -- \
  --compare-artifacts \
  target/mnist-self-test-replay-bundle.json \
  target/mnist-self-test-model-bundle.json \
  target/mnist-samples.json \
  target/mnist-self-test-step-bundle.json \
  target/mnist-self-test-inference-bundle.json \
  target/mnist-evaluation-bundle.json
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
cargo run -p reverie-cli -- run examples/janus_factor.rev --var num=840
cargo run -p reverie-cli -- reverse examples/janus_factor.rev \
  --var num=0 --var try=0 --var z=0 --var i=0 \
  --var 'fact=[0,2,2,2,3,5,7,0,0,0,0,0,0,0,0,0,0,0,0,0]'
```

Expected outputs:

```text
{fact = [0, 2, 2, 2, 3, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], i = 0, num = 0, try = 0, z = 0}
{fact = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], i = 0, num = 840, try = 0, z = 0}
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

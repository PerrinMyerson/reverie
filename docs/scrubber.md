# Scrubber TUI

The scrubber is the Phase 6 time-travel interface. It builds a deterministic
timeline of runtime states, then lets you move through that timeline like a
video.

## Interactive Mode

```sh
cargo run -p reverie-cli -- scrub examples/fib.rev \
  --var n=7 --var i=0 --var a=0 --var b=1
```

The TUI shows:

- source code on the left
- the current runtime state on the right
- a scrubber gauge at the bottom
- the currently executing source line highlighted when span information is
  available

Keys:

```text
left/right or h/l: scrub
home/end: jump to ends
digits+enter: jump to step
q/esc: quit
```

## Dump Mode

For tests, CI, demos in plain terminals, and documentation, use `--dump`:

```sh
cargo run -p reverie-cli -- scrub examples/fib.rev \
  --var n=2 --var i=0 --var a=0 --var b=1 --watch a --watch i --dump
```

The dump prints the same timeline frames the TUI uses:

```text
   0/11  start              {a = 0, b = 1, i = 0, n = 2}
   1/11  if then            {a = 0, b = 1, i = 0, n = 2}
   ...
watch a: 0:0 -> 3:1 -> 8:2 -> 9:1
watch i: 0:0 -> 5:1 -> 10:2
```

## Timeline Semantics

The timeline starts with the initial state. It records state changes and major
control-flow markers while executing forward:

- `skip`
- reversible updates
- swaps
- branch selection
- loop entry/repeat/exit
- `call` and `uncall` enter/exit
- procedure body frames in the procedure's local parameter store
- `local` and `delocal`
- reversible tape I/O labels for `read`, `unread`, `write`, and `unwrite`

Seed tape values with `--input` and `--output`, matching the `run` and
`reverse` commands:

```sh
cargo run -p reverie-cli -- scrub examples/io.rev --input 7 --dump
```

Watched variables are rendered as compact change timelines. In the interactive
TUI, pass `--watch name` one or more times. In dump mode, the same watch lines
are appended after the frame list.

Procedure calls are traced through their bodies instead of collapsed into one
opaque frame. While the cursor is inside a procedure, the source pane
highlights the procedure body statement and the state pane shows the
procedure-local parameter names. When the procedure exits, the timeline returns
to the caller's store.

Scrubbing backward in the TUI moves to earlier recorded states. The language
itself still supports real backward execution through `reverie reverse`; the
scrubber is a UI over a forward timeline, which is enough for the v1 demo and
keeps the implementation easy to reason about.

## README GIF

The README demo GIF is generated from the real `fib.rev` scrubber timeline,
not from a hand-authored mock:

```sh
python3 scripts/render_scrubber_gif.py
```

The script runs `reverie scrub examples/fib.rev --dump`, renders a compact SVG
frame for each timeline state, and stitches the frames into
`docs/assets/reverie-scrub-demo.gif`. It uses macOS `sips` for SVG
rasterization and ImageMagick `magick` for GIF assembly.

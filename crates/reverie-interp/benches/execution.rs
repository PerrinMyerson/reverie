use criterion::{Criterion, black_box, criterion_group, criterion_main};
use reverie_core::invert_program;
use reverie_interp::{CompiledProgram, IoState, State, Value, build_timeline, execute, execute_io};
use reverie_syntax::parse_program;

fn fib_state(n: i64) -> State {
    State::from_bindings([
        ("n".to_owned(), Value::Int(n)),
        ("i".to_owned(), Value::Int(0)),
        ("a".to_owned(), Value::Int(0)),
        ("b".to_owned(), Value::Int(1)),
    ])
}

fn janus_sort_state() -> State {
    State::from_bindings([
        ("n".to_owned(), Value::Int(50)),
        ("i".to_owned(), Value::Int(0)),
        ("j".to_owned(), Value::Int(0)),
        (
            "list".to_owned(),
            Value::Array((1..=50).rev().map(Value::Int).collect()),
        ),
        (
            "perm".to_owned(),
            Value::Array((0..50).map(|_| Value::Int(0)).collect()),
        ),
    ])
}

fn janus_sort_final_state() -> State {
    State::from_bindings([
        ("n".to_owned(), Value::Int(50)),
        ("i".to_owned(), Value::Int(0)),
        ("j".to_owned(), Value::Int(0)),
        (
            "list".to_owned(),
            Value::Array((1..=50).map(Value::Int).collect()),
        ),
        (
            "perm".to_owned(),
            Value::Array((0..50).rev().map(Value::Int).collect()),
        ),
    ])
}

fn rle_state() -> State {
    State::from_bindings([
        ("n".to_owned(), Value::Int(8)),
        ("run".to_owned(), Value::Int(0)),
        (
            "data".to_owned(),
            Value::Array(
                [2, 2, 2, 5, 5, 1, 1, 1]
                    .into_iter()
                    .map(Value::Int)
                    .collect(),
            ),
        ),
        (
            "symbols".to_owned(),
            Value::Array((0..8).map(|_| Value::Int(0)).collect()),
        ),
        (
            "counts".to_owned(),
            Value::Array((0..8).map(|_| Value::Int(0)).collect()),
        ),
    ])
}

fn rle_final_state() -> State {
    State::from_bindings([
        ("n".to_owned(), Value::Int(8)),
        ("run".to_owned(), Value::Int(3)),
        (
            "data".to_owned(),
            Value::Array(
                [2, 2, 2, 5, 5, 1, 1, 1]
                    .into_iter()
                    .map(Value::Int)
                    .collect(),
            ),
        ),
        (
            "symbols".to_owned(),
            Value::Array(
                [2, 5, 1, 0, 0, 0, 0, 0]
                    .into_iter()
                    .map(Value::Int)
                    .collect(),
            ),
        ),
        (
            "counts".to_owned(),
            Value::Array(
                [3, 2, 3, 0, 0, 0, 0, 0]
                    .into_iter()
                    .map(Value::Int)
                    .collect(),
            ),
        ),
    ])
}

fn procedure_call_state(n: i64) -> State {
    State::from_bindings([
        ("n".to_owned(), Value::Int(n)),
        ("i".to_owned(), Value::Int(0)),
        ("x".to_owned(), Value::Int(0)),
        ("y".to_owned(), Value::Int(1)),
    ])
}

fn constant_element_call_state(n: i64) -> State {
    State::from_bindings([
        ("n".to_owned(), Value::Int(n)),
        ("i".to_owned(), Value::Int(0)),
        (
            "xs".to_owned(),
            Value::Array(vec![Value::Int(0), Value::Int(1)]),
        ),
    ])
}

fn matrix_state() -> State {
    State::from_bindings([
        (
            "a".to_owned(),
            Value::Array(vec![
                Value::Array(vec![Value::Int(2), Value::Int(4), Value::Int(4)]),
                Value::Array(vec![Value::Int(4), Value::Int(1), Value::Int(1)]),
                Value::Array(vec![Value::Int(2), Value::Int(3), Value::Int(4)]),
            ]),
        ),
        (
            "b".to_owned(),
            Value::Array(vec![
                Value::Array(vec![Value::Int(24), Value::Int(-18), Value::Int(32)]),
                Value::Array(vec![Value::Int(-12), Value::Int(-19), Value::Int(-9)]),
                Value::Array(vec![Value::Int(11), Value::Int(9), Value::Int(10)]),
            ]),
        ),
        (
            "out".to_owned(),
            Value::Array(
                (0..3)
                    .map(|_| Value::Array((0..3).map(|_| Value::Int(0)).collect()))
                    .collect(),
            ),
        ),
    ])
}

fn matrix_transpose_state() -> State {
    State::from_bindings([
        ("n".to_owned(), Value::Int(3)),
        (
            "m".to_owned(),
            Value::Array(vec![
                Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
                Value::Array(vec![Value::Int(4), Value::Int(5), Value::Int(6)]),
                Value::Array(vec![Value::Int(7), Value::Int(8), Value::Int(9)]),
            ]),
        ),
    ])
}

fn matrix_transpose_final_state() -> State {
    State::from_bindings([
        ("n".to_owned(), Value::Int(3)),
        (
            "m".to_owned(),
            Value::Array(vec![
                Value::Array(vec![Value::Int(1), Value::Int(4), Value::Int(7)]),
                Value::Array(vec![Value::Int(2), Value::Int(5), Value::Int(8)]),
                Value::Array(vec![Value::Int(3), Value::Int(6), Value::Int(9)]),
            ]),
        ),
    ])
}

fn fixed_point_wave_state() -> State {
    State::from_bindings([
        ("n".to_owned(), Value::Int(16)),
        ("steps".to_owned(), Value::Int(100)),
        ("coeff".to_owned(), Value::Int(536_870_912)),
        (
            "psi_r".to_owned(),
            Value::Array(
                (0..16)
                    .map(|index| Value::Int((index as i64 + 1) * 67_108_864))
                    .collect(),
            ),
        ),
        (
            "psi_i".to_owned(),
            Value::Array(
                (0..16)
                    .map(|index| Value::Int((16 - index as i64) * 33_554_432))
                    .collect(),
            ),
        ),
    ])
}

fn procedure_call_program() -> &'static str {
    r#"
proc bump(x, y) {
  x += y
}

if n != 0 then
  from i == 0 do
    call bump(x, y);
    uncall bump(x, y);
    i += 1
  loop
    skip
  until i == n
else
  skip
fi n != 0
"#
}

fn constant_element_call_program() -> &'static str {
    r#"
proc bump(x, y) {
  x += y
}

if n != 0 then
  from i == 0 do
    call bump(xs[0], xs[1]);
    uncall bump(xs[0], xs[1]);
    i += 1
  loop
    skip
  until i == n
else
  skip
fi n != 0
"#
}

fn fixed_point_wave_program() -> &'static str {
    r#"
iterate int t = 0 to steps - 1
  iterate int i = 0 to n - 1
    psi_r[i] += psi_i[i] */ coeff;
    psi_i[i] -= psi_r[i] */ coeff
  end
end
"#
}

fn matrix_accumulate_program() -> &'static str {
    r#"
proc add_product(cell, left, right) {
  cell += left * right
}

iterate int i = 0 to 2
  iterate int j = 0 to 2
    iterate int k = 0 to 2
      call add_product(out[i][j], a[i][k], b[k][j])
    end
  end
end
"#
}

fn fibonacci_pair_transform(c: &mut Criterion) {
    let program = parse_program(include_str!("../../../examples/fib.rev")).expect("fib parses");
    let initial = fib_state(1_000);
    let compiled = CompiledProgram::for_state(&program, &initial).expect("fib compiles");

    let mut group = c.benchmark_group("fib_pair_transform_n1000");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute(black_box(&program), black_box(initial.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn janus_style_sort(c: &mut Criterion) {
    let program =
        parse_program(include_str!("../../../examples/janus_sort.rev")).expect("sort parses");
    let initial = janus_sort_state();
    let compiled = CompiledProgram::for_state(&program, &initial).expect("sort compiles");

    let mut group = c.benchmark_group("janus_style_sort_n50_reverse_order");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute(black_box(&program), black_box(initial.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn janus_style_sort_reverse(c: &mut Criterion) {
    let program =
        parse_program(include_str!("../../../examples/janus_sort.rev")).expect("sort parses");
    let inverse = invert_program(&program);
    let final_state = janus_sort_final_state();
    let compiled = CompiledProgram::for_state(&inverse, &final_state).expect("sort compiles");

    let mut group = c.benchmark_group("janus_style_sort_n50_reverse_order_reverse");
    group.bench_function("tree_walk", |b| {
        b.iter(|| {
            execute(black_box(&inverse), black_box(final_state.clone())).expect("tree runs");
        });
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(final_state.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn rle_compression(c: &mut Criterion) {
    let program =
        parse_program(include_str!("../../../examples/rle_compression.rev")).expect("rle parses");
    let initial = rle_state();
    let compiled = CompiledProgram::for_state(&program, &initial).expect("rle compiles");

    let mut group = c.benchmark_group("rle_compression_n8");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute(black_box(&program), black_box(initial.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn rle_compression_reverse(c: &mut Criterion) {
    let program =
        parse_program(include_str!("../../../examples/rle_compression.rev")).expect("rle parses");
    let inverse = invert_program(&program);
    let final_state = rle_final_state();
    let compiled = CompiledProgram::for_state(&inverse, &final_state).expect("rle compiles");

    let mut group = c.benchmark_group("rle_compression_n8_reverse");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute(black_box(&inverse), black_box(final_state.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(final_state.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn rle_compression_roundtrip(c: &mut Criterion) {
    let program =
        parse_program(include_str!("../../../examples/rle_compression.rev")).expect("rle parses");
    let inverse = invert_program(&program);
    let initial = rle_state();
    let final_state = rle_final_state();
    let compiled_forward = CompiledProgram::for_state(&program, &initial).expect("rle compiles");
    let compiled_reverse =
        CompiledProgram::for_state(&inverse, &final_state).expect("rle inverse compiles");

    let mut group = c.benchmark_group("rle_compression_n8_roundtrip");
    group.bench_function("tree_walk", |b| {
        b.iter(|| {
            let forward =
                execute(black_box(&program), black_box(initial.clone())).expect("tree runs");
            execute(black_box(&inverse), black_box(forward)).expect("tree inverse runs");
        });
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            let forward = compiled_forward
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
            compiled_reverse
                .execute(black_box(forward))
                .expect("compiled inverse runs");
        });
    });
    group.finish();
}

fn fixed_point_wave_n16_t100(c: &mut Criterion) {
    let program = parse_program(fixed_point_wave_program()).expect("fixed-point wave parses");
    let initial = fixed_point_wave_state();
    let compiled = CompiledProgram::for_state(&program, &initial).expect("wave compiles");

    let mut group = c.benchmark_group("fixed_point_wave_n16_t100");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute(black_box(&program), black_box(initial.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn procedure_call_uncall(c: &mut Criterion) {
    let program = parse_program(procedure_call_program()).expect("procedure benchmark parses");
    let initial = procedure_call_state(1_000);
    let compiled = CompiledProgram::for_state(&program, &initial).expect("procedure compiles");

    let mut group = c.benchmark_group("procedure_call_uncall_n1000");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute(black_box(&program), black_box(initial.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn constant_element_call_uncall(c: &mut Criterion) {
    let program =
        parse_program(constant_element_call_program()).expect("element procedure benchmark parses");
    let initial = constant_element_call_state(1_000);
    let compiled =
        CompiledProgram::for_state(&program, &initial).expect("element procedure compiles");

    let mut group = c.benchmark_group("constant_element_call_uncall_n1000");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute(black_box(&program), black_box(initial.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn janus_turing_binary_inc(c: &mut Criterion) {
    let program =
        parse_program(include_str!("../../../examples/janus_turing.rev")).expect("turing parses");
    let initial = State::empty();
    let compiled = CompiledProgram::for_state(&program, &initial).expect("turing compiles");

    let mut group = c.benchmark_group("janus_turing_binary_inc");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute(black_box(&program), black_box(initial.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn janus_stack_reverse(c: &mut Criterion) {
    let program = parse_program(include_str!("../../../examples/janus_stack_reverse.rev"))
        .expect("stack reverse parses");
    let initial = State::empty();
    let compiled = CompiledProgram::for_state(&program, &initial).expect("stack reverse compiles");

    let mut group = c.benchmark_group("janus_stack_reverse_n5");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute(black_box(&program), black_box(initial.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn matrix_accumulate_3x3(c: &mut Criterion) {
    let program = parse_program(matrix_accumulate_program()).expect("matrix benchmark parses");
    let initial = matrix_state();
    let compiled = CompiledProgram::for_state(&program, &initial).expect("matrix compiles");

    let mut group = c.benchmark_group("matrix_accumulate_3x3");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute(black_box(&program), black_box(initial.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn matrix_transpose_3x3(c: &mut Criterion) {
    let program = parse_program(include_str!("../../../examples/matrix_transpose.rev"))
        .expect("matrix transpose parses");
    let initial = matrix_transpose_state();
    let compiled =
        CompiledProgram::for_state(&program, &initial).expect("matrix transpose compiles");

    let mut group = c.benchmark_group("matrix_transpose_3x3");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute(black_box(&program), black_box(initial.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn matrix_transpose_3x3_reverse(c: &mut Criterion) {
    let program = parse_program(include_str!("../../../examples/matrix_transpose.rev"))
        .expect("matrix transpose parses");
    let inverse = invert_program(&program);
    let final_state = matrix_transpose_final_state();
    let compiled =
        CompiledProgram::for_state(&inverse, &final_state).expect("matrix transpose compiles");

    let mut group = c.benchmark_group("matrix_transpose_3x3_reverse");
    group.bench_function("tree_walk", |b| {
        b.iter(|| {
            execute(black_box(&inverse), black_box(final_state.clone())).expect("tree runs");
        });
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(final_state.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn tape_io_read_write(c: &mut Criterion) {
    let program = parse_program(include_str!("../../../examples/io.rev")).expect("io parses");
    let initial = IoState::new(State::empty(), [Value::Int(7)]);
    let compiled = CompiledProgram::for_state(&program, initial.store()).expect("io compiles");

    let mut group = c.benchmark_group("tape_io_read_write");
    group.bench_function("tree_walk", |b| {
        b.iter(|| execute_io(black_box(&program), black_box(initial.clone())).expect("tree runs"));
    });
    group.bench_function("slot_compiled", |b| {
        b.iter(|| {
            compiled
                .execute_io(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn slot_compile_vs_execute(c: &mut Criterion) {
    let program =
        parse_program(include_str!("../../../examples/janus_sort.rev")).expect("sort parses");
    let initial = janus_sort_state();
    let compiled = CompiledProgram::for_state(&program, &initial).expect("sort compiles");

    let mut group = c.benchmark_group("slot_compile_vs_execute_sort_n50");
    group.bench_function("compile", |b| {
        b.iter(|| {
            CompiledProgram::for_state(black_box(&program), black_box(&initial))
                .expect("sort compiles");
        });
    });
    group.bench_function("execute_compiled", |b| {
        b.iter(|| {
            compiled
                .execute(black_box(initial.clone()))
                .expect("compiled runs");
        });
    });
    group.finish();
}

fn scrub_timeline_construction(c: &mut Criterion) {
    let program =
        parse_program(include_str!("../../../examples/janus_sort.rev")).expect("sort parses");
    let initial = janus_sort_state();

    let mut group = c.benchmark_group("scrub_timeline_sort_n50");
    group.bench_function("build_timeline", |b| {
        b.iter(|| {
            build_timeline(black_box(&program), black_box(initial.clone()))
                .expect("timeline builds");
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    fibonacci_pair_transform,
    janus_style_sort,
    janus_style_sort_reverse,
    rle_compression,
    rle_compression_reverse,
    rle_compression_roundtrip,
    fixed_point_wave_n16_t100,
    procedure_call_uncall,
    constant_element_call_uncall,
    janus_turing_binary_inc,
    janus_stack_reverse,
    matrix_accumulate_3x3,
    matrix_transpose_3x3,
    matrix_transpose_3x3_reverse,
    tape_io_read_write,
    slot_compile_vs_execute,
    scrub_timeline_construction
);
criterion_main!(benches);

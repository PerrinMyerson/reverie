//! Property tests over randomly generated, well-formed Reverie *source text*.
//!
//! Unlike the straight-line generators inside the crate, these programs use
//! structured control flow (`if`/`fi`, `from`/`until` loops wrapped in
//! `local`/`delocal`), updates whose right-hand sides read other variables,
//! swaps, and odd-constant multiplicative updates. Every generated program goes
//! through the real parser and checker, then must:
//!
//! - run forward and backward back to its entry store, in both engines;
//! - agree between the tree-walking and slot-compiled engines;
//! - survive `format_program` -> `parse_program` unchanged;
//! - satisfy `invert(invert(p)) == p` up to formatting (and up to constant
//!   folding of `*=` factors).

use proptest::prelude::*;
use proptest::test_runner::Config;
use reverie_interp::{State, Value};
use reverie_syntax::{format_program, parse_program};

const VARS: [&str; 3] = ["x", "y", "z"];

/// An expression over every variable except `target`, built from literals and
/// wrapping `+`, `-`, `*`, `&`, `^`, `|`.
fn expr_avoiding(target: &'static str, depth: u32) -> BoxedStrategy<String> {
    let readable: Vec<&'static str> = VARS
        .iter()
        .copied()
        .filter(|name| *name != target)
        .chain(["g"])
        .collect();
    let leaf = prop_oneof![
        (-50i64..=50).prop_map(|value| {
            if value < 0 {
                format!("(0 - {})", -value)
            } else {
                value.to_string()
            }
        }),
        prop::sample::select(readable).prop_map(str::to_owned),
    ]
    .boxed();
    if depth == 0 {
        return leaf;
    }
    prop_oneof![
        2 => leaf,
        1 => (
            expr_avoiding(target, depth - 1),
            prop::sample::select(vec!["+", "-", "*", "&", "^", "|"]),
            expr_avoiding(target, depth - 1),
        )
            .prop_map(|(left, op, right)| format!("({left} {op} {right})")),
    ]
    .boxed()
}

fn update() -> BoxedStrategy<String> {
    prop::sample::select(VARS.to_vec())
        .prop_flat_map(|target| {
            prop_oneof![
                3 => (
                    prop::sample::select(vec!["+=", "-=", "^="]),
                    expr_avoiding(target, 2),
                )
                    .prop_map(move |(op, expr)| format!("{target} {op} {expr}")),
                1 => (-40i64..=40).prop_map(move |half| {
                    let odd = 2 * half + 1;
                    if odd < 0 {
                        format!("{target} *= (0 - {})", -odd)
                    } else {
                        format!("{target} *= {odd}")
                    }
                }),
            ]
        })
        .boxed()
}

fn swap() -> BoxedStrategy<String> {
    prop::sample::select(vec![("x", "y"), ("x", "z"), ("y", "z")])
        .prop_map(|(left, right)| format!("{left} <=> {right}"))
        .boxed()
}

fn block(depth: u32) -> BoxedStrategy<String> {
    prop::collection::vec(statement(depth), 1..5)
        .prop_map(|statements| statements.join(";\n"))
        .boxed()
}

/// A statement; `depth` bounds nesting and names loop counters so nested
/// `local` counters never shadow each other.
fn statement(depth: u32) -> BoxedStrategy<String> {
    let simple = prop_oneof![3 => update(), 1 => swap(), 1 => Just("skip".to_owned())].boxed();
    if depth == 0 {
        return simple;
    }
    let counter = format!("k{depth}");
    let conditional = (
        -20i64..=20,
        prop::sample::select(vec![">", "<", "==", "!="]),
        block(depth - 1),
        block(depth - 1),
    )
        .prop_map(|(bound, cmp, then_branch, else_branch)| {
            let bound = if bound < 0 {
                format!("(0 - {})", -bound)
            } else {
                bound.to_string()
            };
            // `g` is never written, so the exit assertion equals the entry test.
            format!(
                "if g {cmp} {bound} then\n{then_branch}\nelse\n{else_branch}\nfi g {cmp} {bound}"
            )
        });
    let looped = (1i64..=3, block(depth - 1)).prop_map(move |(times, body)| {
        format!(
            "local {counter}: int = 0\n\
             from {counter} == 0 do\n{body};\n{counter} += 1\n\
             loop\nskip\nuntil {counter} == {times}\n\
             delocal {counter} = {times}"
        )
    });
    prop_oneof![4 => simple, 1 => conditional, 1 => looped].boxed()
}

fn program() -> impl Strategy<Value = String> {
    block(2)
}

fn entry_state() -> impl Strategy<Value = State> {
    (any::<i64>(), any::<i64>(), any::<i64>(), -25i64..=25).prop_map(|(x, y, z, g)| {
        State::from_bindings([
            ("x".to_owned(), Value::Int(x)),
            ("y".to_owned(), Value::Int(y)),
            ("z".to_owned(), Value::Int(z)),
            ("g".to_owned(), Value::Int(g)),
        ])
    })
}

proptest! {
    #![proptest_config(Config { cases: 512, ..Config::default() })]

    #[test]
    fn generated_structured_programs_round_trip(source in program(), state in entry_state()) {
        let program = parse_program(&source)
            .unwrap_or_else(|errors| panic!("generated source parses: {errors:?}\n{source}"));
        reverie_core::check_program(&program)
            .unwrap_or_else(|error| panic!("generated source checks: {error}\n{source}"));

        let tree = reverie_interp::execute(&program, state.clone())
            .unwrap_or_else(|error| panic!("tree forward: {error}\n{source}"));
        let slot = reverie_interp::execute_compiled(&program, state.clone())
            .unwrap_or_else(|error| panic!("slot forward: {error}\n{source}"));
        prop_assert_eq!(&tree, &slot, "engines disagree on\n{}", source);

        let back_tree = reverie_interp::execute_backward(&program, tree.clone())
            .unwrap_or_else(|error| panic!("tree backward: {error}\n{source}"));
        let back_slot = reverie_interp::execute_compiled_backward(&program, slot)
            .unwrap_or_else(|error| panic!("slot backward: {error}\n{source}"));
        prop_assert_eq!(&back_tree, &state, "tree roundtrip failed for\n{}", source);
        prop_assert_eq!(&back_slot, &state, "slot roundtrip failed for\n{}", source);

        // Running the mechanically inverted program forward is the same as
        // running the original backward.
        let inverse = reverie_core::invert_program(&program);
        reverie_core::check_program(&inverse)
            .unwrap_or_else(|error| panic!("inverse checks: {error}\n{source}"));
        let via_inverse = reverie_interp::execute(&inverse, tree)
            .unwrap_or_else(|error| panic!("inverse forward: {error}\n{source}"));
        prop_assert_eq!(&via_inverse, &state);
    }

    #[test]
    fn generated_programs_format_and_invert_stably(source in program()) {
        let program = parse_program(&source).expect("generated source parses");
        let formatted = format_program(&program);
        let reparsed = parse_program(&formatted)
            .unwrap_or_else(|errors| panic!("formatted source reparses: {errors:?}\n{formatted}"));
        prop_assert_eq!(format_program(&reparsed), formatted.clone());

        // `invert` folds a `*=` factor into a literal, so with `*=` present the
        // double inverse is equal in meaning but may differ in spelling; from
        // then on inversion is an exact syntactic involution.
        let invert = reverie_core::invert_program;
        let twice = invert(&invert(&program));
        if !formatted.contains("*=") {
            prop_assert_eq!(format_program(&twice), formatted);
        }
        prop_assert_eq!(format_program(&invert(&invert(&twice))), format_program(&twice));
    }
}

#[test]
fn odd_inverse_is_a_true_inverse() {
    for value in [
        1i64,
        3,
        -1,
        -3,
        7,
        0x5555_5555_5555_5555,
        i64::MAX,
        i64::MIN + 1,
    ] {
        let inverse = reverie_core::odd_inverse_mod_2_64(value);
        assert_eq!(value.wrapping_mul(inverse), 1, "inverse of {value}");
    }
}

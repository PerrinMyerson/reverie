use std::path::{Path, PathBuf};

use assert_cmd::Command;
use predicates::prelude::*;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate is inside workspace/crates/reverie-cli")
        .to_path_buf()
}

fn strip_ansi_csi(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for code in chars.by_ref() {
                if code.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            output.push(ch);
        }
    }

    output
}

#[test]
fn check_skip_example_succeeds() {
    let example = workspace_root().join("examples/skip.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));
}

#[test]
fn check_rejects_literal_zero_divisor() {
    let root = workspace_root();
    let example = root.join("target/test-literal-zero-divisor.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "x += y / 0\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("division by constant zero"));
}

#[test]
fn check_rejects_constant_zero_divisor() {
    let root = workspace_root();
    let example = root.join("target/test-constant-zero-divisor.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "x += y / (1 - 1)\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("division by constant zero"));
}

#[test]
fn check_rejects_negative_literal_array_index() {
    let root = workspace_root();
    let example = root.join("target/test-negative-literal-index.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        "local xs: array<int> = [1, 2, 3]\nxs[-1] += 1\ndelocal xs = [1, 2, 3]\n",
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "array index must be non-negative, found -1",
        ));
}

#[test]
fn check_rejects_literal_array_index_out_of_bounds_when_shape_is_known() {
    let root = workspace_root();
    let example = root.join("target/test-literal-index-oob.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        "local xs: array<int> = [1, 2, 3]\nxs[3] += 1\ndelocal xs = [1, 2, 3]\n",
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "array `xs` index 3 out of bounds for length 3",
        ));
}

#[test]
fn check_rejects_constant_array_index_out_of_bounds_when_shape_is_known() {
    let root = workspace_root();
    let example = root.join("target/test-constant-index-oob.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        "local xs: array<int> = [1, 2, 3]\nxs[1 + 2] += 1\ndelocal xs = [1, 2, 3]\n",
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "array `xs` index 3 out of bounds for length 3",
        ));
}

#[test]
fn check_rejects_update_that_reads_same_constant_array_cell() {
    let root = workspace_root();
    let example = root.join("target/test-same-constant-cell-update.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "xs[1 + 1] += xs[2]\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("reads the value being changed"));
}

#[test]
fn check_rejects_update_that_reads_potentially_same_dynamic_array_cell() {
    let root = workspace_root();
    let example = root.join("target/test-dynamic-cell-update-alias.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "xs[0] += xs[i]\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("reads the value being changed"));
}

#[test]
fn check_rejects_constant_duplicate_element_call_arguments() {
    let root = workspace_root();
    let example = root.join("target/test-constant-duplicate-element-call.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        "proc add(x, y) { x += y }\ncall add(xs[1 + 1], xs[2])\n",
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("duplicate element arguments"));
}

#[test]
fn check_rejects_potentially_aliasing_dynamic_element_call_arguments() {
    let root = workspace_root();
    let example = root.join("target/test-dynamic-element-call-alias.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        "proc add(x, y) { x += y }\ncall add(xs[0], xs[i])\n",
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "potentially aliasing element arguments",
        ));
}

#[test]
fn check_rejects_procedure_literal_index_out_of_bounds_for_known_argument_shape() {
    let root = workspace_root();
    let example = root.join("target/test-procedure-literal-index-oob.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        "proc touch(xs: array<int>) { xs[3] += 1 }\n\
         local xs: array<int> = [1, 2, 3]\n\
         call touch(xs)\n\
         delocal xs = [1, 2, 3]\n",
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "array `xs` index 3 out of bounds for length 3",
        ));
}

#[test]
fn check_rejects_nested_procedure_literal_index_out_of_bounds_for_known_argument_shape() {
    let root = workspace_root();
    let example = root.join("target/test-nested-procedure-literal-index-oob.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        "proc inner(xs: array<int>) { xs[3] += 1 }\n\
         proc outer(xs: array<int>) { call inner(xs) }\n\
         local xs: array<int> = [1, 2, 3]\n\
         call outer(xs)\n\
         delocal xs = [1, 2, 3]\n",
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "array `xs` index 3 out of bounds for length 3",
        ));
}

#[test]
fn run_skip_example_prints_empty_store() {
    let example = workspace_root().join("examples/skip.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{}\n");
}

#[test]
fn run_skip_example_accepts_seeded_stack_vars() {
    let example = workspace_root().join("examples/skip.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "s=stack[3,2,1]", "--type", "s=stack"])
        .assert()
        .success()
        .stdout("{s = stack[3, 2, 1]}\n");
}

#[test]
fn duplicate_seed_vars_are_rejected() {
    let example = workspace_root().join("examples/skip.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "x=1", "--var", "x=2"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "duplicate --var annotation for `x`",
        ));
}

#[test]
fn duplicate_seed_types_are_rejected_after_legacy_normalization() {
    let example = workspace_root().join("examples/skip.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(&example)
        .arg("--legacy-janus")
        .args(["--type", "VALUE=int", "--type", "value=bool"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "duplicate --type annotation for `value`",
        ));
}

#[test]
fn unexpected_seed_type_is_rejected_when_program_shape_is_known() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(&example)
        .args(["--type", "nn=bool"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unexpected --type annotation(s) for `nn`",
        ))
        .stderr(predicate::str::contains(format!(
            "reverie explain '{}'",
            example.display()
        )));
}

#[test]
fn missing_external_seed_is_rejected_before_runtime() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "n=7", "--var", "i=0", "--var", "a=0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("missing --var seed(s) for `b`"))
        .stderr(predicate::str::contains(format!(
            "reverie explain '{}'",
            example.display()
        )));
}

#[test]
fn unexpected_external_seed_is_rejected_when_program_shape_is_known() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args([
            "--var", "n=7", "--var", "i=0", "--var", "a=0", "--var", "b=1", "--var", "nn=7",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unexpected --var seed(s) for `nn`",
        ))
        .stderr(predicate::str::contains(format!(
            "reverie explain '{}'",
            example.display()
        )));
}

#[test]
fn unexpected_seed_for_global_only_program_is_rejected() {
    let root = workspace_root();
    let example = root.join("target/test-global-unexpected-seed.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "global x;\nskip\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "x=1", "--var", "typo=1"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unexpected --var seed(s) for `typo`",
        ));
}

#[test]
fn declared_store_seed_is_allowed_alongside_external_seed() {
    let root = workspace_root();
    let example = root.join("target/test-declared-and-external-seeds.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "int x\nx += y\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "x=0", "--var", "y=1"])
        .assert()
        .success()
        .stdout("{x = 1, y = 1}\n");
}

#[test]
fn every_example_is_documented_and_exercised() {
    let root = workspace_root();
    let examples_dir = root.join("examples");
    let examples_guide = std::fs::read_to_string(examples_dir.join("README.md"))
        .expect("examples guide is readable");
    let cli_tests = include_str!("cli.rs");

    for entry in std::fs::read_dir(examples_dir).expect("examples directory is readable") {
        let entry = entry.expect("example entry is readable");
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rev") {
            continue;
        }

        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("example filename is utf-8");
        let mention = format!("examples/{name}");

        assert!(
            examples_guide.contains(&mention),
            "{mention} should be documented in examples/README.md"
        );
        assert!(
            cli_tests.contains(&mention),
            "{mention} should be covered by CLI tests"
        );
    }
}

#[test]
fn run_fib_example_accepts_seeded_integer_vars() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args([
            "--var", "n=7", "--var", "i=0", "--var", "a=0", "--var", "b=1",
        ])
        .assert()
        .success()
        .stdout("{a = 13, b = 21, i = 7, n = 7}\n");
}

#[test]
fn run_fib_example_can_force_tree_engine() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .arg("--engine")
        .arg("tree")
        .args([
            "--var", "n=7", "--var", "i=0", "--var", "a=0", "--var", "b=1",
        ])
        .assert()
        .success()
        .stdout("{a = 13, b = 21, i = 7, n = 7}\n");
}

#[test]
fn reverse_fib_example_accepts_seeded_integer_vars() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var", "n=7", "--var", "i=7", "--var", "a=13", "--var", "b=21",
        ])
        .assert()
        .success()
        .stdout("{a = 0, b = 1, i = 0, n = 7}\n");
}

#[test]
fn reverse_accepts_seeded_source_declarations() {
    let root = workspace_root();
    let example = root.join("target/test-source-declaration-reverse.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "int x\nx += 1\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "x=1"])
        .assert()
        .success()
        .stdout("{x = 0}\n");
}

#[test]
fn increment_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/increment.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{x = 1, xs = [-1, 1]}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "x=1", "--var", "xs=[-1,1]"])
        .assert()
        .success()
        .stdout("{x = 0, xs = [0, 0]}\n");
}

#[test]
fn invert_increment_example_prints_canonical_inverse_updates() {
    let example = workspace_root().join("examples/increment.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("invert")
        .arg(example)
        .assert()
        .success()
        .stdout("global x;\nglobal xs[2];\n\nxs[1] -= 1;\nxs[0] += 1;\nx -= 1\n");
}

#[test]
fn invert_swap_call_alias_prints_canonical_swap() {
    let root = workspace_root();
    let example = root.join("target/test-invert-swap-alias.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "swap(x, y)\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("invert")
        .arg(example)
        .assert()
        .success()
        .stdout("x <=> y\n");
}

#[test]
fn global_seed_values_must_match_declared_type() {
    let root = workspace_root();
    let example = root.join("target/test-global-seed-type.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "global x;\nskip\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "x=true"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("type mismatch"));
}

#[test]
fn global_array_seed_values_must_match_declared_shape() {
    let root = workspace_root();
    let example = root.join("target/test-global-array-shape.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "global xs[3];\nskip\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "xs=[1,2]"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("expected array length 3"));
}

#[test]
fn empty_array_seed_cannot_bypass_global_type() {
    let root = workspace_root();
    let example = root.join("target/test-global-empty-array-seed.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "global x;\nskip\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "x=[]"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("expected int"));
}

#[test]
fn empty_array_seed_cannot_bypass_source_declaration_type() {
    let root = workspace_root();
    let example = root.join("target/test-declare-int-empty-seed.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "int x\nskip\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "x=[]"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("expected int"));
}

#[test]
fn explain_fib_example_summarizes_reversible_shape() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "status: reversible program checks",
        ))
        .stdout(predicate::str::contains("features:"))
        .stdout(predicate::str::contains("- reversible loops"))
        .stdout(predicate::str::contains("- reversible updates"))
        .stdout(predicate::str::contains("safety checks:"))
        .stdout(predicate::str::contains("external store:"))
        .stdout(predicate::str::contains("- n: inferred at runtime"))
        .stdout(predicate::str::contains("declared store:"))
        .stdout(predicate::str::contains("- none"))
        .stdout(predicate::str::contains("run template: reverie run"))
        .stdout(predicate::str::contains("--var n=0"))
        .stdout(predicate::str::contains("inverse: reverie invert"));
}

#[test]
fn explain_lists_declared_store_overrides() {
    let example = workspace_root().join("examples/globals.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("external store:\n- none"))
        .stdout(predicate::str::contains("declared store:"))
        .stdout(predicate::str::contains(
            "- flag: optional --var override; default template false",
        ))
        .stdout(predicate::str::contains(
            "- x: optional --var override; default template 0",
        ))
        .stdout(predicate::str::contains(
            "- xs: optional --var override; default template [0,0,0]",
        ))
        .stdout(predicate::str::contains(
            "declared override template: reverie run",
        ))
        .stdout(predicate::str::contains("--var flag=false"))
        .stdout(predicate::str::contains("--var x=0"))
        .stdout(predicate::str::contains("--var xs='[0,0,0]'"));
}

#[test]
fn explain_json_emits_machine_readable_summary() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg("--json")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "\"status\": \"reversible program checks\"",
        ))
        .stdout(predicate::str::contains("\"features\": ["))
        .stdout(predicate::str::contains("\"reversible loops\""))
        .stdout(predicate::str::contains("\"safety_checks\": ["))
        .stdout(predicate::str::contains("\"safety_check_counts\": {"))
        .stdout(predicate::str::contains("\"external_store\": ["))
        .stdout(predicate::str::contains("\"name\":\"n\""))
        .stdout(predicate::str::contains("\"run_template\": \"reverie run"));
}

#[test]
fn explain_reports_indexed_safety_checks() {
    let example = workspace_root().join("examples/array.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("safety checks:"))
        .stdout(predicate::str::contains(
            "- constant array indexes checked before runtime",
        ))
        .stdout(predicate::str::contains(
            "- same-root update aliases rejected before runtime",
        ));
}

#[test]
fn explain_reports_disjoint_same_root_update_reads() {
    let root = workspace_root();
    let example = root.join("target/test-explain-disjoint-array-read.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "xs[0] += xs[1]\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "- same-root update aliases rejected before runtime",
        ))
        .stdout(predicate::str::contains(
            "- same-root update reads proven disjoint before runtime",
        ));
}

#[test]
fn explain_reports_dynamic_element_argument_location_resolution() {
    let root = workspace_root();
    let example = root.join("target/test-explain-dynamic-element-call.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        "proc add(x, y) { x += y }\ncall add(xs[i], ys[j])\n",
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "- same-root element argument aliases rejected before runtime",
        ))
        .stdout(predicate::str::contains(
            "- dynamic element argument locations resolved at call entry",
        ));
}

#[test]
fn explain_json_preserves_declared_store_templates() {
    let example = workspace_root().join("examples/globals.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg("--json")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("\"declared_store\": ["))
        .stdout(predicate::str::contains("\"name\":\"flag\""))
        .stdout(predicate::str::contains("\"template\":\"false\""))
        .stdout(predicate::str::contains("\"name\":\"xs\""))
        .stdout(predicate::str::contains("\"template\":\"[0,0,0]\""));
}

#[test]
fn explain_templates_nested_array_seeds_with_single_shell_quote() {
    let example = workspace_root().join("examples/matrix_transpose.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg(example)
        .args(["--type", "m=array<array<int>>", "--type", "n=int"])
        .assert()
        .success()
        .stdout(predicate::str::contains("run template: reverie run"))
        .stdout(predicate::str::contains("--var m='[[0]]'"))
        .stdout(predicate::str::contains("--var n=0"))
        .stdout(predicate::str::contains("--type 'm=array<array<int>>'"))
        .stdout(predicate::str::contains("--type n=int"));
}

#[test]
fn explain_quotes_template_paths_with_spaces() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "run template: reverie run '{}'",
            example.display()
        )))
        .stdout(predicate::str::contains(format!(
            "inverse: reverie invert '{}'",
            example.display()
        )));
}

#[test]
fn explain_preserves_unit_type_annotations_in_run_template() {
    let example = workspace_root().join("examples/top_level_units.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg(example)
        .args([
            "--type",
            "speed=int<m/s>",
            "--type",
            "distance=int<m>",
            "--type",
            "time=int<s>",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("--var distance=0"))
        .stdout(predicate::str::contains("--var speed=0"))
        .stdout(predicate::str::contains("--var time=0"))
        .stdout(predicate::str::contains("--type 'distance=int<m>'"))
        .stdout(predicate::str::contains("--type 'speed=int<m/s>'"))
        .stdout(predicate::str::contains("--type 'time=int<s>'"));
}

#[test]
fn wrapping_example_runs_both_directions() {
    let example = workspace_root().join("examples/wrapping.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "x=9223372036854775807"])
        .assert()
        .success()
        .stdout("{x = -9223372036854775808}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "x=-9223372036854775808"])
        .assert()
        .success()
        .stdout("{x = 9223372036854775807}\n");
}

#[test]
fn unary_negation_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/negation.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .assert()
        .success()
        .stdout("{}\n");
}

#[test]
fn janus_operator_aliases_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/janus_operators.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .assert()
        .success()
        .stdout("{}\n");
}

#[test]
fn janus_optional_control_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/janus_optional_control.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .assert()
        .success()
        .stdout("{}\n");
}

#[test]
fn janus_procedure_syntax_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/janus_procedure_syntax.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{x = 0}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "x=0"])
        .assert()
        .success()
        .stdout("{x = 0}\n");
}

#[test]
fn legacy_janus_mode_accepts_case_insensitive_identifiers_and_semicolon_comments() {
    let root = workspace_root();
    let example = root.join("target/test-legacy-janus.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        r#"
VALUE
PrOcEdUrE Main
  IF VALUE = 0 THEN
    Value += 1 + 2 * 3; VALUE += 99
  FI value = 7
"#,
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .arg("--legacy-janus")
        .args(["--var", "VALUE=0"])
        .assert()
        .success()
        .stdout("{value = 7}\n");
}

#[test]
fn janus_show_accepts_multiple_names() {
    let root = workspace_root();
    let example = root.join("target/test-janus-show-multiple.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        r#"
procedure main()
  int x = 1
  int y = 2
  show(x, y)
"#,
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .assert()
        .success()
        .stdout("x = 1\ny = 2\n{x = 1, y = 2}\n");
}

#[test]
fn standalone_assert_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/assert.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "x=0", "--var", "target=2"])
        .assert()
        .success()
        .stdout("{target = 2, x = 0}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "x=0", "--var", "target=2"])
        .assert()
        .success()
        .stdout("{target = 2, x = 0}\n");
}

#[test]
fn standalone_assert_failure_example_reports_runtime_span() {
    let example = workspace_root().join("examples/assert_failure.rev");

    let assert = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "x=0"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr);
    let plain_stderr = strip_ansi_csi(&stderr);

    assert!(
        plain_stderr.contains("assertion expected true"),
        "{plain_stderr}"
    );
    assert!(plain_stderr.contains("assert_eq(x, 1)"), "{plain_stderr}");
    assert!(
        plain_stderr.contains("runtime failed here"),
        "{plain_stderr}"
    );
}

#[test]
fn injectivized_max_example_runs_both_branches_and_reverses() {
    let example = workspace_root().join("examples/injectivized_max.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "x=9", "--var", "y=4", "--var", "swapped=0"])
        .assert()
        .success()
        .stdout("{swapped = 0, x = 9, y = 4}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "x=4", "--var", "y=9", "--var", "swapped=0"])
        .assert()
        .success()
        .stdout("{swapped = 1, x = 9, y = 4}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(&example)
        .args(["--var", "x=9", "--var", "y=4", "--var", "swapped=0"])
        .assert()
        .success()
        .stdout("{swapped = 0, x = 9, y = 4}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "x=9", "--var", "y=4", "--var", "swapped=1"])
        .assert()
        .success()
        .stdout("{swapped = 0, x = 4, y = 9}\n");
}

#[test]
fn loop_assertion_failure_example_reports_runtime_span() {
    let example = workspace_root().join("examples/loop_assertion_failure.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "i=0", "--var", "n=2"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("loop re-entry assertion"))
        .stderr(predicate::str::contains("runtime failed here"));
}

#[test]
fn if_assertion_failure_example_reports_runtime_span() {
    let example = workspace_root().join("examples/if_assertion_failure.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "x=0", "--var", "y=0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("if exit assertion"))
        .stderr(predicate::str::contains("runtime failed here"));
}

#[test]
fn delocal_assertion_failure_example_reports_runtime_span() {
    let example = workspace_root().join("examples/delocal_assertion_failure.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("delocal `x` expected 2, found 1"))
        .stderr(predicate::str::contains("runtime failed here"));
}

#[test]
fn naive_max_without_witness_example_reports_runtime_span() {
    let example = workspace_root().join("examples/naive_max_no_witness.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "x=9", "--var", "y=4"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("if exit assertion"))
        .stderr(predicate::str::contains("runtime failed here"));
}

#[test]
fn invert_fib_example_prints_inverse_source() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("invert")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("from i == n do"))
        .stdout(predicate::str::contains("i -= 1;"))
        .stdout(predicate::str::contains("a <=> b;"))
        .stdout(predicate::str::contains("a -= b"))
        .stdout(predicate::str::contains("until i == 0"));
}

#[test]
fn fmt_prints_canonical_source() {
    let root = workspace_root();
    let example = root.join("target/test-fmt.rev");
    std::fs::create_dir_all(root.join("target")).expect("target directory can be created");
    std::fs::write(
        &example,
        "++x\n--y\nswap(z,w)\nassert_eq(x, 1)\nassert_ne(y, 2)",
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("fmt")
        .arg(example)
        .assert()
        .success()
        .stdout("x += 1;\ny -= 1;\nz <=> w;\nassert x == 1;\nassert y != 2\n");
}

#[test]
fn fmt_check_rejects_noncanonical_source() {
    let root = workspace_root();
    let example = root.join("target/test-fmt-check.rev");
    std::fs::create_dir_all(root.join("target")).expect("target directory can be created");
    std::fs::write(&example, "x+=1").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("fmt")
        .arg("--check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("is not canonical"))
        .stderr(predicate::str::contains("reverie fmt"));
}

#[test]
fn fmt_write_rewrites_source_file() {
    let root = workspace_root();
    let example = root.join("target/test-fmt-write.rev");
    std::fs::create_dir_all(root.join("target")).expect("target directory can be created");
    std::fs::write(&example, "x+=1\ny<=>z").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("fmt")
        .arg("--write")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));

    let formatted = std::fs::read_to_string(example).expect("formatted source is readable");
    assert_eq!(formatted, "x += 1;\ny <=> z\n");
}

#[test]
fn fmt_rejects_check_with_write() {
    let root = workspace_root();
    let example = root.join("target/test-fmt-check-write.rev");
    std::fs::create_dir_all(root.join("target")).expect("target directory can be created");
    std::fs::write(&example, "skip").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("fmt")
        .arg("--check")
        .arg("--write")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "cannot combine --check and --write",
        ));
}

#[test]
fn run_proc_example_handles_call_and_local_block() {
    let example = workspace_root().join("examples/proc.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "n=4"])
        .assert()
        .success()
        .stdout("{n = 5}\n");
}

#[test]
fn reverse_proc_example_handles_uncall_and_delocal() {
    let example = workspace_root().join("examples/proc.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "n=5"])
        .assert()
        .success()
        .stdout("{n = 4}\n");
}

#[test]
fn alias_rejection_example_fails_check() {
    let example = workspace_root().join("examples/alias_rejection.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("duplicate arguments"))
        .stderr(predicate::str::contains("semantic check failed here"));
}

#[test]
fn irreversible_update_example_fails_check() {
    let example = workspace_root().join("examples/irreversible_update.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("reads the value being changed"))
        .stderr(predicate::str::contains("semantic check failed here"));
}

#[test]
fn local_shadow_rejection_example_fails_check() {
    let example = workspace_root().join("examples/local_shadow_rejection.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "shadows an existing typed variable",
        ))
        .stderr(predicate::str::contains("semantic check failed here"));
}

#[test]
fn static_array_bounds_example_fails_check() {
    let example = workspace_root().join("examples/static_array_bounds.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "array `xs` index 3 out of bounds for length 3",
        ))
        .stderr(predicate::str::contains("semantic check failed here"));
}

#[test]
fn procedure_runtime_error_reports_call_context() {
    let example = workspace_root().join("examples/proc_runtime_error.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "n=1", "--var", "z=0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("division by zero"))
        .stderr(predicate::str::contains("runtime failed here"))
        .stderr(predicate::str::contains("while calling `boom` here"));
}

#[test]
fn run_array_example_accepts_seeded_array_vars() {
    let example = workspace_root().join("examples/array.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "xs=[10,20,30]", "--var", "delta=5"])
        .assert()
        .success()
        .stdout("{delta = 5, xs = [30, 25, 10]}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "xs=[30,25,10]", "--var", "delta=5"])
        .assert()
        .success()
        .stdout("{delta = 5, xs = [10, 20, 30]}\n");
}

#[test]
fn run_skip_example_accepts_seeded_nested_array_vars() {
    let example = workspace_root().join("examples/skip.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args([
            "--var",
            "matrix=[[1,2],[3,4]]",
            "--type",
            "matrix=array<array<int>>",
        ])
        .assert()
        .success()
        .stdout("{matrix = [[1, 2], [3, 4]]}\n");
}

#[test]
fn run_skip_example_accepts_typed_empty_array_seed() {
    let example = workspace_root().join("examples/skip.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "xs=[]", "--type", "xs=array<int>"])
        .assert()
        .success()
        .stdout("{xs = []}\n");
}

#[test]
fn heterogeneous_array_seed_is_rejected() {
    let example = workspace_root().join("examples/skip.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "xs=[1,true]"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "seed `xs` has a heterogeneous array value",
        ));
}

#[test]
fn ragged_array_seed_is_rejected() {
    let example = workspace_root().join("examples/skip.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "matrix=[[1,2],[3]]"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "seed `matrix` has a ragged array value",
        ));
}

#[test]
fn run_accepts_seeded_boolean_vars() {
    let root = workspace_root();
    let example = root.join("target/test-seeded-bool.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        r#"
if flag then
  x += 1
else
  skip
fi flag
"#,
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "flag=true", "--var", "x=0"])
        .assert()
        .success()
        .stdout("{flag = true, x = 1}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "flag=true", "--var", "x=1"])
        .assert()
        .success()
        .stdout("{flag = true, x = 0}\n");
}

#[test]
fn seed_value_types_are_inferred_for_static_checking() {
    let root = workspace_root();
    let example = root.join("target/test-seed-type-inference.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "x += 1").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "x=true"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("expected int expression"));
}

#[test]
fn bool_toggle_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/bool_toggle.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "flag=false", "--type", "flag=bool"])
        .assert()
        .success()
        .stdout("{flag = true}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "flag=true", "--type", "flag=bool"])
        .assert()
        .success()
        .stdout("{flag = false}\n");
}

#[test]
fn bool_flags_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/bool_flags.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args([
            "--var",
            "flags=[false,true,false]",
            "--type",
            "flags=array<bool>",
        ])
        .assert()
        .success()
        .stdout("{flags = [true, true, true]}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "flags=[true,true,true]",
            "--type",
            "flags=array<bool>",
        ])
        .assert()
        .success()
        .stdout("{flags = [false, true, false]}\n");
}

#[test]
fn run_skip_example_accepts_seeded_bool_arrays() {
    let example = workspace_root().join("examples/skip.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args([
            "--var",
            "flags=[true,false,true]",
            "--type",
            "flags=array<bool>",
        ])
        .assert()
        .success()
        .stdout("{flags = [true, false, true]}\n");
}

#[test]
fn bool_global_defaults_to_false() {
    let root = workspace_root();
    let example = root.join("target/test-bool-global.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        r#"
global flag: bool;
assert !flag
"#,
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .assert()
        .success()
        .stdout("{flag = false}\n");
}

#[test]
fn typed_global_arrays_can_be_indexed() {
    let root = workspace_root();
    let example = root.join("target/test-typed-global-array.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(
        &example,
        r#"
global bool flags[3] = [true, false, true];
flags[1] ^= true
"#,
    )
    .expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{flags = [true, true, true]}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "flags=[true,true,true]",
            "--type",
            "flags=array<bool>",
        ])
        .assert()
        .success()
        .stdout("{flags = [true, false, true]}\n");
}

#[test]
fn array_size_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/size.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .assert()
        .success()
        .stdout("{}\n");
}

#[test]
fn stack_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/stack.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .assert()
        .success()
        .stdout("{}\n");
}

#[test]
fn janus_stack_reverse_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/janus_stack_reverse.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("s = <5, 4, 3, 2, 1]\n{s = nil}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .assert()
        .success()
        .stdout("s = <5, 4, 3, 2, 1]\n{s = nil}\n");
}

#[test]
fn globals_example_zero_initializes_and_reverses() {
    let example = workspace_root().join("examples/globals.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{flag = true, x = 1, xs = [1, 2, 1]}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "flag=true", "--var", "x=1", "--var", "xs=[1,2,1]"])
        .assert()
        .success()
        .stdout("{flag = false, x = 0, xs = [0, 0, 0]}\n");
}

#[test]
fn tape_io_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/io.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--input", "7"])
        .assert()
        .success()
        .stdout("{x = 7}\noutput: [0, 0, 7]\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var", "x=7", "--output", "0", "--output", "0", "--output", "7",
        ])
        .assert()
        .success()
        .stdout("{x = 0}\ninput: [7]\n");
}

#[test]
fn heterogeneous_input_tape_array_is_rejected() {
    let example = workspace_root().join("examples/io.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--input", "[1,true]"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--input has a heterogeneous array value",
        ));
}

#[test]
fn ragged_input_tape_array_is_rejected() {
    let example = workspace_root().join("examples/io.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--input", "[[1,2],[3]]"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--input has a ragged array value"));
}

#[test]
fn heterogeneous_output_tape_array_is_rejected() {
    let example = workspace_root().join("examples/io.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "x=0", "--output", "[1,true]"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--output has a heterogeneous array value",
        ));
}

#[test]
fn ragged_output_tape_array_is_rejected() {
    let example = workspace_root().join("examples/io.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "x=0", "--output", "[[1,2],[3]]"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--output has a ragged array value",
        ));
}

#[test]
fn janus_sort_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/janus_sort.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args([
            "--var",
            "n=5",
            "--var",
            "i=0",
            "--var",
            "j=0",
            "--var",
            "list=[5,3,4,1,2]",
            "--var",
            "perm=[0,0,0,0,0]",
        ])
        .assert()
        .success()
        .stdout("{i = 0, j = 0, list = [1, 2, 3, 4, 5], n = 5, perm = [3, 4, 1, 2, 0]}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "n=5",
            "--var",
            "i=0",
            "--var",
            "j=0",
            "--var",
            "list=[1,2,3,4,5]",
            "--var",
            "perm=[3,4,1,2,0]",
        ])
        .assert()
        .success()
        .stdout("{i = 0, j = 0, list = [5, 3, 4, 1, 2], n = 5, perm = [0, 0, 0, 0, 0]}\n");
}

#[test]
fn bit_reversal_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/bit_reversal.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args([
            "--var",
            "n=8",
            "--var",
            "xs=[10,20,30,40,50,60,70,80]",
            "--var",
            "perm=[0,4,2,6,1,5,3,7]",
        ])
        .assert()
        .success()
        .stdout(
            "{n = 8, perm = [0, 4, 2, 6, 1, 5, 3, 7], xs = [10, 50, 30, 70, 20, 60, 40, 80]}\n",
        );

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "n=8",
            "--var",
            "xs=[10,50,30,70,20,60,40,80]",
            "--var",
            "perm=[0,4,2,6,1,5,3,7]",
        ])
        .assert()
        .success()
        .stdout(
            "{n = 8, perm = [0, 4, 2, 6, 1, 5, 3, 7], xs = [10, 20, 30, 40, 50, 60, 70, 80]}\n",
        );
}

#[test]
fn matrix_transpose_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/matrix_transpose.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "n=3", "--var", "m=[[1,2,3],[4,5,6],[7,8,9]]"])
        .assert()
        .success()
        .stdout("{m = [[1, 4, 7], [2, 5, 8], [3, 6, 9]], n = 3}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "n=3", "--var", "m=[[1,4,7],[2,5,8],[3,6,9]]"])
        .assert()
        .success()
        .stdout("{m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], n = 3}\n");
}

#[test]
fn perm_to_code_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/perm_to_code.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "x=[2,0,3,1,5,4]"])
        .assert()
        .success()
        .stdout("{x = [0, 0, 2, 1, 4, 4]}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "x=[0,0,2,1,4,4]"])
        .assert()
        .success()
        .stdout("{x = [2, 0, 3, 1, 5, 4]}\n");
}

#[test]
fn janus_automata_example_runs_forward() {
    let example = workspace_root().join("examples/janus_automata.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("cell = [0, 100, 100]"))
        .stdout(predicate::str::contains("steps = 1"))
        .stdout(predicate::str::contains(
            "output: [100, 100, 0, 0, 100, 100]",
        ));
}

#[test]
fn janus_turing_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/janus_turing.rev");

    let final_vars = [
        ("PC_MAX", "8"),
        ("QF", "6"),
        ("QS", "1"),
        ("blank", "200"),
        ("left", "100"),
        ("leftsp", "0"),
        ("leftstk", "[0,0,0,0]"),
        ("pc", "0"),
        ("q", "6"),
        ("q1", "[1,2,3,3,3,4,5,5]"),
        ("q2", "[2,3,4,2,4,5,4,6]"),
        ("right", "102"),
        ("rightsp", "0"),
        ("rightstk", "[0,0,0,0]"),
        ("s", "200"),
        ("s1", "[200,101,0,1,200,101,0,200]"),
        ("s2", "[200,102,1,0,200,100,0,200]"),
        ("slash", "101"),
    ];

    let final_state = "{PC_MAX = 8, QF = 6, QS = 1, blank = 200, left = 100, leftsp = 0, leftstk = [0, 0, 0, 0], pc = 0, q = 6, q1 = [1, 2, 3, 3, 3, 4, 5, 5], q2 = [2, 3, 4, 2, 4, 5, 4, 6], right = 102, rightsp = 0, rightstk = [0, 0, 0, 0], s = 200, s1 = [200, 101, 0, 1, 200, 101, 0, 200], s2 = [200, 102, 1, 0, 200, 100, 0, 200], slash = 101}\n";
    let zero_state = "{PC_MAX = 0, QF = 0, QS = 0, blank = 0, left = 0, leftsp = 0, leftstk = [0, 0, 0, 0], pc = 0, q = 0, q1 = [0, 0, 0, 0, 0, 0, 0, 0], q2 = [0, 0, 0, 0, 0, 0, 0, 0], right = 0, rightsp = 0, rightstk = [0, 0, 0, 0], s = 0, s1 = [0, 0, 0, 0, 0, 0, 0, 0], s2 = [0, 0, 0, 0, 0, 0, 0, 0], slash = 0}\n";

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout(final_state);

    let mut command = Command::cargo_bin("reverie").expect("binary exists");
    command.arg("reverse").arg(example);
    for (name, value) in final_vars {
        command.arg("--var").arg(format!("{name}={value}"));
    }
    command.assert().success().stdout(zero_state);
}

#[test]
fn rle_compression_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/rle_compression.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args([
            "--var",
            "n=8",
            "--var",
            "run=0",
            "--var",
            "data=[2,2,2,5,5,1,1,1]",
            "--var",
            "symbols=[0,0,0,0,0,0,0,0]",
            "--var",
            "counts=[0,0,0,0,0,0,0,0]",
        ])
        .assert()
        .success()
        .stdout("{counts = [3, 2, 3, 0, 0, 0, 0, 0], data = [2, 2, 2, 5, 5, 1, 1, 1], n = 8, run = 3, symbols = [2, 5, 1, 0, 0, 0, 0, 0]}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "n=8",
            "--var",
            "run=3",
            "--var",
            "data=[2,2,2,5,5,1,1,1]",
            "--var",
            "symbols=[2,5,1,0,0,0,0,0]",
            "--var",
            "counts=[3,2,3,0,0,0,0,0]",
        ])
        .assert()
        .success()
        .stdout("{counts = [0, 0, 0, 0, 0, 0, 0, 0], data = [2, 2, 2, 5, 5, 1, 1, 1], n = 8, run = 0, symbols = [0, 0, 0, 0, 0, 0, 0, 0]}\n");
}

#[test]
fn procedure_call_loop_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/procedure_call_loop.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args([
            "--var", "n=5", "--var", "i=0", "--var", "x=0", "--var", "y=1",
        ])
        .assert()
        .success()
        .stdout("{i = 5, n = 5, x = 0, y = 1}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var", "n=5", "--var", "i=5", "--var", "x=0", "--var", "y=1",
        ])
        .assert()
        .success()
        .stdout("{i = 0, n = 5, x = 0, y = 1}\n");
}

#[test]
fn element_args_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/element_args.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "xs=[0,1]"])
        .assert()
        .success()
        .stdout("{xs = [1, 1]}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "xs=[1,1]"])
        .assert()
        .success()
        .stdout("{xs = [0, 1]}\n");
}

#[test]
fn janus_root_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/janus_root.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "num=27"])
        .assert()
        .success()
        .stdout("{bit = 0, num = 2, root = 5, z = 0}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var", "num=2", "--var", "root=5", "--var", "z=0", "--var", "bit=0",
        ])
        .assert()
        .success()
        .stdout("{bit = 0, num = 27, root = 0, z = 0}\n");
}

#[test]
fn janus_factor_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/janus_factor.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "num=84"])
        .assert()
        .success()
        .stdout("{fact = [0, 2, 2, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], i = 0, num = 0, try = 0, z = 0}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "num=0",
            "--var",
            "try=0",
            "--var",
            "z=0",
            "--var",
            "i=0",
            "--var",
            "fact=[0,2,2,3,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]",
        ])
        .assert()
        .success()
        .stdout("{fact = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], i = 0, num = 84, try = 0, z = 0}\n");
}

#[test]
fn check_unitful_array_example_succeeds_and_erases_units() {
    let example = workspace_root().join("examples/array_units.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .assert()
        .success()
        .stdout("{}\n");
}

#[test]
fn check_units_example_succeeds_and_runtime_erases_units() {
    let example = workspace_root().join("examples/units.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .assert()
        .success()
        .stdout("{}\n");
}

#[test]
fn check_unit_exponent_example_succeeds_and_runtime_erases_units() {
    let example = workspace_root().join("examples/units_exponents.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .assert()
        .success()
        .stdout("{}\n");
}

#[test]
fn typed_top_level_seed_units_are_checked() {
    let example = workspace_root().join("examples/top_level_units.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args([
            "--var",
            "speed=0",
            "--type",
            "speed=int<m/s>",
            "--var",
            "distance=100",
            "--type",
            "distance=int<m>",
            "--var",
            "time=10",
            "--type",
            "time=int<s>",
        ])
        .assert()
        .success()
        .stdout("{distance = 100, speed = 10, time = 10}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "speed=10",
            "--type",
            "speed=int<m/s>",
            "--var",
            "distance=100",
            "--type",
            "distance=int<m>",
            "--var",
            "time=10",
            "--type",
            "time=int<s>",
        ])
        .assert()
        .success()
        .stdout("{distance = 100, speed = 0, time = 10}\n");
}

#[test]
fn typed_top_level_seed_unit_mismatch_fails_check() {
    let example = workspace_root().join("examples/top_level_unit_mismatch.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .args(["--type", "speed=int<m/s>", "--type", "distance=int<m>"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unit mismatch"))
        .stderr(predicate::str::contains("expected unit `m*s^-1`"))
        .stderr(predicate::str::contains("found unit `m`"));
}

#[test]
fn check_unit_mismatch_example_fails() {
    let example = workspace_root().join("examples/unit_mismatch.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("unit mismatch"))
        .stderr(predicate::str::contains("expected unit `m*s^-1`"))
        .stderr(predicate::str::contains("found unit `m`"));
}

#[test]
fn refinement_example_runs_both_directions() {
    let example = workspace_root().join("examples/refinement.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .assert()
        .success()
        .stdout("{}\n");
}

#[test]
fn unitful_loop_refinement_example_runs_both_directions() {
    let example = workspace_root().join("examples/refinement_units_loop.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout("{}\n");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .assert()
        .success()
        .stdout("{}\n");
}

#[test]
fn refinement_violation_fails_at_runtime() {
    let example = workspace_root().join("examples/refinement_violation.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("refinement for `n`"))
        .stderr(predicate::str::contains("runtime failed here"));
}

#[test]
fn unitful_loop_refinement_violation_fails_at_runtime() {
    let example = workspace_root().join("examples/refinement_units_loop_violation.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .assert()
        .failure()
        .stderr(predicate::str::contains("refinement for `distance`"))
        .stderr(predicate::str::contains("runtime failed here"));
}

#[test]
fn array_out_of_bounds_reports_runtime_source_span() {
    let example = workspace_root().join("examples/array_oob.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--var", "xs=[1,2,3]"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("out of bounds"))
        .stderr(predicate::str::contains("runtime failed here"));
}

#[test]
fn scrub_dump_prints_timeline_without_tui() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("scrub")
        .arg(example)
        .args([
            "--var", "n=2", "--var", "i=0", "--var", "a=0", "--var", "b=1", "--watch", "a",
            "--watch", "i", "--dump",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("start"))
        .stdout(predicate::str::contains("loop exit"))
        .stdout(predicate::str::contains("{a = 1, b = 2, i = 2, n = 2}"))
        .stdout(predicate::str::contains("watch a:"))
        .stdout(predicate::str::contains("watch i:"));
}

#[test]
fn scrub_dump_rejects_unknown_watch_name() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("scrub")
        .arg(example)
        .args([
            "--var", "n=2", "--var", "i=0", "--var", "a=0", "--var", "b=1", "--watch", "nn",
            "--dump",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown --watch name(s) `nn`"));
}

#[test]
fn scrub_dump_rejects_duplicate_watch_name() {
    let example = workspace_root().join("examples/fib.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("scrub")
        .arg(example)
        .args([
            "--var", "n=2", "--var", "i=0", "--var", "a=0", "--var", "b=1", "--watch", "a",
            "--watch", "a", "--dump",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "duplicate --watch annotation for `a`",
        ));
}

#[test]
fn scrub_dump_supports_tape_io() {
    let example = workspace_root().join("examples/io.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("scrub")
        .arg(example)
        .args(["--input", "7", "--dump"])
        .assert()
        .success()
        .stdout(predicate::str::contains("write x = 0"))
        .stdout(predicate::str::contains("read x <- 7"))
        .stdout(predicate::str::contains("{x = 7}"));
}

#[test]
fn scrub_dump_traces_procedure_body_frames() {
    let example = workspace_root().join("examples/proc.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("scrub")
        .arg(example)
        .args(["--var", "n=4", "--dump"])
        .assert()
        .success()
        .stdout(predicate::str::contains("call bump enter"))
        .stdout(predicate::str::contains("{x = 4}"))
        .stdout(predicate::str::contains("x +="))
        .stdout(predicate::str::contains("{t = 4, x = 5}"))
        .stdout(predicate::str::contains("call bump exit"))
        .stdout(predicate::str::contains("{n = 5}"));
}

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::json;
use sha2::{Digest, Sha256};

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

fn shell_quote_arg(value: &str) -> String {
    if value.chars().all(|char| {
        char.is_ascii_alphanumeric() || matches!(char, '_' | '-' | '.' | '/' | ':' | '=')
    }) {
        return value.to_owned();
    }

    format!("'{}'", value.replace('\'', "'\\''"))
}

fn sha256_json(value: &serde_json::Value) -> String {
    let bytes = serde_json::to_vec(value).expect("JSON fingerprint payload serializes");
    let normalized: serde_json::Value =
        serde_json::from_slice(&bytes).expect("JSON fingerprint payload normalizes");
    let normalized_bytes =
        serde_json::to_vec(&normalized).expect("normalized JSON fingerprint payload serializes");
    sha256_bytes(&normalized_bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_lower(&digest)
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn resign_mnist_training_bundle(root: &Path, bundle: &mut serde_json::Value) {
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .remove("fingerprints");
    let train_source = std::fs::read(root.join("examples/mnist_reversible_step.rev"))
        .expect("train source is readable");
    let identify_source = std::fs::read(root.join("examples/mnist_identify.rev"))
        .expect("identify source is readable");
    let train_source = sha256_bytes(&train_source);
    let identify_source = sha256_bytes(&identify_source);
    let mut computation = json!({
        "train_source": train_source.clone(),
        "identify_source": identify_source.clone(),
        "final_model": &bundle["final_model"],
        "witness_trace": &bundle["witness_trace"],
    });
    if let Some(proof) = bundle.get("proof") {
        computation
            .as_object_mut()
            .expect("training computation is an object")
            .insert("proof".to_owned(), proof.clone());
    }
    if let Some(lineage_ledger) = bundle.get("lineage_ledger") {
        computation
            .as_object_mut()
            .expect("training computation is an object")
            .insert("lineage_ledger".to_owned(), lineage_ledger.clone());
    }
    let fingerprints = json!({
        "algorithm": "sha256",
        "train_source": train_source,
        "identify_source": identify_source,
        "computation": sha256_json(&computation),
        "final_model": sha256_json(&bundle["final_model"]),
        "witness_trace": sha256_json(&bundle["witness_trace"]),
        "proof": sha256_json(bundle.get("proof").unwrap_or(&serde_json::Value::Null)),
        "lineage_ledger": sha256_json(bundle.get("lineage_ledger").unwrap_or(&serde_json::Value::Null)),
        "report": sha256_json(&bundle["report"]),
        "payload": sha256_json(bundle),
    });
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints);
}

fn resign_mnist_inference_bundle(root: &Path, bundle: &mut serde_json::Value) {
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .remove("fingerprints");
    let identify_source = std::fs::read(root.join("examples/mnist_identify.rev"))
        .expect("identify source is readable");
    let identify_source = sha256_bytes(&identify_source);
    let mut computation = json!({
        "identify_source": identify_source.clone(),
        "model": &bundle["model"],
        "sample": &bundle["sample"],
        "result": &bundle["result"],
    });
    if let Some(proof) = bundle.get("proof") {
        computation
            .as_object_mut()
            .expect("inference computation is an object")
            .insert("proof".to_owned(), proof.clone());
    }
    let fingerprints = json!({
        "algorithm": "sha256",
        "identify_source": identify_source,
        "computation": sha256_json(&computation),
        "model": sha256_json(&bundle["model"]),
        "sample": sha256_json(&bundle["sample"]),
        "result": sha256_json(&bundle["result"]),
        "proof": sha256_json(bundle.get("proof").unwrap_or(&serde_json::Value::Null)),
        "report": sha256_json(&bundle["report"]),
        "payload": sha256_json(bundle),
    });
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints);
}

fn resign_mnist_step_bundle(root: &Path, bundle: &mut serde_json::Value) {
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .remove("fingerprints");
    let train_source = std::fs::read(root.join("examples/mnist_reversible_step.rev"))
        .expect("train source is readable");
    let train_source = sha256_bytes(&train_source);
    let mut computation = json!({
        "train_source": train_source.clone(),
        "before_model": &bundle["before_model"],
        "after_model": &bundle["after_model"],
        "sample": &bundle["sample"],
        "witness": &bundle["witness"],
    });
    if let Some(proof) = bundle.get("proof") {
        computation
            .as_object_mut()
            .expect("step computation is an object")
            .insert("proof".to_owned(), proof.clone());
    }
    let fingerprints = json!({
        "algorithm": "sha256",
        "train_source": train_source,
        "computation": sha256_json(&computation),
        "before_model": sha256_json(&bundle["before_model"]),
        "after_model": sha256_json(&bundle["after_model"]),
        "sample": sha256_json(&bundle["sample"]),
        "witness": sha256_json(&bundle["witness"]),
        "update": sha256_json(&bundle["update"]),
        "proof": sha256_json(bundle.get("proof").unwrap_or(&serde_json::Value::Null)),
        "payload": sha256_json(bundle),
    });
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints);
}

fn resign_mnist_model_bundle(root: &Path, bundle: &mut serde_json::Value) {
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .remove("fingerprints");
    let train_source = std::fs::read(root.join("examples/mnist_reversible_step.rev"))
        .expect("train source is readable");
    let identify_source = std::fs::read(root.join("examples/mnist_identify.rev"))
        .expect("identify source is readable");
    let train_source = sha256_bytes(&train_source);
    let identify_source = sha256_bytes(&identify_source);
    let computation = json!({
        "train_source": train_source.clone(),
        "identify_source": identify_source.clone(),
        "model": &bundle["model"],
        "provenance": &bundle["provenance"],
    });
    let fingerprints = json!({
        "algorithm": "sha256",
        "train_source": train_source,
        "identify_source": identify_source,
        "computation": sha256_json(&computation),
        "model": sha256_json(&bundle["model"]),
        "provenance": sha256_json(&bundle["provenance"]),
        "report": sha256_json(&bundle["report"]),
        "payload": sha256_json(bundle),
    });
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints);
}

fn resign_mnist_sample_set_bundle(bundle: &mut serde_json::Value) {
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .remove("fingerprints");
    let mut computation = json!({
        "source_audit_bundle": &bundle["source_audit_bundle"],
        "samples": &bundle["samples"],
        "report": &bundle["report"],
    });
    if let Some(proof) = bundle.get("proof") {
        computation
            .as_object_mut()
            .expect("sample set computation is an object")
            .insert("proof".to_owned(), proof.clone());
    }
    let fingerprints = json!({
        "algorithm": "sha256",
        "computation": sha256_json(&computation),
        "source_audit": sha256_json(&bundle["source_audit_bundle"]),
        "samples": sha256_json(&bundle["samples"]),
        "proof": sha256_json(bundle.get("proof").unwrap_or(&serde_json::Value::Null)),
        "report": sha256_json(&bundle["report"]),
        "payload": sha256_json(bundle),
    });
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints);
}

fn resign_mnist_evaluation_bundle(root: &Path, bundle: &mut serde_json::Value) {
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .remove("fingerprints");
    let identify_source = std::fs::read(root.join("examples/mnist_identify.rev"))
        .expect("identify source is readable");
    let identify_source = sha256_bytes(&identify_source);
    let gate_policy = bundle
        .get("gate_policy")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let gate = bundle
        .get("gate")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let computation = json!({
        "identify_source": identify_source.clone(),
        "model": &bundle["model"],
        "samples": &bundle["samples"],
        "summary": &bundle["summary"],
        "proof": &bundle["proof"],
        "rows": &bundle["rows"],
        "gate_policy": gate_policy,
        "gate": gate,
    });
    let fingerprints = json!({
        "algorithm": "sha256",
        "identify_source": identify_source,
        "computation": sha256_json(&computation),
        "model": sha256_json(&bundle["model"]),
        "samples": sha256_json(&bundle["samples"]),
        "summary": sha256_json(&bundle["summary"]),
        "proof": sha256_json(&bundle["proof"]),
        "rows": sha256_json(&bundle["rows"]),
        "gate_policy": sha256_json(bundle.get("gate_policy").unwrap_or(&serde_json::Value::Null)),
        "gate": sha256_json(bundle.get("gate").unwrap_or(&serde_json::Value::Null)),
        "report": sha256_json(&bundle["report"]),
        "payload": sha256_json(bundle),
    });
    bundle
        .as_object_mut()
        .expect("bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints);
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
fn legacy_janus_allows_update_that_reads_potentially_same_dynamic_array_cell() {
    let root = workspace_root();
    let example = root.join("target/test-legacy-dynamic-cell-update-alias.rev");
    std::fs::create_dir_all(example.parent().expect("target path has parent"))
        .expect("target directory can be created");
    std::fs::write(&example, "xs[0] += xs[i]\n").expect("test source can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg("--legacy-janus")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));
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
            "reverie explain {}",
            shell_quote_arg(&example.display().to_string())
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
            "reverie explain {}",
            shell_quote_arg(&example.display().to_string())
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
            "reverie explain {}",
            shell_quote_arg(&example.display().to_string())
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
            "run template: reverie run {}",
            shell_quote_arg(&example.display().to_string())
        )))
        .stdout(predicate::str::contains(format!(
            "inverse: reverie invert {}",
            shell_quote_arg(&example.display().to_string())
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
fn explain_reports_witness_store() {
    let example = workspace_root().join("examples/mnist_witness_tape_loop.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg(example)
        .arg("--json")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"dataset-shaped iterate loops\""))
        .stdout(predicate::str::contains("\"dataset_loops\": ["))
        .stdout(predicate::str::contains(
            "{\"index\":\"sample\",\"size_sources\":[\"labels\"]}",
        ))
        .stdout(predicate::str::contains("\"witness tapes\""))
        .stdout(predicate::str::contains("\"witness_store\": ["))
        .stdout(predicate::str::contains("\"logits_tape\""))
        .stdout(predicate::str::contains("\"error_tape\""))
        .stdout(predicate::str::contains("\"prediction_tape\""))
        .stdout(predicate::str::contains("\"correct_tape\""))
        .stdout(predicate::str::contains("\"witness_metrics\""))
        .stdout(predicate::str::contains("\"known_cells\":44"))
        .stdout(predicate::str::contains("\"known_payload_bytes\":352"))
        .stdout(predicate::str::contains(
            "{\"name\":\"logits_tape\",\"cells\":20,\"payload_bytes\":160}",
        ));
}

#[test]
fn explain_reports_dataset_shaped_witness_loop() {
    let example = workspace_root().join("examples/mnist_witness_tape_loop.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("- dataset-shaped iterate loops"))
        .stdout(predicate::str::contains("dataset loops:"))
        .stdout(predicate::str::contains("- sample: bound by labels"));
}

#[test]
fn explain_ml_profile_classifies_mlp_witness_kernel() {
    let example = workspace_root().join("examples/mnist_mlp_witness.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg("--ml")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ml profile:"))
        .stdout(predicate::str::contains("goal_fit: auditable_ml_kernel"))
        .stdout(predicate::str::contains("deterministic_q31_fixed_point"))
        .stdout(predicate::str::contains("known_witness_payload_bytes=1632"))
        .stdout(predicate::str::contains("replay_cost: forward_statements="))
        .stdout(predicate::str::contains(
            "non_injective_signal_calls: argmax=1, argmax_eq=1, relu=1",
        ));
}

#[test]
fn explain_ml_profile_json_reports_tensor_witness_q31_budget() {
    let example = workspace_root().join("examples/mnist_mlp_witness.rev");
    let output = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg("--ml")
        .arg("--json")
        .arg(example)
        .output()
        .expect("explain command runs");

    assert!(output.status.success());
    let result: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("explain output is JSON");
    let profile = &result["ml_profile"];
    assert_eq!(profile["schema"], "reverie_explain_ml_profile_v1");
    assert_eq!(profile["goal_fit"], "auditable_ml_kernel");
    assert_eq!(
        profile["tensor_metrics"]["known_witness_payload_bytes"].as_u64(),
        Some(1632)
    );
    let replay_cost = &profile["replay_cost"];
    let forward_statements = replay_cost["forward_statement_count"]
        .as_u64()
        .expect("forward statement count is present");
    let inverse_statements = replay_cost["inverse_statement_count"]
        .as_u64()
        .expect("inverse statement count is present");
    assert_eq!(
        replay_cost["roundtrip_statement_count"].as_u64(),
        Some(forward_statements + inverse_statements)
    );
    let known_state_bytes = replay_cost["known_state_tensor_payload_bytes"]
        .as_u64()
        .expect("known state tensor payload bytes are present");
    assert_eq!(
        replay_cost["known_witness_payload_bytes"].as_u64(),
        Some(1632)
    );
    assert_eq!(
        replay_cost["known_replay_payload_bytes"].as_u64(),
        Some(known_state_bytes + 1632)
    );
    assert!(
        replay_cost["witness_to_state_payload_ratio"]
            .as_f64()
            .expect("witness/state ratio is present")
            > 0.0
    );
    assert_eq!(profile["q31_builtin_calls"].as_u64(), Some(12));
    assert_eq!(
        profile["non_injective_signal_calls"]["relu"].as_u64(),
        Some(1)
    );
    assert!(
        profile["capabilities"]
            .as_array()
            .expect("capabilities is an array")
            .iter()
            .any(|capability| capability == "roundtrip_proof_ready")
    );
}

#[test]
fn explain_ml_profile_reports_lossy_preprocessing_signals() {
    let example = workspace_root().join("examples/reversible_normalize.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg("--ml")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "non_injective_signal_calls: normalize_q31=1",
        ));

    let output = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg("--ml")
        .arg("--json")
        .arg(example)
        .output()
        .expect("explain command runs");

    assert!(output.status.success());
    let result: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("explain output is JSON");
    let profile = &result["ml_profile"];

    assert_eq!(
        profile["non_injective_signal_calls"]["normalize_q31"].as_u64(),
        Some(1)
    );
    assert!(
        !profile["capabilities"]
            .as_array()
            .expect("capabilities is an array")
            .iter()
            .any(|capability| capability == "non_injective_signals_captured_as_state")
    );
}

#[test]
fn explain_ml_profile_reports_inference_margin_signals() {
    let example = workspace_root().join("examples/reversible_inference_trace.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg("--ml")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "non_injective_signal_calls: argmax=1, argmax_eq=1, rank_of=1, runner_up=1, top2_margin=1, top_k_contains=1, top_k_indices=1, top_k_values=1",
        ));

    let output = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg("--ml")
        .arg("--json")
        .arg(example)
        .output()
        .expect("explain command runs");

    assert!(output.status.success());
    let result: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("explain output is JSON");
    let profile = &result["ml_profile"];

    assert_eq!(
        profile["non_injective_signal_calls"]["runner_up"].as_u64(),
        Some(1)
    );
    assert_eq!(
        profile["non_injective_signal_calls"]["top2_margin"].as_u64(),
        Some(1)
    );
    assert_eq!(
        profile["non_injective_signal_calls"]["top_k_indices"].as_u64(),
        Some(1)
    );
    assert_eq!(
        profile["non_injective_signal_calls"]["top_k_values"].as_u64(),
        Some(1)
    );
    assert_eq!(
        profile["non_injective_signal_calls"]["top_k_contains"].as_u64(),
        Some(1)
    );
    assert_eq!(
        profile["non_injective_signal_calls"]["rank_of"].as_u64(),
        Some(1)
    );
}

#[test]
fn witness_type_cli_annotations_run() {
    let root = workspace_root();
    let example = root.join("target/test-witness-type-cli.rev");
    std::fs::write(&example, "trace += x").expect("example can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args([
            "--var",
            "x=[1,2]",
            "--var",
            "trace=[0,0]",
            "--type",
            "x=tensor<int,2>",
            "--type",
            "trace=witness<tensor<int,2>>",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("trace = [1, 2]"));

    let example = workspace_root().join("target/test-witness-type-cli.rev");
    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("explain")
        .arg(example)
        .args([
            "--type",
            "x=tensor<int,2>",
            "--type",
            "trace=witness<tensor<int,2>>",
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"witness tapes\""))
        .stdout(predicate::str::contains("\"witness_store\": [\"trace\"]"))
        .stdout(predicate::str::contains("\"known_cells\":2"))
        .stdout(predicate::str::contains("\"known_payload_bytes\":16"));
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
fn tensor_linear_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/tensor_linear.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains("y = [[58, 64], [139, 154]]"))
        .stdout(predicate::str::contains("qy = [[1073741824]]"));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "y=[[58,64],[139,154]]",
            "--var",
            "qy=[[1073741824]]",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("y = [[0, 0], [0, 0]]"))
        .stdout(predicate::str::contains("qy = [[0]]"));
}

#[test]
fn invertible_coupling_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/invertible_coupling.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "left = [2684354560, -805306368, 1073741824, -805306368]",
        ))
        .stdout(predicate::str::contains(
            "right = [2147483648, 536870912, 0, -536870912]",
        ));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "left=[2684354560,-805306368,1073741824,-805306368]",
            "--var",
            "right=[2147483648,536870912,0,-536870912]",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "left = [2147483648, -1073741824, 536870912, -536870912]",
        ))
        .stdout(predicate::str::contains(
            "right = [0, 1073741824, -536870912, 536870912]",
        ));
}

#[test]
fn triangular_residual_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/triangular_residual.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "x = [1610612736, -805306368, 536870912, -671088640]",
        ));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "x=[1610612736,-805306368,536870912,-671088640]"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "x = [2147483648, -1073741824, 536870912, -536870912]",
        ));
}

#[test]
fn mnist_reversible_step_example_checks() {
    let example = workspace_root().join("examples/mnist_reversible_step.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));
}

#[test]
fn mnist_identify_example_checks() {
    let example = workspace_root().join("examples/mnist_identify.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));
}

#[test]
fn mnist_witness_tape_example_checks() {
    let example = workspace_root().join("examples/mnist_witness_tape.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));
}

#[test]
fn mnist_witness_tape_loop_example_checks() {
    let example = workspace_root().join("examples/mnist_witness_tape_loop.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));
}

#[test]
fn mnist_mlp_witness_example_checks() {
    let example = workspace_root().join("examples/mnist_mlp_witness.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok:"));
}

#[test]
fn mnist_linear_runner_self_test_passes() {
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--self-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("self-test ok"))
        .stdout(predicate::str::contains(
            "trace: entries=100 payload_bytes=20000 bytes_per_step=200",
        ))
        .stdout(predicate::str::contains(
            "proof: model_bytes=62800 sample_bytes=78400 witness_bytes=20000 replay_bytes=161200",
        ))
        .stdout(predicate::str::contains(
            "reverse: checked=100 restored_initial_model=true",
        ));
}

#[test]
fn mnist_linear_runner_self_test_json_reports_audit_metrics() {
    let assert = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--self-test")
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&assert.get_output().stdout).expect("runner JSON report is UTF-8");
    let report: serde_json::Value = serde_json::from_str(stdout).expect("runner emits valid JSON");

    assert_eq!(report["kind"].as_str(), Some("reverie_mnist_linear_q31"));
    assert_eq!(report["self_test"].as_bool(), Some(true));
    assert_eq!(report["config"]["reverse_check"].as_bool(), Some(true));
    assert_eq!(report["train"]["samples"].as_u64(), Some(100));
    assert_eq!(report["eval"]["samples"].as_u64(), Some(10));
    assert_eq!(report["trace"]["entries"].as_u64(), Some(100));
    assert_eq!(report["trace"]["payload_bytes"].as_u64(), Some(20_000));
    assert_eq!(report["trace"]["bytes_per_step"].as_u64(), Some(200));
    assert_eq!(report["proof"]["enabled"].as_bool(), Some(true));
    assert_eq!(report["proof"]["entries"].as_u64(), Some(100));
    assert_eq!(
        report["proof"]["model_payload_bytes"].as_u64(),
        Some(62_800)
    );
    assert_eq!(
        report["proof"]["sample_payload_bytes"].as_u64(),
        Some(78_400)
    );
    assert_eq!(
        report["proof"]["witness_payload_bytes"].as_u64(),
        Some(20_000)
    );
    assert_eq!(
        report["proof"]["trace_replay_payload_bytes"].as_u64(),
        Some(98_400)
    );
    assert_eq!(
        report["proof"]["full_replay_payload_bytes"].as_u64(),
        Some(161_200)
    );
    assert_eq!(report["proof"]["sample_bytes_per_step"].as_u64(), Some(784));
    assert_eq!(
        report["proof"]["witness_bytes_per_step"].as_u64(),
        Some(200)
    );
    assert_eq!(
        report["proof"]["trace_replay_bytes_per_step"].as_u64(),
        Some(984)
    );
    assert_eq!(
        report["proof"]["forward_recompute_steps"].as_u64(),
        Some(100)
    );
    assert_eq!(
        report["proof"]["inverse_recompute_steps"].as_u64(),
        Some(100)
    );
    assert_eq!(report["reverse"]["checked"].as_u64(), Some(100));
    assert_eq!(
        report["reverse"]["restored_initial_model"].as_bool(),
        Some(true)
    );
    assert_eq!(
        report["memory"]["model_payload_bytes"].as_u64(),
        Some(62_800)
    );
    assert_eq!(
        report["memory"]["dataset_payload_bytes"].as_u64(),
        Some(47_100)
    );
    assert_eq!(
        report["memory"]["trace_payload_bytes"].as_u64(),
        Some(20_000)
    );
    assert_eq!(
        report["memory"]["estimated_payload_bytes"].as_u64(),
        Some(129_900)
    );
    assert!(
        report["memory"]["peak_rss_bytes"].is_u64() || report["memory"]["peak_rss_bytes"].is_null()
    );
}

#[test]
fn mnist_linear_runner_writes_and_verifies_replay_bundle() {
    let artifact_dir = workspace_root()
        .join("target/reverie-cli-tests")
        .join(format!(
            "mnist-linear-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock is after Unix epoch")
                .as_nanos()
        ));
    std::fs::create_dir_all(&artifact_dir).expect("test artifact directory can be created");
    let bundle = artifact_dir.join("self-test-replay-bundle.json");
    let _ = std::fs::remove_file(&bundle);

    let assert = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--self-test")
        .arg("--json")
        .arg("--audit-output")
        .arg(&bundle)
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&assert.get_output().stdout).expect("runner JSON report is UTF-8");
    let report: serde_json::Value = serde_json::from_str(stdout).expect("runner emits valid JSON");
    assert_eq!(
        report["config"]["audit_output"].as_str(),
        Some(bundle.display().to_string().as_str())
    );

    let bundle_text = std::fs::read_to_string(&bundle).expect("replay bundle is written");
    let bundle_json: serde_json::Value =
        serde_json::from_str(&bundle_text).expect("replay bundle is valid JSON");
    assert_eq!(
        bundle_json["kind"].as_str(),
        Some("reverie_mnist_linear_q31_replay_bundle")
    );
    assert_eq!(bundle_json["schema_version"].as_u64(), Some(1));
    assert_eq!(
        bundle_json["report"]["trace"]["entries"].as_u64(),
        Some(100)
    );
    assert_eq!(bundle_json["proof"]["entries"].as_u64(), Some(100));
    assert_eq!(
        bundle_json["proof"]["trace_replay_payload_bytes"].as_u64(),
        Some(98_400)
    );
    assert_eq!(
        bundle_json["proof"]["full_replay_payload_bytes"].as_u64(),
        Some(161_200)
    );
    assert_eq!(
        bundle_json["final_model"]["weights"]
            .as_array()
            .map(Vec::len),
        Some(784)
    );
    assert_eq!(
        bundle_json["final_model"]["bias"].as_array().map(Vec::len),
        Some(10)
    );
    let trace = bundle_json["witness_trace"]
        .as_array()
        .expect("bundle has witness trace");
    assert_eq!(trace.len(), 100);
    assert_eq!(trace[0]["image_u8"].as_array().map(Vec::len), Some(784));
    assert_eq!(trace[0]["logits"].as_array().map(Vec::len), Some(10));
    assert_eq!(trace[0]["error"].as_array().map(Vec::len), Some(10));
    assert_eq!(
        bundle_json["fingerprints"]["algorithm"].as_str(),
        Some("sha256")
    );
    assert_eq!(
        bundle_json["fingerprints"]["proof"].as_str(),
        Some(sha256_json(&bundle_json["proof"]).as_str())
    );
    assert_eq!(
        bundle_json["fingerprints"]["lineage_ledger"].as_str(),
        Some(sha256_json(&bundle_json["lineage_ledger"]).as_str())
    );
    assert_eq!(
        bundle_json["lineage_ledger"]["schema"].as_str(),
        Some("q31_training_lineage_ledger_v1")
    );
    let payload_fingerprint = bundle_json["fingerprints"]["payload"]
        .as_str()
        .expect("bundle has payload fingerprint")
        .to_owned();
    let computation_fingerprint = bundle_json["fingerprints"]["computation"]
        .as_str()
        .expect("bundle has computation fingerprint")
        .to_owned();
    assert_eq!(payload_fingerprint.len(), 64);
    assert_eq!(computation_fingerprint.len(), 64);
    assert_ne!(computation_fingerprint, payload_fingerprint);

    let audit_markdown = artifact_dir.join("self-test-audit-verification.md");
    let _ = std::fs::remove_file(&audit_markdown);
    let verify = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-audit")
        .arg(&bundle)
        .arg("--markdown-output")
        .arg(&audit_markdown)
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&verify.get_output().stdout).expect("verify JSON report is UTF-8");
    let verify: serde_json::Value = serde_json::from_str(stdout).expect("verify emits valid JSON");
    assert_eq!(
        verify["kind"].as_str(),
        Some("reverie_mnist_linear_q31_audit_verification")
    );
    assert_eq!(verify["checked"].as_u64(), Some(100));
    assert_eq!(
        verify["witnesses_match_forward_replay"].as_bool(),
        Some(true)
    );
    assert_eq!(verify["final_model_replayed"].as_bool(), Some(true));
    assert_eq!(verify["restored_initial_model"].as_bool(), Some(true));
    assert_eq!(verify["proof_matches"].as_bool(), Some(true));
    assert_eq!(verify["lineage_ledger_matches"].as_bool(), Some(true));
    assert_eq!(verify["proof"]["entries"].as_u64(), Some(100));
    assert_eq!(
        verify["lineage_ledger"]["payload"]["steps"].as_u64(),
        Some(100)
    );
    assert_eq!(
        verify["proof"]["full_replay_payload_bytes"].as_u64(),
        Some(161_200)
    );
    assert_eq!(
        verify["proof"]["forward_recompute_steps"].as_u64(),
        Some(100)
    );
    assert_eq!(verify["forward"]["checked"].as_u64(), Some(100));
    assert_eq!(verify["forward"]["witnesses_match"].as_bool(), Some(true));
    assert_eq!(
        verify["forward"]["final_model_matches"].as_bool(),
        Some(true)
    );
    assert_eq!(verify["reverse"]["checked"].as_u64(), Some(100));
    assert_eq!(
        verify["reverse"]["restored_initial_model"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verify["fingerprints"]["payload"].as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert_eq!(
        verify["fingerprints"]["computation"].as_str(),
        Some(computation_fingerprint.as_str())
    );
    assert_eq!(
        verify["markdown_output"].as_str(),
        Some(audit_markdown.display().to_string().as_str())
    );
    let audit_markdown_text =
        std::fs::read_to_string(&audit_markdown).expect("audit verification Markdown is written");
    assert!(audit_markdown_text.contains("# Reverie Training Audit Verification"));
    assert!(audit_markdown_text.contains("## Replay Cost"));
    assert!(audit_markdown_text.contains("## Lineage"));
    assert!(audit_markdown_text.contains("## Boundary Transitions"));
    assert!(
        audit_markdown_text.contains(
            &bundle_json["lineage_ledger"]["payload"]["final_chain"]
                .as_str()
                .expect("lineage final chain is present")[..12]
        )
    );

    let forward_mismatch = artifact_dir.join("self-test-replay-bundle-forward-mismatch.json");
    let mut forward_mismatch_json = bundle_json.clone();
    forward_mismatch_json["witness_trace"][0]["prediction"] = serde_json::Value::from(9_i64);
    resign_mnist_training_bundle(&workspace_root(), &mut forward_mismatch_json);
    std::fs::write(
        &forward_mismatch,
        serde_json::to_string_pretty(&forward_mismatch_json)
            .expect("forward-mismatch bundle serializes"),
    )
    .expect("forward-mismatch bundle can be written");

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-audit")
        .arg(&forward_mismatch)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "audit bundle lineage ledger does not match recomputed training lineage",
        ));

    let proof_mismatch = artifact_dir.join("self-test-replay-bundle-proof-mismatch.json");
    let mut proof_mismatch_json = bundle_json.clone();
    proof_mismatch_json["proof"]["full_replay_payload_bytes"] = serde_json::Value::from(1_u64);
    resign_mnist_training_bundle(&workspace_root(), &mut proof_mismatch_json);
    std::fs::write(
        &proof_mismatch,
        serde_json::to_string_pretty(&proof_mismatch_json)
            .expect("proof-mismatch bundle serializes"),
    )
    .expect("proof-mismatch bundle can be written");

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-audit")
        .arg(&proof_mismatch)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "audit bundle proof does not match recomputed training replay cost",
        ));

    let scan = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--scan-audit")
        .arg(&bundle)
        .arg("--audit-limit")
        .arg("3")
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&scan.get_output().stdout).expect("scan JSON report is UTF-8");
    let scan: serde_json::Value = serde_json::from_str(stdout).expect("scan emits valid JSON");
    assert_eq!(
        scan["kind"].as_str(),
        Some("reverie_mnist_linear_q31_audit_scan")
    );
    assert_eq!(
        scan["fingerprints"]["payload"].as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert_eq!(
        scan["fingerprints"]["computation"].as_str(),
        Some(computation_fingerprint.as_str())
    );
    assert_eq!(scan["summary"]["steps"].as_u64(), Some(100));
    let scan_correct = scan["summary"]["correct"]
        .as_u64()
        .expect("scan has correct count");
    let scan_incorrect = scan["summary"]["incorrect"]
        .as_u64()
        .expect("scan has incorrect count");
    assert_eq!(scan_correct + scan_incorrect, 100);
    assert_eq!(scan["summary"]["witness_mismatches"].as_u64(), Some(0));
    assert_eq!(scan["summary"]["lowest_margin_step"].as_u64(), Some(0));
    assert_eq!(scan["summary"]["lowest_margin"].as_i64(), Some(0));
    assert_eq!(
        scan["summary"]["trace_payload_bytes"].as_u64(),
        Some(20_000)
    );
    assert_eq!(scan["summary"]["bytes_per_step"].as_u64(), Some(200));
    assert!(scan["summary"]["largest_update_step"].is_u64());
    let by_label = scan["by_label"]
        .as_array()
        .expect("scan has label summaries");
    assert_eq!(by_label.len(), 10);
    assert_eq!(by_label[0]["label"].as_u64(), Some(0));
    assert_eq!(by_label[0]["samples"].as_u64(), Some(10));
    assert_eq!(
        by_label[0]["correct"].as_u64().unwrap_or(0)
            + by_label[0]["incorrect"].as_u64().unwrap_or(0),
        10
    );
    assert!(by_label[0]["min_margin_step"].is_u64());
    assert!(by_label[0]["min_margin"].is_i64());
    assert!(by_label[0]["largest_update_step"].is_u64());
    assert!(by_label[0]["max_abs_weight_delta"].is_i64());
    let top_confusions = scan["top_confusions"]
        .as_array()
        .expect("scan has confusion summaries");
    assert_eq!(top_confusions.len(), 3);
    assert!(top_confusions[0]["label"].is_i64());
    assert!(top_confusions[0]["prediction"].is_i64());
    assert!(top_confusions[0]["count"].is_u64());
    assert!(top_confusions[0]["first_step"].is_u64());
    assert!(top_confusions[0]["min_margin"].is_i64());
    assert!(top_confusions[0]["max_abs_weight_delta"].is_i64());
    let top_suspicious = scan["top_suspicious"]
        .as_array()
        .expect("scan has ranked rows");
    assert_eq!(top_suspicious.len(), 3);
    assert!(top_suspicious[0]["step"].is_u64());
    assert!(top_suspicious[0]["sample_index"].is_u64());
    assert!(top_suspicious[0]["predicted_logit"].is_i64());
    assert!(top_suspicious[0]["runner_up_digit"].is_u64());
    assert!(top_suspicious[0]["runner_up_logit"].is_i64());
    assert!(top_suspicious[0]["margin"].is_i64());
    assert!(top_suspicious[0]["active_pixel_count"].is_u64());
    assert!(top_suspicious[0]["max_abs_weight_delta"].is_i64());
    assert_eq!(
        top_suspicious[0]["prediction_matches_logits"].as_bool(),
        Some(true)
    );
    assert_eq!(
        top_suspicious[0]["correct_matches_logits"].as_bool(),
        Some(true)
    );
    let top_low_margin = scan["top_low_margin"]
        .as_array()
        .expect("scan has low-margin rows");
    assert_eq!(top_low_margin.len(), 3);
    assert_eq!(top_low_margin[0]["step"].as_u64(), Some(0));
    assert_eq!(top_low_margin[0]["margin"].as_i64(), Some(0));
    assert_eq!(top_low_margin[0]["runner_up_digit"].as_u64(), Some(1));
    let top_large_updates = scan["top_large_updates"]
        .as_array()
        .expect("scan has large-update rows");
    assert_eq!(top_large_updates.len(), 3);
    assert!(top_large_updates[0]["step"].is_u64());
    assert_eq!(
        top_large_updates[0]["max_abs_weight_delta"].as_i64(),
        Some(536_870_912)
    );
    assert!(top_large_updates[0]["margin"].is_i64());

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--scan-audit")
        .arg(&bundle)
        .arg("--audit-limit")
        .arg("2")
        .assert()
        .success()
        .stdout(predicate::str::contains("audit scan"))
        .stdout(predicate::str::contains("summary: steps=100"))
        .stdout(predicate::str::contains(
            "lowest_margin_step=0 lowest_margin=0",
        ))
        .stdout(predicate::str::contains("top_suspicious:"))
        .stdout(predicate::str::contains("by_label:"))
        .stdout(predicate::str::contains("label=0 samples=10"))
        .stdout(predicate::str::contains("top_confusions:"))
        .stdout(predicate::str::contains("top_low_margin:"))
        .stdout(predicate::str::contains("top_large_updates:"))
        .stdout(predicate::str::contains("margin="))
        .stdout(predicate::str::contains("witness_ok=true"));

    let gated_scan = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--scan-audit")
        .arg(&bundle)
        .arg("--audit-limit")
        .arg("1")
        .arg("--audit-min-accuracy")
        .arg("0")
        .arg("--audit-min-margin")
        .arg("0")
        .arg("--audit-max-witness-mismatches")
        .arg("0")
        .arg("--audit-max-weight-delta")
        .arg("536870912")
        .arg("--audit-require-label-coverage")
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&gated_scan.get_output().stdout).expect("gate JSON is UTF-8");
    let gated_scan: serde_json::Value =
        serde_json::from_str(stdout).expect("gated scan emits valid JSON");
    assert_eq!(gated_scan["gate"]["passed"].as_bool(), Some(true));
    assert_eq!(gated_scan["gate"]["checks"].as_array().unwrap().len(), 5);
    assert_eq!(
        gated_scan["gate"]["checks"][0]["metric"].as_str(),
        Some("accuracy_percent")
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--scan-audit")
        .arg(&bundle)
        .arg("--audit-min-margin")
        .arg("1")
        .assert()
        .failure()
        .stdout(predicate::str::contains("gate: failed"))
        .stdout(predicate::str::contains(
            "lowest_margin actual=0 requirement=\">= 1\" passed=false",
        ))
        .stderr(predicate::str::contains("audit gate failed"));

    let inspect = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-audit")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&inspect.get_output().stdout).expect("inspect JSON report is UTF-8");
    let step: serde_json::Value = serde_json::from_str(stdout).expect("inspect emits valid JSON");
    assert_eq!(
        step["kind"].as_str(),
        Some("reverie_mnist_linear_q31_audit_step")
    );
    assert_eq!(
        step["fingerprints"]["payload"].as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert_eq!(
        step["fingerprints"]["computation"].as_str(),
        Some(computation_fingerprint.as_str())
    );
    assert_eq!(step["step"].as_u64(), Some(0));
    assert_eq!(step["sample_index"].as_u64(), Some(0));
    assert_eq!(step["label"].as_i64(), Some(0));
    assert_eq!(step["prediction"].as_i64(), Some(0));
    assert_eq!(step["correct"].as_bool(), Some(true));
    assert_eq!(step["top_logits"][0]["digit"].as_u64(), Some(0));
    assert_eq!(step["top_logits"][0]["logit"].as_i64(), Some(0));
    assert_eq!(step["top_logits"][1]["digit"].as_u64(), Some(1));
    assert_eq!(step["top_logits"][1]["logit"].as_i64(), Some(0));
    assert_eq!(step["logit_margin"]["predicted_digit"].as_u64(), Some(0));
    assert_eq!(step["logit_margin"]["runner_up_digit"].as_u64(), Some(1));
    assert_eq!(step["logit_margin"]["margin"].as_i64(), Some(0));
    assert_eq!(
        step["witness_checks"]["prediction_matches_logits"].as_bool(),
        Some(true)
    );
    assert_eq!(
        step["witness_checks"]["correct_matches_logits"].as_bool(),
        Some(true)
    );
    assert_eq!(step["active_pixels"].as_array().map(Vec::len), Some(2));
    assert_eq!(step["update"]["bias_delta"][0].as_i64(), Some(536_870_912));
    assert_eq!(
        step["update"]["nonzero_weight_delta_count"].as_u64(),
        Some(2)
    );
    assert_eq!(
        step["update"]["top_weight_deltas"][0]["delta"].as_i64(),
        Some(536_870_912)
    );
    assert_eq!(step["model_window"]["reconstructed"].as_bool(), Some(true));
    assert_eq!(
        step["model_window"]["reversed_later_steps"].as_u64(),
        Some(99)
    );
    assert_eq!(step["model_window"]["bias_before"][0].as_i64(), Some(0));
    assert_eq!(
        step["model_window"]["bias_after"][0].as_i64(),
        Some(536_870_912)
    );
    assert_eq!(
        step["model_window"]["bias_delta_matches"].as_bool(),
        Some(true)
    );
    assert_eq!(
        step["model_window"]["weight_delta_matches"].as_bool(),
        Some(true)
    );
    assert_eq!(
        step["model_window"]["top_weight_windows"][0]["before"].as_i64(),
        Some(0)
    );
    assert_eq!(
        step["model_window"]["top_weight_windows"][0]["after"].as_i64(),
        Some(536_870_912)
    );
    assert_eq!(
        step["model_window"]["top_weight_windows"][0]["delta_matches"].as_bool(),
        Some(true)
    );
    assert_eq!(
        step["debug_contract"]["claim"].as_str(),
        Some("step_backward_from_model_update")
    );
    assert_eq!(step["debug_contract"]["passed"].as_bool(), Some(true));
    let debug_checks = step["debug_contract"]["checks"]
        .as_array()
        .expect("audit step has debug contract checks");
    assert_eq!(debug_checks.len(), 5);
    assert!(debug_checks.iter().all(|check| {
        check["metric"].as_str().is_some()
            && check["actual"].as_str().is_some()
            && check["requirement"].as_str().is_some()
            && check["passed"].as_bool() == Some(true)
    }));
    assert!(
        debug_checks
            .iter()
            .any(|check| { check["metric"].as_str() == Some("witness_recomputes_prediction") })
    );
    assert!(
        debug_checks
            .iter()
            .any(|check| { check["metric"].as_str() == Some("update_ledger_fingerprints") })
    );
    assert_eq!(
        step["debug_contract"]["replay_direction"]["backward_from_final_model_steps"].as_u64(),
        Some(99)
    );
    assert_eq!(
        step["debug_contract"]["debug_focus"]["reversibility"].as_str(),
        Some("model_window and witness_checks")
    );
    assert_eq!(step["selection"]["strategy"].as_str(), Some("explicit"));
    assert_eq!(step["selection"]["requested_step"].as_u64(), Some(0));
    assert_eq!(step["selection"]["selected_step"].as_u64(), Some(0));
    assert_eq!(
        step["selection"]["scan_lowest_margin_step"].as_u64(),
        scan["summary"]["lowest_margin_step"].as_u64()
    );
    assert_eq!(
        step["selection"]["scan_largest_update_step"].as_u64(),
        scan["summary"]["largest_update_step"].as_u64()
    );

    let strategy_inspect = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-audit")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("7")
        .arg("--audit-step-strategy")
        .arg("lowest-margin")
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&strategy_inspect.get_output().stdout)
        .expect("strategy inspect JSON report is UTF-8");
    let strategy_step: serde_json::Value =
        serde_json::from_str(stdout).expect("strategy inspect emits valid JSON");
    assert_eq!(
        strategy_step["selection"]["strategy"].as_str(),
        Some("lowest-margin")
    );
    assert_eq!(
        strategy_step["selection"]["requested_step"].as_u64(),
        Some(7)
    );
    assert_eq!(
        strategy_step["selection"]["selected_step"].as_u64(),
        scan["summary"]["lowest_margin_step"].as_u64()
    );
    assert_eq!(
        strategy_step["selection"]["matches_scan_lowest_margin"].as_bool(),
        Some(true)
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-audit")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .assert()
        .success()
        .stdout(predicate::str::contains("audit step 0"))
        .stdout(predicate::str::contains(
            "witness_checks: computed_prediction=0 prediction_matches_logits=true correct_matches_logits=true",
        ))
        .stdout(predicate::str::contains("top_logits: [d0=0, d1=0, d2=0]"))
        .stdout(predicate::str::contains(
            "logit_margin: predicted=0 runner_up=1 margin=0",
        ))
        .stdout(predicate::str::contains("bias_delta: [536870912"))
        .stdout(predicate::str::contains(
            "model_window: reversed_later_steps=99 bias_delta_matches=true weight_delta_matches=true",
        ))
        .stdout(predicate::str::contains(
            "debug_contract: claim=step_backward_from_model_update passed=true checks=5",
        ));

    let inspect_markdown = artifact_dir.join("self-test-step-debug-inspect.md");
    let _ = std::fs::remove_file(&inspect_markdown);
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-audit")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .arg("--markdown-output")
        .arg(&inspect_markdown)
        .assert()
        .success()
        .stdout(predicate::str::contains("markdown:"));
    let inspect_markdown_text =
        std::fs::read_to_string(&inspect_markdown).expect("inspect Markdown is written");
    assert!(inspect_markdown_text.contains("# Reverie Training Step Debug"));
    assert!(inspect_markdown_text.contains("step_backward_from_model_update"));
    assert!(inspect_markdown_text.contains("## Top Weight Deltas"));
    assert!(inspect_markdown_text.contains("## Ledgers"));
    assert!(inspect_markdown_text.contains("## Selection"));
    assert!(inspect_markdown_text.contains("| `explicit` | 0 | 0 |"));
    assert!(inspect_markdown_text.contains("6f333f1a8176"));

    let step_bundle = artifact_dir.join("self-test-step-bundle.json");
    let _ = std::fs::remove_file(&step_bundle);
    let step_output = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-audit")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .arg("--step-output")
        .arg(&step_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&step_output.get_output().stdout)
        .expect("step output JSON report is UTF-8");
    let step_output: serde_json::Value =
        serde_json::from_str(stdout).expect("step output emits valid JSON");
    assert_eq!(
        step_output["step_output"].as_str(),
        Some(step_bundle.display().to_string().as_str())
    );

    let step_bundle_text = std::fs::read_to_string(&step_bundle).expect("step bundle is written");
    let step_bundle_json: serde_json::Value =
        serde_json::from_str(&step_bundle_text).expect("step bundle is valid JSON");
    assert_eq!(
        step_bundle_json["kind"].as_str(),
        Some("reverie_mnist_linear_q31_step_replay_bundle")
    );
    assert_eq!(step_bundle_json["schema_version"].as_u64(), Some(1));
    assert_eq!(
        step_bundle_json["source_training_bundle"]["fingerprints"]["computation"].as_str(),
        Some(computation_fingerprint.as_str())
    );
    assert_eq!(
        step_bundle_json["source_training_bundle"]["reversed_later_steps"].as_u64(),
        Some(99)
    );
    assert_eq!(
        step_bundle_json["before_model"]["weights"]
            .as_array()
            .map(Vec::len),
        Some(784)
    );
    assert_eq!(
        step_bundle_json["after_model"]["bias"][0].as_i64(),
        Some(536_870_912)
    );
    assert_eq!(
        step_bundle_json["sample"]["image_u8"]
            .as_array()
            .map(Vec::len),
        Some(784)
    );
    assert_eq!(step_bundle_json["sample"]["label"].as_i64(), Some(0));
    assert_eq!(
        step_bundle_json["witness"]["logits"]
            .as_array()
            .map(Vec::len),
        Some(10)
    );
    assert_eq!(step_bundle_json["witness"]["prediction"].as_i64(), Some(0));
    assert_eq!(
        step_bundle_json["update"]["bias_delta"][0].as_i64(),
        Some(536_870_912)
    );
    assert_eq!(
        step_bundle_json["proof"]["claim"].as_str(),
        Some("deterministic_q31_training_step_replay")
    );
    assert_eq!(
        step_bundle_json["proof"]["kernel"].as_str(),
        Some("examples/mnist_reversible_step.rev")
    );
    assert_eq!(
        step_bundle_json["proof"]["model_payload_bytes"].as_u64(),
        Some(125_600)
    );
    assert_eq!(
        step_bundle_json["proof"]["sample_payload_bytes"].as_u64(),
        Some(785)
    );
    assert_eq!(
        step_bundle_json["proof"]["witness_payload_bytes"].as_u64(),
        Some(184)
    );
    assert_eq!(
        step_bundle_json["proof"]["derived_update_payload_bytes"].as_u64(),
        Some(216)
    );
    assert_eq!(
        step_bundle_json["proof"]["trace_payload_bytes"].as_u64(),
        Some(0)
    );
    assert_eq!(
        step_bundle_json["proof"]["replay_payload_bytes"].as_u64(),
        Some(126_785)
    );
    assert_eq!(
        step_bundle_json["proof"]["nonzero_bias_delta_count"].as_u64(),
        Some(1)
    );
    assert_eq!(
        step_bundle_json["proof"]["nonzero_weight_delta_count"].as_u64(),
        Some(2)
    );
    assert_eq!(
        step_bundle_json["proof"]["update_ledger_fingerprints"]["algorithm"].as_str(),
        Some("sha256")
    );
    assert_eq!(
        step_bundle_json["proof"]["update_ledger_fingerprints"]["bias_delta"].as_str(),
        step_bundle_json["update"]["bias_delta_ledger_fingerprint"].as_str()
    );
    assert_eq!(
        step_bundle_json["proof"]["update_ledger_fingerprints"]["weight_delta"].as_str(),
        step_bundle_json["update"]["weight_delta_ledger_fingerprint"].as_str()
    );
    assert_eq!(
        step_bundle_json["proof"]["checks"]["witnesses_match_forward_replay"].as_bool(),
        Some(true)
    );
    assert_eq!(
        step_bundle_json["proof"]["checks"]["update_matches_witnesses"].as_bool(),
        Some(true)
    );
    assert_eq!(
        step_bundle_json["proof"]["fingerprints"]["before_model"].as_str(),
        Some(sha256_json(&step_bundle_json["before_model"]).as_str())
    );
    assert_eq!(
        step_bundle_json["proof"]["fingerprints"]["after_model"].as_str(),
        Some(sha256_json(&step_bundle_json["after_model"]).as_str())
    );
    assert_eq!(
        step_bundle_json["proof"]["fingerprints"]["sample"].as_str(),
        Some(sha256_json(&step_bundle_json["sample"]).as_str())
    );
    assert_eq!(
        step_bundle_json["proof"]["fingerprints"]["witness"].as_str(),
        Some(sha256_json(&step_bundle_json["witness"]).as_str())
    );
    assert_eq!(
        step_bundle_json["proof"]["fingerprints"]["update"].as_str(),
        Some(sha256_json(&step_bundle_json["update"]).as_str())
    );
    assert_eq!(
        step_bundle_json["fingerprints"]["algorithm"].as_str(),
        Some("sha256")
    );
    assert_eq!(
        step_bundle_json["fingerprints"]["proof"].as_str(),
        Some(sha256_json(&step_bundle_json["proof"]).as_str())
    );
    let step_payload_fingerprint = step_bundle_json["fingerprints"]["payload"]
        .as_str()
        .expect("step bundle has payload fingerprint")
        .to_owned();
    let step_computation_fingerprint = step_bundle_json["fingerprints"]["computation"]
        .as_str()
        .expect("step bundle has computation fingerprint")
        .to_owned();
    assert_eq!(step_payload_fingerprint.len(), 64);
    assert_eq!(step_computation_fingerprint.len(), 64);

    let verify_step = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-step")
        .arg(&step_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&verify_step.get_output().stdout)
        .expect("verify step JSON report is UTF-8");
    let verify_step: serde_json::Value =
        serde_json::from_str(stdout).expect("verify step emits valid JSON");
    assert_eq!(
        verify_step["kind"].as_str(),
        Some("reverie_mnist_linear_q31_step_verification")
    );
    assert_eq!(
        verify_step["fingerprints"]["payload"].as_str(),
        Some(step_payload_fingerprint.as_str())
    );
    assert_eq!(
        verify_step["fingerprints"]["computation"].as_str(),
        Some(step_computation_fingerprint.as_str())
    );
    assert_eq!(verify_step["proof_matches"].as_bool(), Some(true));
    assert_eq!(
        verify_step["proof"]["claim"].as_str(),
        Some("deterministic_q31_training_step_replay")
    );
    assert_eq!(
        verify_step["proof"]["replay_payload_bytes"].as_u64(),
        Some(126_785)
    );
    assert_eq!(
        verify_step["proof"]["update_ledger_fingerprints"]["bias_delta"].as_str(),
        step_bundle_json["update"]["bias_delta_ledger_fingerprint"].as_str()
    );
    assert_eq!(
        verify_step["proof"]["update_ledger_fingerprints"]["weight_delta"].as_str(),
        step_bundle_json["update"]["weight_delta_ledger_fingerprint"].as_str()
    );
    assert_eq!(
        verify_step["proof"]["fingerprints"]["update"].as_str(),
        Some(sha256_json(&step_bundle_json["update"]).as_str())
    );
    assert_eq!(
        verify_step["forward"]["witnesses_match"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verify_step["forward"]["after_model_matches"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verify_step["reverse"]["before_model_restored"].as_bool(),
        Some(true)
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-step")
        .arg(&step_bundle)
        .assert()
        .success()
        .stdout(predicate::str::contains("step audit ok"))
        .stdout(predicate::str::contains("before_model_restored=true"))
        .stdout(predicate::str::contains(
            "proof: claim=deterministic_q31_training_step_replay model_bytes=125600 sample_bytes=785 witness_bytes=184 update_bytes=216",
        ));

    let verify_step_markdown = artifact_dir.join("self-test-step-debug-verify.md");
    let _ = std::fs::remove_file(&verify_step_markdown);
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-step")
        .arg(&step_bundle)
        .arg("--markdown-output")
        .arg(&verify_step_markdown)
        .assert()
        .success()
        .stdout(predicate::str::contains("markdown:"));
    let verify_step_markdown_text =
        std::fs::read_to_string(&verify_step_markdown).expect("verify-step Markdown is written");
    assert!(verify_step_markdown_text.contains("# Reverie Training Step Debug"));
    assert!(verify_step_markdown_text.contains("## Replay Proof"));
    assert!(verify_step_markdown_text.contains("deterministic_q31_training_step_replay"));
    assert!(verify_step_markdown_text.contains("| `before_model_restored` | true |"));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--step-output")
        .arg(artifact_dir.join("unused-step-bundle.json"))
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--step-output requires --inspect-audit",
        ));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--markdown-output")
        .arg(artifact_dir.join("unused-step-debug.md"))
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--markdown-output requires --verify-audit, --inspect-audit, --verify-step, --inspect-inference, --inspect-model-inference, --inspect-evaluation, or --verify-inference",
        ));

    let tampered_step = artifact_dir.join("self-test-step-bundle-tampered.json");
    let proof_mismatch_step = artifact_dir.join("self-test-step-bundle-proof-mismatch.json");
    let mut proof_mismatch_step_json = step_bundle_json.clone();
    proof_mismatch_step_json["proof"]["replay_payload_bytes"] = serde_json::Value::from(1_u64);
    resign_mnist_step_bundle(&workspace_root(), &mut proof_mismatch_step_json);
    std::fs::write(
        &proof_mismatch_step,
        serde_json::to_string_pretty(&proof_mismatch_step_json)
            .expect("proof-mismatch step bundle serializes"),
    )
    .expect("proof-mismatch step bundle can be written");

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-step")
        .arg(&proof_mismatch_step)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "step bundle proof does not match recomputed step replay cost",
        ));

    let mut tampered_step_json = step_bundle_json;
    tampered_step_json["witness"]["prediction"] = serde_json::Value::from(9_i64);
    std::fs::write(
        &tampered_step,
        serde_json::to_string_pretty(&tampered_step_json).expect("tampered step bundle serializes"),
    )
    .expect("tampered step bundle can be written");

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-step")
        .arg(&tampered_step)
        .assert()
        .failure()
        .stderr(predicate::str::contains("fingerprint mismatch"));

    let model_bundle = artifact_dir.join("self-test-model-bundle.json");
    let _ = std::fs::remove_file(&model_bundle);
    let model_export = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--export-model")
        .arg(&bundle)
        .arg("--model-output")
        .arg(&model_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&model_export.get_output().stdout).expect("model export JSON is UTF-8");
    let model_export: serde_json::Value =
        serde_json::from_str(stdout).expect("model export emits valid JSON");
    assert_eq!(
        model_export["kind"].as_str(),
        Some("reverie_mnist_linear_q31_model_export")
    );
    assert_eq!(
        model_export["model_output"].as_str(),
        Some(model_bundle.display().to_string().as_str())
    );
    assert_eq!(model_export["checked"].as_u64(), Some(100));
    assert_eq!(
        model_export["source_fingerprints"]["payload"].as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert_eq!(model_export["restored_initial_model"].as_bool(), Some(true));

    let model_bundle_text =
        std::fs::read_to_string(&model_bundle).expect("model bundle is written");
    let model_bundle_json: serde_json::Value =
        serde_json::from_str(&model_bundle_text).expect("model bundle is valid JSON");
    assert_eq!(
        model_bundle_json["kind"].as_str(),
        Some("reverie_mnist_linear_q31_model_bundle")
    );
    assert_eq!(model_bundle_json["schema_version"].as_u64(), Some(1));
    assert_eq!(
        model_bundle_json["model"]["weights"]
            .as_array()
            .map(Vec::len),
        Some(784)
    );
    assert_eq!(
        model_bundle_json["model"]["bias"].as_array().map(Vec::len),
        Some(10)
    );
    assert_eq!(
        model_bundle_json["storage"]["model_payload_bytes"].as_u64(),
        Some(62_800)
    );
    assert_eq!(
        model_bundle_json["provenance"]["source_training_bundle"]["fingerprints"]["payload"]
            .as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert_eq!(
        model_bundle_json["report"]["source_audit_path"].as_str(),
        Some(bundle.display().to_string().as_str())
    );
    assert_eq!(
        model_bundle_json["report"]["source_fingerprints"]["payload"].as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert_eq!(
        model_bundle_json["report"]["training_steps"].as_u64(),
        Some(100)
    );
    assert_eq!(
        model_bundle_json["report"]["model_payload_bytes"].as_u64(),
        Some(62_800)
    );
    assert_eq!(
        model_bundle_json["provenance"]["proof"],
        model_bundle_json["provenance"]["training_report"]["proof"]
    );
    assert_eq!(
        model_bundle_json["provenance"]["proof"]["entries"].as_u64(),
        Some(100)
    );
    assert_eq!(
        model_bundle_json["provenance"]["proof"]["full_replay_payload_bytes"].as_u64(),
        Some(161_200)
    );
    let model_payload_fingerprint = model_bundle_json["fingerprints"]["payload"]
        .as_str()
        .expect("model bundle has payload fingerprint")
        .to_owned();
    let model_computation_fingerprint = model_bundle_json["fingerprints"]["computation"]
        .as_str()
        .expect("model bundle has computation fingerprint")
        .to_owned();
    assert_eq!(model_payload_fingerprint.len(), 64);
    assert_eq!(model_computation_fingerprint.len(), 64);
    assert_ne!(model_computation_fingerprint, model_payload_fingerprint);

    let verify_model = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-model")
        .arg(&model_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&verify_model.get_output().stdout).expect("verify model JSON is UTF-8");
    let verify_model: serde_json::Value =
        serde_json::from_str(stdout).expect("verify model emits valid JSON");
    assert_eq!(
        verify_model["kind"].as_str(),
        Some("reverie_mnist_linear_q31_model_verification")
    );
    assert_eq!(
        verify_model["fingerprints"]["payload"].as_str(),
        Some(model_payload_fingerprint.as_str())
    );
    assert_eq!(
        verify_model["fingerprints"]["computation"].as_str(),
        Some(model_computation_fingerprint.as_str())
    );
    assert_eq!(verify_model["shape_matches"].as_bool(), Some(true));
    assert_eq!(verify_model["provenance_matches"].as_bool(), Some(true));
    assert_eq!(verify_model["proof_matches"].as_bool(), Some(true));
    assert_eq!(verify_model["training_steps"].as_u64(), Some(100));
    assert_eq!(
        verify_model["source_audit_payload"].as_str(),
        Some(payload_fingerprint.as_str())
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-model")
        .arg(&model_bundle)
        .assert()
        .success()
        .stdout(predicate::str::contains("model ok"))
        .stdout(predicate::str::contains("shape_matches=true"))
        .stdout(predicate::str::contains("provenance_matches=true"))
        .stdout(predicate::str::contains("proof_matches=true"))
        .stdout(predicate::str::contains("training_steps=100"));

    let imported_model_source = artifact_dir.join("self-test-import-model.json");
    let imported_model_source_json = json!({
        "format": "weights_bias_q31_json",
        "model": model_bundle_json["model"].clone(),
    });
    std::fs::write(
        &imported_model_source,
        serde_json::to_string_pretty(&imported_model_source_json)
            .expect("imported model source serializes"),
    )
    .expect("imported model source can be written");
    let imported_model_source_fingerprint = sha256_json(&imported_model_source_json);
    let imported_model_bundle = artifact_dir.join("self-test-imported-model-bundle.json");
    let imported_model = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--import-model-json")
        .arg(&imported_model_source)
        .arg("--model-output")
        .arg(&imported_model_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&imported_model.get_output().stdout)
        .expect("model import JSON is UTF-8");
    let imported_model: serde_json::Value =
        serde_json::from_str(stdout).expect("model import emits valid JSON");
    assert_eq!(
        imported_model["kind"].as_str(),
        Some("reverie_mnist_linear_q31_model_import")
    );
    assert_eq!(
        imported_model["source_model_json_path"].as_str(),
        Some(imported_model_source.display().to_string().as_str())
    );
    assert_eq!(
        imported_model["source_model_json_fingerprint"].as_str(),
        Some(imported_model_source_fingerprint.as_str())
    );
    assert_eq!(
        imported_model["provenance_kind"].as_str(),
        Some("external_import")
    );

    let imported_model_bundle_text =
        std::fs::read_to_string(&imported_model_bundle).expect("imported model bundle is written");
    let imported_model_bundle_json: serde_json::Value =
        serde_json::from_str(&imported_model_bundle_text)
            .expect("imported model bundle is valid JSON");
    assert_eq!(
        imported_model_bundle_json["model"],
        model_bundle_json["model"]
    );
    assert_eq!(
        imported_model_bundle_json["report"]["provenance_kind"].as_str(),
        Some("external_import")
    );
    assert!(imported_model_bundle_json["report"]["training_steps"].is_null());
    assert_eq!(
        imported_model_bundle_json["provenance"]["source_model_json"]["fingerprint"].as_str(),
        Some(imported_model_source_fingerprint.as_str())
    );

    let imported_model_payload_fingerprint = imported_model_bundle_json["fingerprints"]["payload"]
        .as_str()
        .expect("imported model bundle has payload fingerprint")
        .to_owned();
    let verify_imported_model = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-model")
        .arg(&imported_model_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&verify_imported_model.get_output().stdout)
        .expect("verify imported model JSON is UTF-8");
    let verify_imported_model: serde_json::Value =
        serde_json::from_str(stdout).expect("verify imported model emits valid JSON");
    assert_eq!(
        verify_imported_model["fingerprints"]["payload"].as_str(),
        Some(imported_model_payload_fingerprint.as_str())
    );
    assert_eq!(
        verify_imported_model["provenance_kind"].as_str(),
        Some("external_import")
    );
    assert!(verify_imported_model["proof_matches"].is_null());
    assert!(verify_imported_model["training_steps"].is_null());
    assert_eq!(
        verify_imported_model["source_model_json_checked"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verify_imported_model["source_model_json"]["fingerprint"].as_str(),
        Some(imported_model_source_fingerprint.as_str())
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-model")
        .arg(&imported_model_bundle)
        .assert()
        .success()
        .stdout(predicate::str::contains("provenance_kind=external_import"))
        .stdout(predicate::str::contains("proof_matches=n/a"))
        .stdout(predicate::str::contains("training_steps=n/a"))
        .stdout(predicate::str::contains("source_model_json: checked=true"));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-model-inference")
        .arg(&imported_model_bundle)
        .arg("--sample-audit")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .arg("--json")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "\"kind\": \"reverie_mnist_linear_q31_model_inference_audit\"",
        ))
        .stdout(predicate::str::contains("\"prediction\": 0"))
        .stdout(predicate::str::contains("\"correct\": true"));

    let mut changed_imported_model_source_json = imported_model_source_json.clone();
    changed_imported_model_source_json["model"]["bias"][0] = json!(123_i64);
    std::fs::write(
        &imported_model_source,
        serde_json::to_string_pretty(&changed_imported_model_source_json)
            .expect("changed imported model source serializes"),
    )
    .expect("changed imported model source can be written");
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-model")
        .arg(&imported_model_bundle)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "imported model source file hash does not match model bundle provenance",
        ));

    let model_bad_provenance = artifact_dir.join("self-test-model-bundle-bad-provenance.json");
    let mut model_bad_provenance_json = model_bundle_json.clone();
    model_bad_provenance_json["provenance"]["proof"]["full_replay_payload_bytes"] =
        serde_json::Value::from(1_u64);
    resign_mnist_model_bundle(&workspace_root(), &mut model_bad_provenance_json);
    std::fs::write(
        &model_bad_provenance,
        serde_json::to_string_pretty(&model_bad_provenance_json)
            .expect("bad provenance model bundle serializes"),
    )
    .expect("bad provenance model bundle can be written");

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-model")
        .arg(&model_bad_provenance)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "model bundle provenance proof does not match training steps",
        ));

    let tampered_model = artifact_dir.join("self-test-model-bundle-tampered.json");
    let mut tampered_model_json = model_bundle_json.clone();
    tampered_model_json["model"]["bias"][0] = serde_json::Value::from(123_i64);
    std::fs::write(
        &tampered_model,
        serde_json::to_string_pretty(&tampered_model_json)
            .expect("tampered model bundle serializes"),
    )
    .expect("tampered model bundle can be written");

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-model")
        .arg(&tampered_model)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "model bundle fingerprint mismatch",
        ));

    let inference = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-inference")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&inference.get_output().stdout)
        .expect("inference JSON report is UTF-8");
    let inference: serde_json::Value =
        serde_json::from_str(stdout).expect("inference inspect emits valid JSON");
    assert_eq!(
        inference["kind"].as_str(),
        Some("reverie_mnist_linear_q31_inference_audit")
    );
    assert_eq!(
        inference["fingerprints"]["payload"].as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert_eq!(
        inference["fingerprints"]["computation"].as_str(),
        Some(computation_fingerprint.as_str())
    );
    assert_eq!(inference["audit_step"].as_u64(), Some(0));
    assert_eq!(inference["sample_index"].as_u64(), Some(0));
    assert_eq!(inference["label"].as_i64(), Some(0));
    assert_eq!(inference["prediction"].as_i64(), Some(0));
    assert_eq!(inference["correct"].as_bool(), Some(true));
    assert_eq!(
        inference["witness_checks"]["prediction_matches_logits"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference["witness_checks"]["correct_matches_logits"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference["inverse"]["restored_initial_state"].as_bool(),
        Some(true)
    );
    assert_eq!(inference["active_pixels"].as_array().map(Vec::len), Some(2));
    assert_eq!(inference["top_logits"][0]["digit"].as_u64(), Some(0));
    assert_eq!(
        inference["attribution"]["predicted_digit"].as_u64(),
        Some(0)
    );
    assert_eq!(
        inference["attribution"]["runner_up_digit"].as_u64(),
        Some(1)
    );
    assert_eq!(
        inference["attribution"]["predicted_logit"].as_i64(),
        Some(2_127_803_771)
    );
    assert_eq!(
        inference["attribution"]["runner_up_logit"].as_i64(),
        Some(25_713_192)
    );
    assert_eq!(
        inference["attribution"]["margin"].as_i64(),
        Some(2_102_090_579)
    );
    assert_eq!(inference["attribution"]["bias"].as_i64(), Some(180_827_374));
    assert_eq!(
        inference["attribution"]["runner_up_bias"].as_i64(),
        Some(187_255_672)
    );
    assert_eq!(
        inference["attribution"]["margin_bias"].as_i64(),
        Some(-6_428_298)
    );
    assert_eq!(
        inference["attribution"]["contribution_sum"].as_i64(),
        Some(1_946_976_397)
    );
    assert_eq!(
        inference["attribution"]["margin_contribution_sum"].as_i64(),
        Some(2_108_518_877)
    );
    assert_eq!(
        inference["attribution"]["reconstructed_logit"].as_i64(),
        Some(2_127_803_771)
    );
    assert_eq!(
        inference["attribution"]["reconstructed_margin"].as_i64(),
        Some(2_102_090_579)
    );
    assert_eq!(
        inference["attribution"]["matches_logit"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference["attribution"]["matches_margin"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference["attribution"]["top_contributions"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(
        inference["attribution"]["top_contributions"][0]["pixel"].as_u64(),
        Some(0)
    );
    assert_eq!(
        inference["attribution"]["top_contributions"][0]["weight"].as_i64(),
        Some(1_555_136_903)
    );
    assert_eq!(
        inference["attribution"]["top_contributions"][0]["contribution"].as_i64(),
        Some(1_555_136_903)
    );
    assert_eq!(
        inference["attribution"]["top_contributions"][1]["pixel"].as_u64(),
        Some(10)
    );
    assert_eq!(
        inference["attribution"]["top_contributions"][1]["weight"].as_i64(),
        Some(780_617_743)
    );
    assert_eq!(
        inference["attribution"]["top_contributions"][1]["contribution"].as_i64(),
        Some(391_839_494)
    );
    assert_eq!(
        inference["attribution"]["top_margin_contributions"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(
        inference["attribution"]["top_margin_contributions"][0]["pixel"].as_u64(),
        Some(0)
    );
    assert_eq!(
        inference["attribution"]["top_margin_contributions"][0]["predicted_weight"].as_i64(),
        Some(1_555_136_903)
    );
    assert_eq!(
        inference["attribution"]["top_margin_contributions"][0]["runner_up_weight"].as_i64(),
        Some(-129_031_187)
    );
    assert_eq!(
        inference["attribution"]["top_margin_contributions"][0]["weight_delta"].as_i64(),
        Some(1_684_168_090)
    );
    assert_eq!(
        inference["attribution"]["top_margin_contributions"][0]["contribution"].as_i64(),
        Some(1_684_168_090)
    );
    assert_eq!(
        inference["attribution"]["top_margin_contributions"][1]["pixel"].as_u64(),
        Some(10)
    );
    assert_eq!(
        inference["attribution"]["top_margin_contributions"][1]["predicted_weight"].as_i64(),
        Some(780_617_743)
    );
    assert_eq!(
        inference["attribution"]["top_margin_contributions"][1]["runner_up_weight"].as_i64(),
        Some(-64_768_591)
    );
    assert_eq!(
        inference["attribution"]["top_margin_contributions"][1]["weight_delta"].as_i64(),
        Some(845_386_334)
    );
    assert_eq!(
        inference["attribution"]["top_margin_contributions"][1]["contribution"].as_i64(),
        Some(424_350_787)
    );

    let direct_inference_markdown = artifact_dir.join("self-test-direct-inference-audit.md");
    let _ = std::fs::remove_file(&direct_inference_markdown);
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-inference")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .arg("--markdown-output")
        .arg(&direct_inference_markdown)
        .assert()
        .success()
        .stdout(predicate::str::contains("inference audit step 0"))
        .stdout(predicate::str::contains(
            "witness_checks: computed_prediction=0 prediction_matches_logits=true correct_matches_logits=true",
        ))
        .stdout(predicate::str::contains(
            "attribution: predicted=0 runner_up=1 margin=2102090579",
        ))
        .stdout(predicate::str::contains("matches_margin=true"))
        .stdout(predicate::str::contains(
            "top_contributions=[p0 u8=255 w=1555136903 c=1555136903",
        ))
        .stdout(predicate::str::contains(
            "top_margin_contributions=[p0 u8=255 dw=1684168090 c=1684168090",
        ))
        .stdout(predicate::str::contains(
            "inverse: restored_initial_state=true",
        ));
    let direct_inference_markdown = std::fs::read_to_string(&direct_inference_markdown)
        .expect("direct inference Markdown is written");
    assert!(direct_inference_markdown.contains("# Reverie Inference Audit"));
    assert!(direct_inference_markdown.contains("## Top Logits"));
    assert!(direct_inference_markdown.contains("## Top Contributions"));
    assert!(direct_inference_markdown.contains("## Replay Proof"));
    assert!(direct_inference_markdown.contains("`deterministic_q31_inference_replay`"));
    assert!(direct_inference_markdown.contains("| training audit |"));

    let model_inference_markdown = artifact_dir.join("self-test-model-inference-audit.md");
    let _ = std::fs::remove_file(&model_inference_markdown);
    let model_inference = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-model-inference")
        .arg(&model_bundle)
        .arg("--sample-audit")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .arg("--markdown-output")
        .arg(&model_inference_markdown)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&model_inference.get_output().stdout)
        .expect("model inference JSON report is UTF-8");
    let model_inference: serde_json::Value =
        serde_json::from_str(stdout).expect("model inference emits valid JSON");
    assert_eq!(
        model_inference["kind"].as_str(),
        Some("reverie_mnist_linear_q31_model_inference_audit")
    );
    assert_eq!(
        model_inference["model_bundle"]["fingerprints"]["payload"].as_str(),
        Some(model_payload_fingerprint.as_str())
    );
    assert_eq!(
        model_inference["sample_audit"]["fingerprints"]["payload"].as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert_eq!(model_inference["audit_step"].as_u64(), Some(0));
    assert_eq!(model_inference["sample_index"].as_u64(), Some(0));
    assert_eq!(model_inference["prediction"].as_i64(), Some(0));
    assert_eq!(model_inference["correct"].as_bool(), Some(true));
    assert_eq!(
        model_inference["attribution"]["reconstructed_margin"].as_i64(),
        Some(2_102_090_579)
    );
    assert_eq!(
        model_inference["witness_checks"]["prediction_matches_logits"].as_bool(),
        Some(true)
    );
    assert_eq!(
        model_inference["inverse"]["restored_initial_state"].as_bool(),
        Some(true)
    );
    let model_inference_markdown = std::fs::read_to_string(&model_inference_markdown)
        .expect("model inference Markdown is written");
    assert!(model_inference_markdown.contains("# Reverie Inference Audit"));
    assert!(model_inference_markdown.contains("| model bundle |"));
    assert!(model_inference_markdown.contains("| sample source `audit_bundle` |"));
    assert!(model_inference_markdown.contains("## Contract Checks"));

    let standalone_rev = artifact_dir.join("self-test-model-inference.rev");
    let _ = std::fs::remove_file(&standalone_rev);
    let standalone_report = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-model-inference")
        .arg(&model_bundle)
        .arg("--sample-audit")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .arg("--standalone-rev-output")
        .arg(&standalone_rev)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&standalone_report.get_output().stdout)
        .expect("standalone model inference JSON report is UTF-8");
    let standalone_report: serde_json::Value =
        serde_json::from_str(stdout).expect("standalone model inference emits valid JSON");
    assert_eq!(
        standalone_report["standalone_rev_output"].as_str(),
        Some(standalone_rev.display().to_string().as_str())
    );
    let standalone_source =
        std::fs::read_to_string(&standalone_rev).expect("standalone .rev is written");
    assert!(standalone_source.contains("global image: tensor<int, 784>"));
    assert!(standalone_source.contains("global weights: tensor<int, 784, 10>"));
    assert!(standalone_source.contains("logits += vecmat_q31(image, weights);"));
    assert!(standalone_source.contains("assert prediction == expected_prediction;"));
    assert!(standalone_source.contains("assert correct == expected_correct"));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("check")
        .arg(&standalone_rev)
        .assert()
        .success();

    let standalone_run = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&standalone_rev)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&standalone_run.get_output().stdout)
        .expect("standalone run JSON report is UTF-8");
    let standalone_run: serde_json::Value =
        serde_json::from_str(stdout).expect("standalone run emits valid JSON");
    assert_eq!(standalone_run["store"]["prediction"].as_i64(), Some(0));
    assert_eq!(standalone_run["store"]["correct"].as_i64(), Some(1));
    assert_eq!(
        standalone_run["ml_profile"]["builtin_counts"]["vecmat_q31"].as_u64(),
        Some(1)
    );

    let standalone_roundtrip = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("roundtrip")
        .arg(&standalone_rev)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&standalone_roundtrip.get_output().stdout)
        .expect("standalone roundtrip JSON report is UTF-8");
    let standalone_roundtrip: serde_json::Value =
        serde_json::from_str(stdout).expect("standalone roundtrip emits valid JSON");
    assert_eq!(standalone_roundtrip["passed"].as_bool(), Some(true));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-model-inference")
        .arg(&model_bundle)
        .arg("--sample-audit")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .assert()
        .success()
        .stdout(predicate::str::contains("model inference audit step 0"))
        .stdout(predicate::str::contains("model_fingerprint:"))
        .stdout(predicate::str::contains("sample_source: audit path="))
        .stdout(predicate::str::contains("prediction=0 correct=true"))
        .stdout(predicate::str::contains(
            "inverse: restored_initial_state=true",
        ));

    let model_inference_bundle = artifact_dir.join("self-test-model-inference-bundle.json");
    let _ = std::fs::remove_file(&model_inference_bundle);
    let model_inference_output = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-model-inference")
        .arg(&model_bundle)
        .arg("--sample-audit")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .arg("--inference-output")
        .arg(&model_inference_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&model_inference_output.get_output().stdout)
        .expect("model inference output JSON report is UTF-8");
    let model_inference_output: serde_json::Value =
        serde_json::from_str(stdout).expect("model inference output emits valid JSON");
    assert_eq!(
        model_inference_output["inference_output"].as_str(),
        Some(model_inference_bundle.display().to_string().as_str())
    );
    let model_inference_bundle_text = std::fs::read_to_string(&model_inference_bundle)
        .expect("model inference bundle is written");
    let model_inference_bundle_json: serde_json::Value =
        serde_json::from_str(&model_inference_bundle_text)
            .expect("model inference bundle is valid JSON");
    assert_eq!(
        model_inference_bundle_json["kind"].as_str(),
        Some("reverie_mnist_linear_q31_inference_replay_bundle")
    );
    assert_eq!(
        model_inference_bundle_json["source_model_bundle"]["fingerprints"]["payload"].as_str(),
        Some(model_payload_fingerprint.as_str())
    );
    assert_eq!(
        model_inference_bundle_json["source_sample_audit"]["fingerprints"]["payload"].as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert_eq!(
        model_inference_bundle_json["result"]["attribution"]["matches_margin"].as_bool(),
        Some(true)
    );

    let model_inference_verification = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&model_inference_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&model_inference_verification.get_output().stdout)
        .expect("model inference verification JSON report is UTF-8");
    let model_inference_verification: serde_json::Value =
        serde_json::from_str(stdout).expect("model inference verification emits valid JSON");
    assert_eq!(
        model_inference_verification["source_model_checked"].as_bool(),
        Some(true)
    );
    assert_eq!(
        model_inference_verification["source_sample_checked"].as_bool(),
        Some(true)
    );
    assert_eq!(
        model_inference_verification["source_model"]["payload_fingerprint"].as_str(),
        Some(model_payload_fingerprint.as_str())
    );
    assert_eq!(
        model_inference_verification["source_sample"]["payload_fingerprint"].as_str(),
        Some(payload_fingerprint.as_str())
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&model_inference_bundle)
        .assert()
        .success()
        .stdout(predicate::str::contains("inference audit ok"))
        .stdout(predicate::str::contains("source_model: checked=true"))
        .stdout(predicate::str::contains("source_sample: checked=true"));

    let bad_model_inference_source = artifact_dir.join("self-test-model-inference-bad-source.json");
    let mut bad_model_inference_source_json = model_inference_bundle_json.clone();
    bad_model_inference_source_json["source_sample_audit"]["audit_step"] = json!(1);
    bad_model_inference_source_json["source_sample_audit"]["sample_index"] = json!(1);
    resign_mnist_inference_bundle(&workspace_root(), &mut bad_model_inference_source_json);
    std::fs::write(
        &bad_model_inference_source,
        serde_json::to_string_pretty(&bad_model_inference_source_json)
            .expect("bad model inference source bundle serializes"),
    )
    .expect("bad model inference source bundle can be written");
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&bad_model_inference_source)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "inference source sample audit label expected",
        ));

    let sample_json = artifact_dir.join("self-test-sample.json");
    let sample_json_value = json!({
        "kind": "reverie_mnist_linear_q31_sample",
        "schema_version": 1,
        "image_u8": bundle_json["witness_trace"][0]["image_u8"].clone(),
        "label": 0,
    });
    let sample_json_fingerprint = sha256_json(&sample_json_value);
    std::fs::write(
        &sample_json,
        serde_json::to_string_pretty(&sample_json_value).expect("sample JSON serializes"),
    )
    .expect("sample JSON can be written");

    let json_sample_inference = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-model-inference")
        .arg(&model_bundle)
        .arg("--sample-json")
        .arg(&sample_json)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&json_sample_inference.get_output().stdout)
        .expect("JSON sample inference report is UTF-8");
    let json_sample_inference: serde_json::Value =
        serde_json::from_str(stdout).expect("JSON sample inference emits valid JSON");
    assert_eq!(
        json_sample_inference["kind"].as_str(),
        Some("reverie_mnist_linear_q31_model_inference_audit")
    );
    assert_eq!(
        json_sample_inference["sample_source"]["kind"].as_str(),
        Some("sample_json")
    );
    assert_eq!(
        json_sample_inference["sample_json"]["fingerprint"].as_str(),
        Some(sample_json_fingerprint.as_str())
    );
    assert!(json_sample_inference["sample_audit"].is_null());
    assert!(json_sample_inference["audit_step"].is_null());
    assert!(json_sample_inference["sample_index"].is_null());
    assert_eq!(json_sample_inference["label"].as_i64(), Some(0));
    assert_eq!(json_sample_inference["prediction"].as_i64(), Some(0));
    assert_eq!(
        json_sample_inference["inverse"]["restored_initial_state"].as_bool(),
        Some(true)
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-model-inference")
        .arg(&model_bundle)
        .arg("--sample-json")
        .arg(&sample_json)
        .assert()
        .success()
        .stdout(predicate::str::contains("sample_source: json path="))
        .stdout(predicate::str::contains("source_index=none label=0"))
        .stdout(predicate::str::contains(
            "inverse: restored_initial_state=true",
        ));

    let json_sample_inference_bundle =
        artifact_dir.join("self-test-json-sample-inference-bundle.json");
    let _ = std::fs::remove_file(&json_sample_inference_bundle);
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-model-inference")
        .arg(&model_bundle)
        .arg("--sample-json")
        .arg(&sample_json)
        .arg("--inference-output")
        .arg(&json_sample_inference_bundle)
        .assert()
        .success();
    let json_sample_inference_bundle_text = std::fs::read_to_string(&json_sample_inference_bundle)
        .expect("JSON sample inference bundle is written");
    let json_sample_inference_bundle_json: serde_json::Value =
        serde_json::from_str(&json_sample_inference_bundle_text)
            .expect("JSON sample inference bundle is valid JSON");
    assert_eq!(
        json_sample_inference_bundle_json["source_sample_json"]["fingerprint"].as_str(),
        Some(sample_json_fingerprint.as_str())
    );
    assert_eq!(
        json_sample_inference_bundle_json["sample_source"]["kind"].as_str(),
        Some("sample_json")
    );
    assert_eq!(
        json_sample_inference_bundle_json["result"]["prediction"].as_i64(),
        Some(0)
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&json_sample_inference_bundle)
        .arg("--json")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"source_model_checked\": true"))
        .stdout(predicate::str::contains("\"source_sample_checked\": true"));

    let samples_json = artifact_dir.join("self-test-samples.json");
    let _ = std::fs::remove_file(&samples_json);
    let samples_export = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--export-samples")
        .arg(&bundle)
        .arg("--samples-output")
        .arg(&samples_json)
        .arg("--samples-limit")
        .arg("2")
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&samples_export.get_output().stdout).expect("sample export is UTF-8");
    let samples_export: serde_json::Value =
        serde_json::from_str(stdout).expect("sample export emits valid JSON");
    assert_eq!(
        samples_export["kind"].as_str(),
        Some("reverie_mnist_linear_q31_sample_set_export")
    );
    assert_eq!(samples_export["samples"].as_u64(), Some(2));
    assert_eq!(
        samples_export["source_fingerprints"]["payload"].as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert!(samples_json.exists());

    let samples_json_value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&samples_json).expect("samples JSON can be read"))
            .expect("exported samples JSON is valid");
    let samples_json_fingerprint = sha256_json(&samples_json_value);
    assert_eq!(
        samples_json_value["kind"].as_str(),
        Some("reverie_mnist_linear_q31_samples")
    );
    assert_eq!(
        samples_json_value["source_audit_bundle"]["fingerprints"]["payload"].as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert_eq!(
        samples_json_value["samples"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(
        samples_json_value["samples"][0]["audit_step"].as_u64(),
        Some(0)
    );
    assert_eq!(
        samples_json_value["samples"][0]["source_sample_index"].as_u64(),
        Some(0)
    );
    assert_eq!(
        samples_json_value["samples"][1]["audit_step"].as_u64(),
        Some(1)
    );
    assert_eq!(
        samples_json_value["samples"][1]["source_sample_index"].as_u64(),
        Some(1)
    );
    assert_eq!(
        samples_json_value["proof"]["claim"].as_str(),
        Some("deterministic_q31_sample_set_export")
    );
    assert_eq!(
        samples_json_value["proof"]["source_steps"].as_u64(),
        Some(100)
    );
    assert_eq!(samples_json_value["proof"]["samples"].as_u64(), Some(2));
    assert_eq!(
        samples_json_value["proof"]["sample_payload_bytes"].as_u64(),
        Some(1_570)
    );
    assert_eq!(
        samples_json_value["proof"]["lineage"]["audit_steps_contiguous"].as_bool(),
        Some(true)
    );
    assert_eq!(
        samples_json_value["proof"]["lineage"]["source_sample_indices_present"].as_bool(),
        Some(true)
    );
    assert_eq!(
        samples_json_value["proof"]["fingerprints"]["source_audit_bundle"].as_str(),
        Some(sha256_json(&samples_json_value["source_audit_bundle"]).as_str())
    );
    assert_eq!(
        samples_json_value["proof"]["fingerprints"]["samples"].as_str(),
        Some(sha256_json(&samples_json_value["samples"]).as_str())
    );
    assert_eq!(
        samples_json_value["proof"]["fingerprints"]["report"].as_str(),
        Some(sha256_json(&samples_json_value["report"]).as_str())
    );
    assert_eq!(
        samples_json_value["fingerprints"]["proof"].as_str(),
        Some(sha256_json(&samples_json_value["proof"]).as_str())
    );
    let sample_zero_fingerprint = sha256_json(&samples_json_value["samples"][0]);
    let sample_one_fingerprint = sha256_json(&samples_json_value["samples"][1]);
    let samples_payload_fingerprint = samples_json_value["fingerprints"]["payload"]
        .as_str()
        .expect("sample set has payload fingerprint")
        .to_owned();

    let samples_verification = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-samples")
        .arg(&samples_json)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&samples_verification.get_output().stdout)
        .expect("sample verification is UTF-8");
    let samples_verification: serde_json::Value =
        serde_json::from_str(stdout).expect("sample verification emits valid JSON");
    assert_eq!(
        samples_verification["kind"].as_str(),
        Some("reverie_mnist_linear_q31_sample_set_verification")
    );
    assert_eq!(samples_verification["samples"].as_u64(), Some(2));
    assert_eq!(
        samples_verification["fingerprints"]["payload"].as_str(),
        Some(samples_payload_fingerprint.as_str())
    );
    assert_eq!(samples_verification["proof_matches"].as_bool(), Some(true));
    assert_eq!(
        samples_verification["proof"]["claim"].as_str(),
        Some("deterministic_q31_sample_set_export")
    );
    assert_eq!(
        samples_verification["proof"]["source_steps"].as_u64(),
        Some(100)
    );
    assert_eq!(
        samples_verification["proof"]["sample_payload_bytes"].as_u64(),
        Some(1_570)
    );
    assert_eq!(
        samples_verification["source_audit_checked"].as_bool(),
        Some(true)
    );
    assert_eq!(
        samples_verification["source_audit"]["checked"].as_bool(),
        Some(true)
    );
    assert_eq!(
        samples_verification["source_audit"]["source_steps"].as_u64(),
        Some(100)
    );
    assert_eq!(
        samples_verification["source_audit"]["samples_checked"].as_u64(),
        Some(2)
    );
    assert_eq!(
        samples_verification["source_audit"]["payload_fingerprint"].as_str(),
        Some(payload_fingerprint.as_str())
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-samples")
        .arg(&samples_json)
        .assert()
        .success()
        .stdout(predicate::str::contains("sample set ok"))
        .stdout(predicate::str::contains("samples=2"))
        .stdout(predicate::str::contains("proof_matches=true"))
        .stdout(predicate::str::contains("source_audit_checked=true"));

    let bad_sample_set_proof = artifact_dir.join("self-test-samples-bad-proof.json");
    let mut bad_sample_set_json = samples_json_value.clone();
    bad_sample_set_json["proof"]["sample_payload_bytes"] = json!(1);
    resign_mnist_sample_set_bundle(&mut bad_sample_set_json);
    std::fs::write(
        &bad_sample_set_proof,
        serde_json::to_string_pretty(&bad_sample_set_json)
            .expect("bad sample set bundle serializes"),
    )
    .expect("bad sample set proof bundle can be written");
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-samples")
        .arg(&bad_sample_set_proof)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "sample set proof does not match recomputed sample lineage",
        ));

    let bad_sample_set_source = artifact_dir.join("self-test-samples-bad-source.json");
    let mut bad_sample_set_source_json = samples_json_value.clone();
    let original_pixel = bad_sample_set_source_json["samples"][0]["image_u8"][0]
        .as_u64()
        .expect("sample pixel is numeric");
    bad_sample_set_source_json["samples"][0]["image_u8"][0] =
        json!(if original_pixel == 0 { 1 } else { 0 });
    let updated_samples_fingerprint = sha256_json(&bad_sample_set_source_json["samples"]);
    bad_sample_set_source_json["proof"]["fingerprints"]["samples"] =
        json!(updated_samples_fingerprint);
    resign_mnist_sample_set_bundle(&mut bad_sample_set_source_json);
    std::fs::write(
        &bad_sample_set_source,
        serde_json::to_string_pretty(&bad_sample_set_source_json)
            .expect("bad source sample set bundle serializes"),
    )
    .expect("bad source sample set bundle can be written");
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-samples")
        .arg(&bad_sample_set_source)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "sample set source audit sample 0 image does not match referenced audit_step 0",
        ));

    let model_evaluation = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--evaluate-model")
        .arg(&model_bundle)
        .arg("--samples-json")
        .arg(&samples_json)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&model_evaluation.get_output().stdout)
        .expect("model evaluation JSON report is UTF-8");
    let model_evaluation: serde_json::Value =
        serde_json::from_str(stdout).expect("model evaluation emits valid JSON");
    assert_eq!(
        model_evaluation["kind"].as_str(),
        Some("reverie_mnist_linear_q31_model_evaluation")
    );
    assert_eq!(
        model_evaluation["model_bundle"]["fingerprints"]["payload"].as_str(),
        Some(model_payload_fingerprint.as_str())
    );
    assert_eq!(
        model_evaluation["samples_json"]["fingerprint"].as_str(),
        Some(samples_json_fingerprint.as_str())
    );
    assert_eq!(model_evaluation["summary"]["samples"].as_u64(), Some(2));
    assert_eq!(model_evaluation["summary"]["correct"].as_u64(), Some(2));
    assert_eq!(model_evaluation["summary"]["incorrect"].as_u64(), Some(0));
    assert_eq!(
        model_evaluation["proof"]["model_payload_bytes"].as_u64(),
        Some(62_800)
    );
    assert_eq!(
        model_evaluation["proof"]["sample_payload_bytes"].as_u64(),
        Some(1_570)
    );
    assert_eq!(
        model_evaluation["proof"]["witness_payload_bytes"].as_u64(),
        Some(192)
    );
    assert_eq!(
        model_evaluation["proof"]["replay_payload_bytes"].as_u64(),
        Some(64_562)
    );
    assert_eq!(
        model_evaluation["proof"]["forward_recompute_steps"].as_u64(),
        Some(2)
    );
    let evaluation_rows = model_evaluation["rows"]
        .as_array()
        .expect("model evaluation has rows");
    assert_eq!(evaluation_rows.len(), 2);
    assert_eq!(evaluation_rows[0]["audit_step"].as_u64(), Some(0));
    assert_eq!(evaluation_rows[0]["source_sample_index"].as_u64(), Some(0));
    assert_eq!(
        evaluation_rows[0]["sample_fingerprint"].as_str(),
        Some(sample_zero_fingerprint.as_str())
    );
    assert_eq!(evaluation_rows[0]["label"].as_i64(), Some(0));
    assert_eq!(evaluation_rows[0]["prediction"].as_i64(), Some(0));
    assert_eq!(evaluation_rows[0]["correct"].as_bool(), Some(true));
    assert!(evaluation_rows[0]["margin"].is_i64());
    assert_eq!(evaluation_rows[1]["audit_step"].as_u64(), Some(1));
    assert_eq!(evaluation_rows[1]["source_sample_index"].as_u64(), Some(1));
    assert_eq!(
        evaluation_rows[1]["sample_fingerprint"].as_str(),
        Some(sample_one_fingerprint.as_str())
    );
    assert_eq!(evaluation_rows[1]["label"].as_i64(), Some(1));
    assert_eq!(evaluation_rows[1]["prediction"].as_i64(), Some(1));
    assert_eq!(evaluation_rows[1]["correct"].as_bool(), Some(true));

    let gated_model_evaluation = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--evaluate-model")
        .arg(&model_bundle)
        .arg("--samples-json")
        .arg(&samples_json)
        .arg("--eval-min-accuracy")
        .arg("100")
        .arg("--eval-min-margin")
        .arg("0")
        .arg("--eval-max-incorrect")
        .arg("0")
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&gated_model_evaluation.get_output().stdout)
        .expect("gated model evaluation JSON report is UTF-8");
    let gated_model_evaluation: serde_json::Value =
        serde_json::from_str(stdout).expect("gated model evaluation emits valid JSON");
    assert_eq!(
        gated_model_evaluation["gate"]["passed"].as_bool(),
        Some(true)
    );
    assert_eq!(
        gated_model_evaluation["gate"]["checks"]
            .as_array()
            .map(Vec::len),
        Some(3)
    );
    assert_eq!(
        gated_model_evaluation["gate"]["checks"][0]["metric"].as_str(),
        Some("accuracy_percent")
    );

    let evaluation_bundle = artifact_dir.join("self-test-evaluation-bundle.json");
    let _ = std::fs::remove_file(&evaluation_bundle);
    let bundled_model_evaluation = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--evaluate-model")
        .arg(&model_bundle)
        .arg("--samples-json")
        .arg(&samples_json)
        .arg("--eval-min-accuracy")
        .arg("100")
        .arg("--eval-min-margin")
        .arg("0")
        .arg("--eval-max-incorrect")
        .arg("0")
        .arg("--evaluation-output")
        .arg(&evaluation_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&bundled_model_evaluation.get_output().stdout)
        .expect("bundled model evaluation JSON report is UTF-8");
    let bundled_model_evaluation: serde_json::Value =
        serde_json::from_str(stdout).expect("bundled model evaluation emits valid JSON");
    assert_eq!(
        bundled_model_evaluation["evaluation_output"].as_str(),
        Some(evaluation_bundle.to_string_lossy().as_ref())
    );
    assert_eq!(
        bundled_model_evaluation["model_bundle"]["fingerprints"]["payload"].as_str(),
        Some(model_payload_fingerprint.as_str())
    );
    assert!(bundled_model_evaluation["fingerprints"]["payload"].is_string());
    assert_eq!(
        bundled_model_evaluation["gate"]["passed"].as_bool(),
        Some(true)
    );
    assert!(evaluation_bundle.exists());

    let evaluation_bundle_json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&evaluation_bundle).expect("evaluation bundle can be read"),
    )
    .expect("evaluation bundle is valid JSON");
    assert_eq!(
        evaluation_bundle_json["kind"].as_str(),
        Some("reverie_mnist_linear_q31_model_evaluation_bundle")
    );
    assert_eq!(
        evaluation_bundle_json["source_model_bundle"]["fingerprints"]["payload"].as_str(),
        Some(model_payload_fingerprint.as_str())
    );
    assert_eq!(
        evaluation_bundle_json["source_samples_json"]["fingerprint"].as_str(),
        Some(samples_json_fingerprint.as_str())
    );
    assert_eq!(
        evaluation_bundle_json["summary"]["samples"].as_u64(),
        Some(2)
    );
    assert_eq!(
        evaluation_bundle_json["samples"]["samples"][0]["audit_step"].as_u64(),
        Some(0)
    );
    assert_eq!(
        evaluation_bundle_json["samples"]["samples"][0]["source_sample_index"].as_u64(),
        Some(0)
    );
    assert_eq!(
        evaluation_bundle_json["rows"][0]["sample_fingerprint"].as_str(),
        Some(sample_zero_fingerprint.as_str())
    );
    assert_eq!(
        evaluation_bundle_json["rows"][0]["audit_step"].as_u64(),
        Some(0)
    );
    assert_eq!(
        evaluation_bundle_json["gate_policy"]["min_accuracy"].as_f64(),
        Some(100.0)
    );
    assert_eq!(
        evaluation_bundle_json["gate"]["passed"].as_bool(),
        Some(true)
    );
    let evaluation_payload_fingerprint = evaluation_bundle_json["fingerprints"]["payload"]
        .as_str()
        .expect("evaluation bundle has payload fingerprint")
        .to_owned();

    let evaluation_verification = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-evaluation")
        .arg(&evaluation_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&evaluation_verification.get_output().stdout)
        .expect("evaluation verification JSON report is UTF-8");
    let evaluation_verification: serde_json::Value =
        serde_json::from_str(stdout).expect("evaluation verification emits valid JSON");
    assert_eq!(
        evaluation_verification["kind"].as_str(),
        Some("reverie_mnist_linear_q31_model_evaluation_verification")
    );
    assert_eq!(
        evaluation_verification["fingerprints"]["payload"].as_str(),
        Some(evaluation_payload_fingerprint.as_str())
    );
    assert_eq!(evaluation_verification["rows_match"].as_bool(), Some(true));
    assert_eq!(
        evaluation_verification["proof_matches"].as_bool(),
        Some(true)
    );
    assert_eq!(
        evaluation_verification["source_model_checked"].as_bool(),
        Some(true)
    );
    assert_eq!(
        evaluation_verification["source_samples_checked"].as_bool(),
        Some(true)
    );
    assert_eq!(
        evaluation_verification["source_model"]["payload_fingerprint"].as_str(),
        Some(model_payload_fingerprint.as_str())
    );
    assert_eq!(
        evaluation_verification["source_model"]["items_checked"].as_u64(),
        Some(1)
    );
    assert_eq!(
        evaluation_verification["source_samples_json"]["payload_fingerprint"].as_str(),
        Some(samples_json_fingerprint.as_str())
    );
    assert_eq!(
        evaluation_verification["source_samples_json"]["items_checked"].as_u64(),
        Some(2)
    );
    assert_eq!(
        evaluation_verification["gate"]["passed"].as_bool(),
        Some(true)
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-evaluation")
        .arg(&evaluation_bundle)
        .assert()
        .success()
        .stdout(predicate::str::contains("model evaluation ok"))
        .stdout(predicate::str::contains("rows_match=true"))
        .stdout(predicate::str::contains("source_model_checked=true"))
        .stdout(predicate::str::contains("source_samples_checked=true"))
        .stdout(predicate::str::contains("gate: passed"));

    let bad_evaluation_source = artifact_dir.join("self-test-evaluation-bad-source-samples.json");
    let mut bad_evaluation_source_json = evaluation_bundle_json.clone();
    let original_pixel = bad_evaluation_source_json["samples"]["samples"][0]["image_u8"][0]
        .as_u64()
        .expect("evaluation sample pixel is numeric");
    bad_evaluation_source_json["samples"]["samples"][0]["image_u8"][0] =
        json!(if original_pixel == 0 { 1 } else { 0 });
    resign_mnist_evaluation_bundle(&workspace_root(), &mut bad_evaluation_source_json);
    std::fs::write(
        &bad_evaluation_source,
        serde_json::to_string_pretty(&bad_evaluation_source_json)
            .expect("bad evaluation source bundle serializes"),
    )
    .expect("bad evaluation source bundle can be written");
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-evaluation")
        .arg(&bad_evaluation_source)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "model evaluation source samples sample 0 does not match embedded sample",
        ));

    let evaluation_scan = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--scan-evaluation")
        .arg(&evaluation_bundle)
        .arg("--evaluation-limit")
        .arg("1")
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&evaluation_scan.get_output().stdout)
        .expect("evaluation scan JSON report is UTF-8");
    let evaluation_scan: serde_json::Value =
        serde_json::from_str(stdout).expect("evaluation scan emits valid JSON");
    assert_eq!(
        evaluation_scan["kind"].as_str(),
        Some("reverie_mnist_linear_q31_model_evaluation_scan")
    );
    assert_eq!(evaluation_scan["summary"]["samples"].as_u64(), Some(2));
    assert_eq!(evaluation_scan["summary"]["correct"].as_u64(), Some(2));
    assert_eq!(
        evaluation_scan["top_incorrect"].as_array().map(Vec::len),
        Some(0)
    );
    assert_eq!(
        evaluation_scan["top_low_margin"].as_array().map(Vec::len),
        Some(1)
    );
    let low_margin_row = &evaluation_scan["top_low_margin"][0];
    assert!(low_margin_row["audit_step"].as_u64().is_some());
    assert!(low_margin_row["source_sample_index"].as_u64().is_some());
    assert_eq!(
        low_margin_row["sample_fingerprint"].as_str().map(str::len),
        Some(64)
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--scan-evaluation")
        .arg(&evaluation_bundle)
        .arg("--evaluation-limit")
        .arg("1")
        .assert()
        .success()
        .stdout(predicate::str::contains("model evaluation scan"))
        .stdout(predicate::str::contains("top_low_margin"))
        .stdout(predicate::str::contains("source_sample_index="))
        .stdout(predicate::str::contains("sample="));

    let evaluation_row_markdown = artifact_dir.join("self-test-evaluation-row-inference-audit.md");
    let _ = std::fs::remove_file(&evaluation_row_markdown);
    let evaluation_row_inspection = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-evaluation")
        .arg(&evaluation_bundle)
        .arg("--evaluation-row")
        .arg("0")
        .arg("--markdown-output")
        .arg(&evaluation_row_markdown)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&evaluation_row_inspection.get_output().stdout)
        .expect("evaluation row inspection JSON report is UTF-8");
    let evaluation_row_inspection: serde_json::Value =
        serde_json::from_str(stdout).expect("evaluation row inspection emits valid JSON");
    assert_eq!(
        evaluation_row_inspection["kind"].as_str(),
        Some("reverie_mnist_linear_q31_model_evaluation_row")
    );
    assert_eq!(
        evaluation_row_inspection["evaluation_bundle"]["fingerprints"]["payload"].as_str(),
        Some(evaluation_payload_fingerprint.as_str())
    );
    assert_eq!(
        evaluation_row_inspection["model_bundle"]["fingerprints"]["payload"].as_str(),
        Some(model_payload_fingerprint.as_str())
    );
    assert_eq!(evaluation_row_inspection["row"]["index"].as_u64(), Some(0));
    assert_eq!(
        evaluation_row_inspection["row"]["audit_step"].as_u64(),
        Some(0)
    );
    assert_eq!(
        evaluation_row_inspection["row"]["source_sample_index"].as_u64(),
        Some(0)
    );
    assert_eq!(
        evaluation_row_inspection["row"]["sample_fingerprint"].as_str(),
        Some(sample_zero_fingerprint.as_str())
    );
    assert_eq!(
        evaluation_row_inspection["row_matches_recomputed_inference"].as_bool(),
        Some(true)
    );
    assert_eq!(evaluation_row_inspection["prediction"].as_i64(), Some(0));
    assert_eq!(evaluation_row_inspection["correct"].as_bool(), Some(true));
    assert_eq!(
        evaluation_row_inspection["sample_source"]["kind"].as_str(),
        Some("model_evaluation_bundle")
    );
    assert_eq!(
        evaluation_row_inspection["sample_source"]["row_index"].as_u64(),
        Some(0)
    );
    assert_eq!(
        evaluation_row_inspection["sample_source"]["sample_fingerprint"].as_str(),
        Some(sample_zero_fingerprint.as_str())
    );
    let evaluation_row_markdown = std::fs::read_to_string(&evaluation_row_markdown)
        .expect("evaluation row inference Markdown is written");
    assert!(evaluation_row_markdown.contains("# Reverie Inference Audit"));
    assert!(evaluation_row_markdown.contains("| evaluation bundle |"));
    assert!(evaluation_row_markdown.contains("| sample source `model_evaluation_bundle` |"));
    assert!(evaluation_row_markdown.contains("## Top Margin Contributions"));

    let evaluation_row_inference_bundle =
        artifact_dir.join("self-test-evaluation-row-inference-bundle.json");
    let _ = std::fs::remove_file(&evaluation_row_inference_bundle);
    let evaluation_row_replay = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-evaluation")
        .arg(&evaluation_bundle)
        .arg("--evaluation-row")
        .arg("0")
        .arg("--inference-output")
        .arg(&evaluation_row_inference_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&evaluation_row_replay.get_output().stdout)
        .expect("evaluation row replay JSON report is UTF-8");
    let evaluation_row_replay: serde_json::Value =
        serde_json::from_str(stdout).expect("evaluation row replay emits valid JSON");
    assert_eq!(
        evaluation_row_replay["inference_output"].as_str(),
        Some(evaluation_row_inference_bundle.to_string_lossy().as_ref())
    );
    assert!(evaluation_row_inference_bundle.exists());
    let evaluation_row_inference_json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&evaluation_row_inference_bundle)
            .expect("evaluation row inference bundle can be read"),
    )
    .expect("evaluation row inference bundle is valid JSON");
    assert_eq!(
        evaluation_row_inference_json["source_model_evaluation"]["fingerprints"]["payload"]
            .as_str(),
        Some(evaluation_payload_fingerprint.as_str())
    );
    assert_eq!(
        evaluation_row_inference_json["source_model_evaluation"]["row_index"].as_u64(),
        Some(0)
    );
    assert_eq!(
        evaluation_row_inference_json["source_model_evaluation"]["audit_step"].as_u64(),
        Some(0)
    );
    assert_eq!(
        evaluation_row_inference_json["source_model_evaluation"]["source_sample_index"].as_u64(),
        Some(0)
    );
    assert_eq!(
        evaluation_row_inference_json["source_model_evaluation"]["sample_fingerprint"].as_str(),
        Some(sample_zero_fingerprint.as_str())
    );
    assert_eq!(
        evaluation_row_inference_json["sample_source"]["kind"].as_str(),
        Some("model_evaluation_bundle")
    );

    let evaluation_row_inference_verification = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&evaluation_row_inference_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&evaluation_row_inference_verification.get_output().stdout)
        .expect("evaluation row inference verification JSON report is UTF-8");
    let evaluation_row_inference_verification: serde_json::Value = serde_json::from_str(stdout)
        .expect("evaluation row inference verification emits valid JSON");
    assert_eq!(
        evaluation_row_inference_verification["source_evaluation_checked"].as_bool(),
        Some(true)
    );
    assert_eq!(
        evaluation_row_inference_verification["source_model_evaluation"]["payload_fingerprint"]
            .as_str(),
        Some(evaluation_payload_fingerprint.as_str())
    );
    assert_eq!(
        evaluation_row_inference_verification["source_model_evaluation"]["row_index"].as_u64(),
        Some(0)
    );
    assert_eq!(
        evaluation_row_inference_verification["source_model_evaluation"]["sample_fingerprint"]
            .as_str(),
        Some(sample_zero_fingerprint.as_str())
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&evaluation_row_inference_bundle)
        .assert()
        .success()
        .stdout(predicate::str::contains("inference audit ok"))
        .stdout(predicate::str::contains(
            "source_model_evaluation: checked=true",
        ));

    let bad_evaluation_row_inference_bundle =
        artifact_dir.join("self-test-evaluation-row-inference-bad-source.json");
    let mut bad_evaluation_row_inference_json = evaluation_row_inference_json.clone();
    bad_evaluation_row_inference_json["source_model_evaluation"]["row_index"] = json!(1);
    resign_mnist_inference_bundle(&workspace_root(), &mut bad_evaluation_row_inference_json);
    std::fs::write(
        &bad_evaluation_row_inference_bundle,
        serde_json::to_string_pretty(&bad_evaluation_row_inference_json)
            .expect("bad evaluation row inference bundle serializes"),
    )
    .expect("bad evaluation row inference bundle can be written");
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&bad_evaluation_row_inference_bundle)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "inference source evaluation sample does not match inference sample",
        ));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-evaluation")
        .arg(&evaluation_bundle)
        .arg("--evaluation-row")
        .arg("0")
        .assert()
        .success()
        .stdout(predicate::str::contains("model evaluation row 0"))
        .stdout(predicate::str::contains(
            "row_matches_recomputed_inference=true",
        ))
        .stdout(predicate::str::contains("audit_step=0"))
        .stdout(predicate::str::contains("source_sample_index=0"));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-evaluation")
        .arg(&evaluation_bundle)
        .arg("--evaluation-row")
        .arg("99")
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--evaluation-row 99 is out of range",
        ));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--evaluate-model")
        .arg(&model_bundle)
        .arg("--samples-json")
        .arg(&samples_json)
        .assert()
        .success()
        .stdout(predicate::str::contains("model evaluation"))
        .stdout(predicate::str::contains(
            "summary: samples=2 correct=2 incorrect=0",
        ))
        .stdout(predicate::str::contains("accuracy=100.00%"))
        .stdout(predicate::str::contains(
            "proof: model_bytes=62800 sample_bytes=1570 witness_bytes=192",
        ));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--evaluate-model")
        .arg(&model_bundle)
        .arg("--samples-json")
        .arg(&samples_json)
        .arg("--eval-min-accuracy")
        .arg("100")
        .arg("--eval-require-label-coverage")
        .assert()
        .failure()
        .stdout(predicate::str::contains("gate: failed"))
        .stdout(predicate::str::contains(
            "label_coverage actual=2/10 requirement=\"10/10\" passed=false",
        ))
        .stderr(predicate::str::contains("model evaluation gate failed"));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--samples-json")
        .arg(&samples_json)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--samples-json requires --evaluate-model",
        ));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--samples-output")
        .arg(&samples_json)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--samples-output requires --export-samples",
        ));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--samples-limit")
        .arg("2")
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--samples-limit requires --export-samples",
        ));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--export-samples")
        .arg(&bundle)
        .arg("--samples-limit")
        .arg("0")
        .arg("--samples-output")
        .arg(artifact_dir.join("unused-samples.json"))
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--samples-limit must be greater than zero",
        ));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--eval-min-accuracy")
        .arg("100")
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "evaluation gate thresholds require --evaluate-model",
        ));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--evaluation-output")
        .arg(&evaluation_bundle)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--evaluation-output requires --evaluate-model",
        ));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-model-inference")
        .arg(&model_bundle)
        .arg("--sample-audit")
        .arg(&bundle)
        .arg("--sample-json")
        .arg(&sample_json)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "choose only one of --sample-audit or --sample-json",
        ));

    let inference_bundle = artifact_dir.join("self-test-inference-bundle.json");
    let _ = std::fs::remove_file(&inference_bundle);
    let inference_output = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inspect-inference")
        .arg(&bundle)
        .arg("--audit-step")
        .arg("0")
        .arg("--inference-output")
        .arg(&inference_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&inference_output.get_output().stdout)
        .expect("inference output JSON report is UTF-8");
    let inference_output: serde_json::Value =
        serde_json::from_str(stdout).expect("inference output emits valid JSON");
    assert_eq!(
        inference_output["inference_output"].as_str(),
        Some(inference_bundle.display().to_string().as_str())
    );
    assert_eq!(
        inference_output["memory"]["model_payload_bytes"].as_u64(),
        Some(62_800)
    );
    assert_eq!(
        inference_output["memory"]["sample_payload_bytes"].as_u64(),
        Some(785)
    );
    assert_eq!(
        inference_output["memory"]["witness_payload_bytes"].as_u64(),
        Some(96)
    );
    assert_eq!(
        inference_output["memory"]["trace_payload_bytes"].as_u64(),
        Some(0)
    );
    assert_eq!(
        inference_output["memory"]["replay_payload_bytes"].as_u64(),
        Some(63_681)
    );
    assert_eq!(
        inference_output["memory"]["runtime_state_payload_bytes"].as_u64(),
        Some(69_176)
    );
    assert_eq!(
        inference_output["memory"]["forward_recompute_steps"].as_u64(),
        Some(1)
    );
    assert_eq!(
        inference_output["memory"]["inverse_recompute_steps"].as_u64(),
        Some(1)
    );
    assert_eq!(
        inference_output["proof"]["claim"].as_str(),
        Some("deterministic_q31_inference_replay")
    );
    assert_eq!(
        inference_output["proof"]["replay_payload_bytes"].as_u64(),
        Some(63_681)
    );
    assert_eq!(
        inference_output["proof"]["checks"]["restored_initial_state"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference_output["proof"]["checks"]["attribution_matches_margin"].as_bool(),
        Some(true)
    );

    let inference_bundle_text =
        std::fs::read_to_string(&inference_bundle).expect("inference bundle is written");
    let inference_bundle_json: serde_json::Value =
        serde_json::from_str(&inference_bundle_text).expect("inference bundle is valid JSON");
    assert_eq!(
        inference_bundle_json["kind"].as_str(),
        Some("reverie_mnist_linear_q31_inference_replay_bundle")
    );
    assert_eq!(inference_bundle_json["schema_version"].as_u64(), Some(1));
    assert_eq!(
        inference_bundle_json["source_training_bundle"]["fingerprints"]["computation"].as_str(),
        Some(computation_fingerprint.as_str())
    );
    assert_eq!(
        inference_bundle_json["model"]["weights"]
            .as_array()
            .map(Vec::len),
        Some(784)
    );
    assert_eq!(
        inference_bundle_json["sample"]["image_u8"]
            .as_array()
            .map(Vec::len),
        Some(784)
    );
    assert_eq!(
        inference_bundle_json["result"]["logits"]
            .as_array()
            .map(Vec::len),
        Some(10)
    );
    assert_eq!(
        inference_bundle_json["result"]["prediction"].as_i64(),
        Some(0)
    );
    assert_eq!(
        inference_bundle_json["result"]["attribution"]["matches_logit"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference_bundle_json["result"]["attribution"]["matches_margin"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference_bundle_json["result"]["attribution"]["top_contributions"][0]["pixel"].as_u64(),
        Some(0)
    );
    assert_eq!(
        inference_bundle_json["result"]["attribution"]["top_margin_contributions"][0]
            ["weight_delta"]
            .as_i64(),
        Some(1_684_168_090)
    );
    assert_eq!(
        inference_bundle_json["report"]["attribution"]["reconstructed_logit"].as_i64(),
        Some(2_127_803_771)
    );
    assert_eq!(
        inference_bundle_json["report"]["attribution"]["reconstructed_margin"].as_i64(),
        Some(2_102_090_579)
    );
    assert_eq!(
        inference_bundle_json["result"]["correct"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference_bundle_json["result"]["inverse"]["restored_initial_state"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference_bundle_json["storage"]["replay_payload_bytes"].as_u64(),
        Some(63_681)
    );
    assert_eq!(
        inference_bundle_json["storage"]["trace_payload_bytes"].as_u64(),
        Some(0)
    );
    assert_eq!(
        inference_bundle_json["proof"]["claim"].as_str(),
        Some("deterministic_q31_inference_replay")
    );
    assert_eq!(
        inference_bundle_json["proof"]["kernel"].as_str(),
        Some("examples/mnist_identify.rev")
    );
    assert_eq!(
        inference_bundle_json["proof"]["model_payload_bytes"].as_u64(),
        Some(62_800)
    );
    assert_eq!(
        inference_bundle_json["proof"]["sample_payload_bytes"].as_u64(),
        Some(785)
    );
    assert_eq!(
        inference_bundle_json["proof"]["witness_payload_bytes"].as_u64(),
        Some(96)
    );
    assert_eq!(
        inference_bundle_json["proof"]["trace_payload_bytes"].as_u64(),
        Some(0)
    );
    assert_eq!(
        inference_bundle_json["proof"]["replay_payload_bytes"].as_u64(),
        Some(63_681)
    );
    assert_eq!(
        inference_bundle_json["proof"]["checks"]["prediction_matches_logits"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference_bundle_json["proof"]["checks"]["correct_matches_logits"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference_bundle_json["proof"]["checks"]["restored_initial_state"].as_bool(),
        Some(true)
    );
    assert_eq!(
        inference_bundle_json["proof"]["fingerprints"]["algorithm"].as_str(),
        Some("sha256")
    );
    assert_eq!(
        inference_bundle_json["proof"]["fingerprints"]["model"].as_str(),
        Some(sha256_json(&inference_bundle_json["model"]).as_str())
    );
    assert_eq!(
        inference_bundle_json["proof"]["fingerprints"]["sample"].as_str(),
        Some(sha256_json(&inference_bundle_json["sample"]).as_str())
    );
    assert_eq!(
        inference_bundle_json["proof"]["fingerprints"]["result"].as_str(),
        Some(sha256_json(&inference_bundle_json["result"]).as_str())
    );
    assert_eq!(
        inference_bundle_json["result"]["memory"]["witness_payload_bytes"].as_u64(),
        Some(96)
    );
    assert_eq!(
        inference_bundle_json["fingerprints"]["algorithm"].as_str(),
        Some("sha256")
    );
    assert_eq!(
        inference_bundle_json["fingerprints"]["proof"].as_str(),
        Some(sha256_json(&inference_bundle_json["proof"]).as_str())
    );
    let inference_payload_fingerprint = inference_bundle_json["fingerprints"]["payload"]
        .as_str()
        .expect("inference bundle has payload fingerprint")
        .to_owned();
    let inference_computation_fingerprint = inference_bundle_json["fingerprints"]["computation"]
        .as_str()
        .expect("inference bundle has computation fingerprint")
        .to_owned();
    assert_eq!(inference_payload_fingerprint.len(), 64);
    assert_eq!(inference_computation_fingerprint.len(), 64);
    assert_ne!(
        inference_computation_fingerprint,
        inference_payload_fingerprint
    );

    let verify_inference = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&inference_bundle)
        .arg("--json")
        .assert()
        .success();
    let stdout = std::str::from_utf8(&verify_inference.get_output().stdout)
        .expect("verify inference JSON report is UTF-8");
    let verify_inference: serde_json::Value =
        serde_json::from_str(stdout).expect("verify inference emits valid JSON");
    assert_eq!(
        verify_inference["kind"].as_str(),
        Some("reverie_mnist_linear_q31_inference_verification")
    );
    assert_eq!(
        verify_inference["fingerprints"]["payload"].as_str(),
        Some(inference_payload_fingerprint.as_str())
    );
    assert_eq!(
        verify_inference["fingerprints"]["computation"].as_str(),
        Some(inference_computation_fingerprint.as_str())
    );
    assert_eq!(verify_inference["prediction"].as_i64(), Some(0));
    assert_eq!(verify_inference["correct"].as_bool(), Some(true));
    assert_eq!(verify_inference["result_matches"].as_bool(), Some(true));
    assert_eq!(verify_inference["proof_matches"].as_bool(), Some(true));
    assert_eq!(
        verify_inference["restored_initial_state"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verify_inference["attribution"]["matches_logit"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verify_inference["attribution"]["matches_margin"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verify_inference["attribution"]["top_contributions"][0]["contribution"].as_i64(),
        Some(1_555_136_903)
    );
    assert_eq!(
        verify_inference["attribution"]["top_margin_contributions"][0]["contribution"].as_i64(),
        Some(1_684_168_090)
    );
    assert_eq!(
        verify_inference["memory"]["replay_payload_bytes"].as_u64(),
        Some(63_681)
    );
    assert_eq!(
        verify_inference["memory"]["runtime_state_payload_bytes"].as_u64(),
        Some(69_176)
    );
    assert_eq!(
        verify_inference["proof"]["claim"].as_str(),
        Some("deterministic_q31_inference_replay")
    );
    assert_eq!(
        verify_inference["proof"]["fingerprints"]["result"].as_str(),
        Some(sha256_json(&inference_bundle_json["result"]).as_str())
    );
    assert_eq!(
        verify_inference["source_training_checked"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verify_inference["source_training_bundle"]["payload_fingerprint"].as_str(),
        Some(payload_fingerprint.as_str())
    );
    assert_eq!(
        verify_inference["source_training_bundle"]["items_checked"].as_u64(),
        Some(2)
    );

    let inference_markdown = artifact_dir.join("self-test-inference-verification.md");
    let _ = std::fs::remove_file(&inference_markdown);
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&inference_bundle)
        .arg("--markdown-output")
        .arg(&inference_markdown)
        .assert()
        .success()
        .stdout(predicate::str::contains("inference audit ok"))
        .stdout(predicate::str::contains("restored_initial_state=true"))
        .stdout(predicate::str::contains(
            "storage: replay_payload_bytes=63681 witness_payload_bytes=96 trace_payload_bytes=0",
        ))
        .stdout(predicate::str::contains(
            "proof: claim=deterministic_q31_inference_replay model_bytes=62800 sample_bytes=785 witness_bytes=96",
        ))
        .stdout(predicate::str::contains(
            "source_training_bundle: checked=true",
        ));
    let inference_markdown =
        std::fs::read_to_string(&inference_markdown).expect("inference Markdown is written");
    assert!(inference_markdown.contains("# Reverie Inference Verification"));
    assert!(inference_markdown.contains("## Explanation"));
    assert!(inference_markdown.contains("## Top Contributions"));
    assert!(inference_markdown.contains("## Top Margin Contributions"));
    assert!(inference_markdown.contains("## Replay Proof"));
    assert!(inference_markdown.contains("## Contract Checks"));
    assert!(inference_markdown.contains("## Source Checks"));
    assert!(inference_markdown.contains("`deterministic_q31_inference_replay`"));
    assert!(inference_markdown.contains("`source_inputs_checked`"));
    assert!(inference_markdown.contains("| training bundle | ok | 2 |"));

    let bad_training_inference_source =
        artifact_dir.join("self-test-training-inference-bad-source.json");
    let mut bad_training_inference_source_json = inference_bundle_json.clone();
    bad_training_inference_source_json["source_training_bundle"]["audit_step"] = json!(1);
    bad_training_inference_source_json["source_training_bundle"]["sample_index"] = json!(1);
    resign_mnist_inference_bundle(&workspace_root(), &mut bad_training_inference_source_json);
    std::fs::write(
        &bad_training_inference_source,
        serde_json::to_string_pretty(&bad_training_inference_source_json)
            .expect("bad training inference source bundle serializes"),
    )
    .expect("bad training inference source bundle can be written");
    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&bad_training_inference_source)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "inference source training label expected",
        ));

    let comparison = Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--json")
        .arg("--compare-artifacts")
        .arg(&bundle)
        .arg(&model_bundle)
        .arg(&step_bundle)
        .arg(&inference_bundle)
        .arg(&evaluation_bundle)
        .arg(&samples_json)
        .assert()
        .success();
    let stdout = std::str::from_utf8(&comparison.get_output().stdout)
        .expect("artifact comparison JSON report is UTF-8");
    let comparison: serde_json::Value =
        serde_json::from_str(stdout).expect("artifact comparison emits valid JSON");
    assert_eq!(
        comparison["kind"].as_str(),
        Some("reverie_mnist_linear_q31_artifact_comparison")
    );
    assert_eq!(comparison["totals"]["count"].as_u64(), Some(6));
    assert!(comparison["totals"]["file_bytes"].as_u64().unwrap_or(0) > 0);
    assert_eq!(
        comparison["totals"]["logical_payload_bytes"].as_u64(),
        Some(480_598)
    );
    assert_eq!(
        comparison["ml_profile"]["total_model_payload_bytes"].as_u64(),
        Some(376_800)
    );
    assert_eq!(
        comparison["ml_profile"]["total_sample_payload_bytes"].as_u64(),
        Some(83_110)
    );
    assert_eq!(
        comparison["ml_profile"]["total_witness_payload_bytes"].as_u64(),
        Some(20_472)
    );
    assert_eq!(
        comparison["ml_profile"]["total_trace_payload_bytes"].as_u64(),
        Some(98_400)
    );
    assert_eq!(
        comparison["ml_profile"]["total_derived_update_payload_bytes"].as_u64(),
        Some(216)
    );
    assert_eq!(comparison["ml_profile"]["total_steps"].as_u64(), Some(106));
    assert_eq!(
        comparison["ml_profile"]["total_forward_recompute_steps"].as_u64(),
        Some(104)
    );
    assert_eq!(
        comparison["ml_profile"]["total_inverse_recompute_steps"].as_u64(),
        Some(104)
    );
    assert_eq!(
        comparison["ml_profile"]["total_recompute_steps"].as_u64(),
        Some(208)
    );
    assert_eq!(
        comparison["ml_profile"]["trace_to_model_payload_ratio"].as_f64(),
        Some(98_400.0 / 376_800.0)
    );
    assert_eq!(
        comparison["ml_profile"]["witness_to_model_payload_ratio"].as_f64(),
        Some(20_472.0 / 376_800.0)
    );
    assert_eq!(
        comparison["audit_contract"]["claim"].as_str(),
        Some("reversible_inspectable_deterministic_q31_ml_kernel")
    );
    assert_eq!(comparison["audit_contract"]["passed"].as_bool(), Some(true));
    let contract_checks = comparison["audit_contract"]["checks"]
        .as_array()
        .expect("comparison has audit contract checks");
    assert_eq!(contract_checks.len(), 10);
    assert!(contract_checks.iter().all(|check| {
        check["metric"].as_str().is_some()
            && check["actual"].as_str().is_some()
            && check["requirement"].as_str().is_some()
            && check["passed"].as_bool() == Some(true)
    }));
    assert!(contract_checks.iter().any(|check| {
        check["metric"].as_str() == Some("fingerprint_coverage")
            && check["passed"].as_bool() == Some(true)
    }));
    assert!(contract_checks.iter().any(|check| {
        check["metric"].as_str() == Some("proof_or_provenance_fingerprints")
            && check["passed"].as_bool() == Some(true)
    }));
    let artifacts = comparison["artifacts"]
        .as_array()
        .expect("comparison has artifacts");
    assert_eq!(artifacts.len(), 6);
    assert_eq!(artifacts[0]["kind"].as_str(), Some("training_trace"));
    assert_eq!(artifacts[0]["steps"].as_u64(), Some(100));
    assert_eq!(
        artifacts[0]["logical_payload_bytes"].as_u64(),
        Some(161_200)
    );
    assert_eq!(artifacts[0]["trace_payload_bytes"].as_u64(), Some(98_400));
    assert_eq!(
        artifacts[0]["fingerprints"]["algorithm"].as_str(),
        Some("sha256")
    );
    assert_eq!(
        artifacts[0]["fingerprints"]["source_count"].as_u64(),
        Some(2)
    );
    assert_eq!(
        artifacts[0]["fingerprints"]["has_computation"].as_bool(),
        Some(true)
    );
    assert_eq!(
        artifacts[0]["fingerprints"]["has_payload"].as_bool(),
        Some(true)
    );
    assert_eq!(
        artifacts[0]["fingerprints"]["has_proof"].as_bool(),
        Some(true)
    );
    assert_eq!(artifacts[1]["kind"].as_str(), Some("model"));
    assert_eq!(artifacts[1]["logical_payload_bytes"].as_u64(), Some(62_800));
    assert_eq!(artifacts[1]["steps"].as_u64(), Some(0));
    assert_eq!(
        artifacts[1]["fingerprints"]["has_provenance"].as_bool(),
        Some(true)
    );
    assert_eq!(
        artifacts[1]["payload_fingerprint"].as_str(),
        Some(model_payload_fingerprint.as_str())
    );
    assert_eq!(artifacts[2]["kind"].as_str(), Some("training_step"));
    assert_eq!(artifacts[2]["steps"].as_u64(), Some(1));
    assert!(artifacts[2]["logical_payload_bytes"].as_u64().unwrap_or(0) > 126_000);
    assert_eq!(
        artifacts[2]["derived_update_payload_bytes"].as_u64(),
        Some(216)
    );
    assert_eq!(artifacts[3]["kind"].as_str(), Some("inference"));
    assert_eq!(artifacts[3]["logical_payload_bytes"].as_u64(), Some(63_681));
    assert_eq!(artifacts[3]["trace_payload_bytes"].as_u64(), Some(0));
    assert_eq!(
        artifacts[3]["payload_fingerprint"].as_str(),
        Some(inference_payload_fingerprint.as_str())
    );
    assert_eq!(artifacts[4]["kind"].as_str(), Some("model_evaluation"));
    assert_eq!(artifacts[4]["steps"].as_u64(), Some(2));
    assert_eq!(artifacts[4]["logical_payload_bytes"].as_u64(), Some(64_562));
    assert_eq!(artifacts[4]["trace_payload_bytes"].as_u64(), Some(0));
    assert_eq!(
        artifacts[4]["payload_fingerprint"].as_str(),
        Some(evaluation_payload_fingerprint.as_str())
    );
    assert_eq!(artifacts[5]["kind"].as_str(), Some("sample_set"));
    assert_eq!(artifacts[5]["steps"].as_u64(), Some(2));
    assert_eq!(artifacts[5]["logical_payload_bytes"].as_u64(), Some(1_570));
    assert_eq!(artifacts[5]["sample_payload_bytes"].as_u64(), Some(1_570));
    assert_eq!(
        artifacts[5]["payload_fingerprint"].as_str(),
        Some(samples_payload_fingerprint.as_str())
    );

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--compare-artifacts")
        .arg(&bundle)
        .arg(&model_bundle)
        .arg(&step_bundle)
        .arg(&inference_bundle)
        .arg(&evaluation_bundle)
        .arg(&samples_json)
        .assert()
        .success()
        .stdout(predicate::str::contains("artifact comparison"))
        .stdout(predicate::str::contains(
            "ml_profile: model_payload_bytes=376800",
        ))
        .stdout(predicate::str::contains("total_recompute_steps=208"))
        .stdout(predicate::str::contains("trace_to_model_ratio=0.261146"))
        .stdout(predicate::str::contains(
            "audit_contract: claim=reversible_inspectable_deterministic_q31_ml_kernel passed=true checks=10 failed=0",
        ))
        .stdout(predicate::str::contains("kind=training_trace"))
        .stdout(predicate::str::contains("kind=model"))
        .stdout(predicate::str::contains("kind=training_step"))
        .stdout(predicate::str::contains("kind=inference"))
        .stdout(predicate::str::contains("kind=model_evaluation"))
        .stdout(predicate::str::contains("kind=sample_set"));

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--inference-output")
        .arg(artifact_dir.join("unused-inference-bundle.json"))
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--inference-output requires --inspect-inference, --inspect-model-inference, or --inspect-evaluation",
        ));

    let bad_attribution = artifact_dir.join("self-test-inference-bundle-bad-attribution.json");
    let mut bad_attribution_json = inference_bundle_json.clone();
    bad_attribution_json["result"]["attribution"]["top_contributions"][0]["contribution"] =
        serde_json::Value::from(1_i64);
    resign_mnist_inference_bundle(&workspace_root(), &mut bad_attribution_json);
    std::fs::write(
        &bad_attribution,
        serde_json::to_string_pretty(&bad_attribution_json)
            .expect("bad attribution inference bundle serializes"),
    )
    .expect("bad attribution inference bundle can be written");

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&bad_attribution)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "attribution does not match recomputed Q31 contributions",
        ));

    let bad_proof = artifact_dir.join("self-test-inference-bundle-bad-proof.json");
    let mut bad_proof_json = inference_bundle_json.clone();
    bad_proof_json["proof"]["replay_payload_bytes"] = serde_json::Value::from(1_u64);
    resign_mnist_inference_bundle(&workspace_root(), &mut bad_proof_json);
    std::fs::write(
        &bad_proof,
        serde_json::to_string_pretty(&bad_proof_json)
            .expect("bad proof inference bundle serializes"),
    )
    .expect("bad proof inference bundle can be written");

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&bad_proof)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "inference bundle proof does not match recomputed Reverie inference",
        ));

    let tampered_inference = artifact_dir.join("self-test-inference-bundle-tampered.json");
    let mut tampered_inference_json = inference_bundle_json;
    tampered_inference_json["result"]["prediction"] = serde_json::Value::from(9_i64);
    std::fs::write(
        &tampered_inference,
        serde_json::to_string_pretty(&tampered_inference_json)
            .expect("tampered inference bundle serializes"),
    )
    .expect("tampered inference bundle can be written");

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-inference")
        .arg(&tampered_inference)
        .assert()
        .failure()
        .stderr(predicate::str::contains("fingerprint mismatch"));

    let tampered = artifact_dir.join("self-test-replay-bundle-tampered.json");
    let mut tampered_json = bundle_json;
    tampered_json["final_model"]["bias"][0] = serde_json::Value::from(123_i64);
    std::fs::write(
        &tampered,
        serde_json::to_string_pretty(&tampered_json).expect("tampered bundle serializes"),
    )
    .expect("tampered bundle can be written");

    Command::cargo_bin("reverie-mnist-linear")
        .expect("binary exists")
        .arg("--verify-audit")
        .arg(&tampered)
        .assert()
        .failure()
        .stderr(predicate::str::contains("fingerprint mismatch"));
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
fn run_accepts_vars_json_seed_file() {
    let root = workspace_root();
    let example = root.join("target/test-vars-json-run.rev");
    let seeds = root.join("target/test-vars-json-run.json");
    std::fs::write(&example, "x += delta").expect("example can be written");
    std::fs::write(&seeds, r#"{"x":1,"delta":4}"#).expect("seed JSON can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .arg("--vars-json")
        .arg(seeds)
        .assert()
        .success()
        .stdout("{delta = 4, x = 5}\n");
}

#[test]
fn run_json_emits_machine_readable_store_and_tapes() {
    let root = workspace_root();
    let example = root.join("target/test-run-json.rev");
    std::fs::write(
        &example,
        r#"
        global x;
        read x;
        show(x);
        push(x, s)
        "#,
    )
    .expect("example can be written");

    let assert = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .args(["--type", "s=stack", "--var", "s=nil", "--input", "7"])
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&assert.get_output().stdout).expect("run JSON output is UTF-8");
    let result: serde_json::Value = serde_json::from_str(stdout).expect("run emits valid JSON");
    assert_eq!(result["kind"].as_str(), Some("reverie_run_result"));
    assert_eq!(
        result["store"]["s"]["stack"].as_array().unwrap(),
        &[json!(7)]
    );
    assert_eq!(
        result["input"].as_array().unwrap(),
        &Vec::<serde_json::Value>::new()
    );
    assert_eq!(result["store"]["x"].as_i64(), Some(0));
    assert_eq!(result["output"].as_array().unwrap(), &[json!(0)]);
    assert_eq!(
        result["observations"].as_array().unwrap(),
        &[json!("x = 7")]
    );
}

#[test]
fn run_and_reverse_json_emit_witness_metadata() {
    let root = workspace_root();
    let example = root.join("target/test-run-json-witness-metadata.rev");
    std::fs::write(
        &example,
        r#"
        global trace: witness<tensor<int, 2>>;
        global x: tensor<int, 2> = [1, 2];
        trace += x
        "#,
    )
    .expect("example can be written");

    let assert = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&assert.get_output().stdout).expect("run JSON output is UTF-8");
    let result: serde_json::Value = serde_json::from_str(stdout).expect("run emits valid JSON");
    let source_bytes = std::fs::read(&example).expect("example source can be read");
    let example_display = example.display().to_string();
    assert_eq!(result["program"]["direction"].as_str(), Some("run"));
    assert_eq!(result["program"]["engine"].as_str(), Some("slot"));
    assert_eq!(result["program"]["legacy_janus"].as_bool(), Some(false));
    assert_eq!(
        result["program"]["file"].as_str(),
        Some(example_display.as_str())
    );
    assert_eq!(
        result["program"]["source_sha256"].as_str(),
        Some(sha256_bytes(&source_bytes).as_str())
    );
    assert!(result["dataset_loops"].as_array().unwrap().is_empty());
    assert_eq!(
        result["store"]["trace"].as_array().unwrap(),
        &[json!(1), json!(2)]
    );
    assert_eq!(
        result["witness_store"].as_array().unwrap(),
        &[json!("trace")]
    );
    assert_eq!(result["witness_metrics"]["variables"].as_u64(), Some(1));
    assert_eq!(result["witness_metrics"]["known_cells"].as_u64(), Some(2));
    assert_eq!(
        result["witness_metrics"]["known_payload_bytes"].as_u64(),
        Some(16)
    );
    assert_eq!(
        result["witness_proof"]["schema"].as_str(),
        Some("reverie_witness_store_proof_v1")
    );
    assert_eq!(
        result["witness_proof"]["algorithm"].as_str(),
        Some("sha256")
    );
    assert_eq!(
        result["witness_proof"]["fingerprint"].as_str(),
        Some(sha256_json(&result["witness_proof"]["payload"]).as_str())
    );
    let witness_entry = result["witness_proof"]["payload"]["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"].as_str() == Some("trace"))
        .expect("trace witness proof entry is present");
    assert_eq!(witness_entry["present"].as_bool(), Some(true));
    assert_eq!(witness_entry["cells"].as_u64(), Some(2));
    assert_eq!(witness_entry["payload_bytes"].as_u64(), Some(16));
    assert_eq!(
        witness_entry["value_fingerprint"].as_str(),
        Some(sha256_json(&json!([1, 2])).as_str())
    );
    let trace_metadata = result["store_metadata"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"].as_str() == Some("trace"))
        .expect("trace metadata is present");
    assert_eq!(trace_metadata["role"].as_str(), Some("witness"));
    assert_eq!(
        trace_metadata["type"].as_str(),
        Some("witness<tensor<int, 2>>")
    );
    assert_eq!(trace_metadata["cells"].as_u64(), Some(2));
    assert_eq!(trace_metadata["payload_bytes"].as_u64(), Some(16));

    let assert = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(&example)
        .args(["--var", "trace=[1,2]"])
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&assert.get_output().stdout).expect("reverse JSON output is UTF-8");
    let result: serde_json::Value = serde_json::from_str(stdout).expect("reverse emits valid JSON");
    assert_eq!(result["kind"].as_str(), Some("reverie_reverse_result"));
    assert_eq!(result["program"]["direction"].as_str(), Some("reverse"));
    assert_eq!(result["program"]["engine"].as_str(), Some("slot"));
    assert_eq!(
        result["program"]["source_sha256"].as_str(),
        Some(sha256_bytes(&source_bytes).as_str())
    );
    assert!(result["dataset_loops"].as_array().unwrap().is_empty());
    assert_eq!(
        result["store"]["trace"].as_array().unwrap(),
        &[json!(0), json!(0)]
    );
    assert_eq!(
        result["witness_store"].as_array().unwrap(),
        &[json!("trace")]
    );
    assert_eq!(
        result["witness_metrics"]["known_payload_bytes"].as_u64(),
        Some(16)
    );
    assert_eq!(
        result["witness_proof"]["fingerprint"].as_str(),
        Some(sha256_json(&result["witness_proof"]["payload"]).as_str())
    );
    let witness_entry = result["witness_proof"]["payload"]["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"].as_str() == Some("trace"))
        .expect("trace reverse witness proof entry is present");
    assert_eq!(
        witness_entry["value_fingerprint"].as_str(),
        Some(sha256_json(&json!([0, 0])).as_str())
    );
}

#[test]
fn roundtrip_plain_text_proves_restoration() {
    let example = workspace_root().join("examples/increment.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("roundtrip")
        .arg(example)
        .assert()
        .success()
        .stdout(predicate::str::contains("roundtrip: ok"))
        .stdout(predicate::str::contains(
            "checks: store=true input=true output=true",
        ))
        .stdout(predicate::str::contains("baseline: {x = 0, xs = [0, 0]}"))
        .stdout(predicate::str::contains("forward: {x = 1, xs = [-1, 1]}"))
        .stdout(predicate::str::contains("restored: {x = 0, xs = [0, 0]}"))
        .stdout(predicate::str::contains("fingerprint: "));
}

#[test]
fn roundtrip_json_emits_fingerprinted_witness_proof() {
    let root = workspace_root();
    let example = root.join("target/test-roundtrip-json-witness.rev");
    std::fs::write(
        &example,
        r#"
        global trace: witness<tensor<int, 2>>;
        global x: tensor<int, 2> = [1, 2];
        trace += x
        "#,
    )
    .expect("example can be written");

    let assert = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("roundtrip")
        .arg(&example)
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&assert.get_output().stdout).expect("roundtrip JSON output is UTF-8");
    let result: serde_json::Value =
        serde_json::from_str(stdout).expect("roundtrip emits valid JSON");
    let payload = &result["payload"];
    let source_bytes = std::fs::read(&example).expect("example source can be read");

    assert_eq!(result["kind"].as_str(), Some("reverie_roundtrip_result"));
    assert_eq!(
        result["schema"].as_str(),
        Some("reverie_roundtrip_proof_v1")
    );
    assert_eq!(result["passed"].as_bool(), Some(true));
    assert_eq!(
        result["fingerprint"].as_str(),
        Some(sha256_json(payload).as_str())
    );
    assert_eq!(
        result["fingerprints"]["payload"].as_str(),
        result["fingerprint"].as_str()
    );
    assert_eq!(
        result["fingerprints"]["baseline"].as_str(),
        Some(sha256_json(&payload["baseline"]).as_str())
    );
    assert_eq!(
        result["fingerprints"]["forward"].as_str(),
        Some(sha256_json(&payload["forward"]).as_str())
    );
    assert_eq!(
        result["fingerprints"]["restored"].as_str(),
        Some(sha256_json(&payload["restored"]).as_str())
    );
    assert_eq!(
        result["fingerprints"]["forward_witness_proof"].as_str(),
        Some(sha256_json(&payload["forward_witness_proof"]).as_str())
    );
    assert_eq!(
        result["fingerprints"]["restored_witness_proof"].as_str(),
        Some(sha256_json(&payload["restored_witness_proof"]).as_str())
    );
    assert_eq!(payload["program"]["direction"].as_str(), Some("roundtrip"));
    assert_eq!(payload["program"]["engine"].as_str(), Some("slot"));
    assert_eq!(payload["program"]["legacy_janus"].as_bool(), Some(false));
    assert_eq!(
        payload["program"]["source_sha256"].as_str(),
        Some(sha256_bytes(&source_bytes).as_str())
    );
    assert_eq!(payload["check"]["passed"].as_bool(), Some(true));
    assert_eq!(
        payload["check"]["baseline_store_restored"].as_bool(),
        Some(true)
    );
    assert_eq!(payload["check"]["input_restored"].as_bool(), Some(true));
    assert_eq!(payload["check"]["output_restored"].as_bool(), Some(true));
    assert_eq!(
        payload["baseline"]["store"]["trace"].as_array().unwrap(),
        &[json!(0), json!(0)]
    );
    assert_eq!(
        payload["forward"]["store"]["trace"].as_array().unwrap(),
        &[json!(1), json!(2)]
    );
    assert_eq!(
        payload["restored"]["store"]["trace"].as_array().unwrap(),
        &[json!(0), json!(0)]
    );
    assert_eq!(
        payload["witness_store"].as_array().unwrap(),
        &[json!("trace")]
    );
    assert_eq!(payload["witness_metrics"]["known_cells"].as_u64(), Some(2));
    assert_eq!(
        payload["ml_profile"]["schema"].as_str(),
        Some("reverie_explain_ml_profile_v1")
    );
    assert_eq!(
        payload["ml_profile"]["goal_fit"].as_str(),
        Some("tensor_reversible_kernel")
    );
    assert_eq!(
        payload["ml_profile"]["tensor_metrics"]["known_witness_payload_bytes"].as_u64(),
        Some(16)
    );
    assert_eq!(
        payload["forward_witness_proof"]["fingerprint"].as_str(),
        Some(sha256_json(&payload["forward_witness_proof"]["payload"]).as_str())
    );
    assert_eq!(
        payload["restored_witness_proof"]["fingerprint"].as_str(),
        Some(sha256_json(&payload["restored_witness_proof"]["payload"]).as_str())
    );
    let forward_entry = payload["forward_witness_proof"]["payload"]["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"].as_str() == Some("trace"))
        .expect("forward witness entry is present");
    assert_eq!(
        forward_entry["value_fingerprint"].as_str(),
        Some(sha256_json(&json!([1, 2])).as_str())
    );
    let restored_entry = payload["restored_witness_proof"]["payload"]["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"].as_str() == Some("trace"))
        .expect("restored witness entry is present");
    assert_eq!(
        restored_entry["value_fingerprint"].as_str(),
        Some(sha256_json(&json!([0, 0])).as_str())
    );
}

#[test]
fn roundtrip_json_embeds_ml_audit_profile_for_q31_kernel() {
    let root = workspace_root();
    let example = root.join("target/test-roundtrip-ml-profile-q31.rev");
    std::fs::write(
        &example,
        r#"
        global x: tensor<int, 2> = [1073741824, -1073741824];
        global w: tensor<int, 2, 2> = [[1073741824, 0], [0, 1073741824]];
        global y: witness<tensor<int, 2>>;
        y += vecmat_q31(x, w)
        "#,
    )
    .expect("example can be written");

    let assert = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("roundtrip")
        .arg(&example)
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&assert.get_output().stdout).expect("roundtrip JSON output is UTF-8");
    let result: serde_json::Value =
        serde_json::from_str(stdout).expect("roundtrip emits valid JSON");
    let payload = &result["payload"];
    let profile = &payload["ml_profile"];

    assert_eq!(result["passed"].as_bool(), Some(true));
    assert_eq!(
        result["fingerprint"].as_str(),
        Some(sha256_json(payload).as_str())
    );
    assert_eq!(profile["goal_fit"].as_str(), Some("auditable_ml_kernel"));
    assert_eq!(profile["q31_builtin_calls"].as_u64(), Some(1));
    assert_eq!(
        profile["tensor_metrics"]["known_witness_payload_bytes"].as_u64(),
        Some(16)
    );
    assert_eq!(
        profile["replay_cost"]["known_witness_payload_bytes"].as_u64(),
        Some(16)
    );
    assert!(
        profile["replay_cost"]["roundtrip_statement_count"]
            .as_u64()
            .expect("roundtrip statement count is present")
            > 0
    );
    assert!(
        profile["replay_cost"]["known_replay_payload_bytes"]
            .as_u64()
            .expect("replay payload bytes are present")
            >= 16
    );
    assert_eq!(
        profile["update_counts"]["witness_update_statements"].as_u64(),
        Some(1)
    );
}

#[test]
fn roundtrip_writes_and_verifies_saved_proof() {
    let root = workspace_root();
    let example = root.join("target/test-roundtrip-proof-output.rev");
    let proof = root.join("target/test-roundtrip-proof-output.json");
    let markdown = root.join("target/test-roundtrip-proof-output.md");
    std::fs::write(
        &example,
        r#"
        global trace: witness<tensor<int, 2>>;
        global x: tensor<int, 2> = [3, 4];
        trace += x
        "#,
    )
    .expect("example can be written");
    let _ = std::fs::remove_file(&proof);
    let _ = std::fs::remove_file(&markdown);

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("roundtrip")
        .arg(&example)
        .arg("--proof-output")
        .arg(&proof)
        .assert()
        .success()
        .stdout(predicate::str::contains("roundtrip: ok"))
        .stdout(predicate::str::contains(format!(
            "proof: {}",
            proof.display()
        )));

    let proof_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&proof).expect("proof is written"))
            .expect("proof file is valid JSON");
    assert_eq!(
        proof_json["fingerprint"].as_str(),
        Some(sha256_json(&proof_json["payload"]).as_str())
    );

    let assert = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("verify-roundtrip")
        .arg(&proof)
        .arg("--json")
        .arg("--markdown-output")
        .arg(&markdown)
        .assert()
        .success();
    let stdout = std::str::from_utf8(&assert.get_output().stdout)
        .expect("roundtrip verification JSON output is UTF-8");
    let verification: serde_json::Value =
        serde_json::from_str(stdout).expect("verification emits valid JSON");

    assert_eq!(
        verification["kind"].as_str(),
        Some("reverie_roundtrip_verification")
    );
    assert_eq!(verification["passed"].as_bool(), Some(true));
    assert_eq!(
        verification["proof_fingerprint"].as_str(),
        proof_json["fingerprint"].as_str()
    );
    assert_eq!(
        verification["replay_fingerprint"].as_str(),
        proof_json["fingerprint"].as_str()
    );
    assert_eq!(
        verification["checks"]["payload_fingerprint"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verification["checks"]["source_hash_matches"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verification["checks"]["replay_fingerprint_matches"].as_bool(),
        Some(true)
    );
    assert_eq!(verification["ml_profile_present"].as_bool(), Some(true));
    assert_eq!(
        verification["checks"]["ml_profile_schema"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verification["checks"]["ml_profile_replayed"].as_bool(),
        Some(true)
    );
    assert_eq!(
        verification["ml_profile_fingerprint"].as_str(),
        Some(sha256_json(&verification["ml_profile"]).as_str())
    );
    assert!(
        verification["ml_profile"]["replay_cost"]["roundtrip_statement_count"]
            .as_u64()
            .expect("verified ML profile carries replay cost")
            > 0
    );
    let markdown_text = std::fs::read_to_string(&markdown).expect("Markdown card is written");
    assert!(markdown_text.contains("# Reverie Roundtrip Verification"));
    assert!(markdown_text.contains("**Verdict:** pass"));
    assert!(markdown_text.contains(proof_json["fingerprint"].as_str().unwrap()));
    assert!(markdown_text.contains("| Variables | Known cells | Known payload bytes |"));
    assert!(markdown_text.contains("## ML Profile"));
    assert!(markdown_text.contains("Replay payload bytes"));
    assert!(markdown_text.contains("tensor_reversible_kernel"));
    assert!(markdown_text.contains("| ml profile replayed | true |"));
    assert!(markdown_text.contains("| payload |"));
    assert!(markdown_text.contains("| replay fingerprint matches | true |"));
}

#[test]
fn verify_roundtrip_accepts_legacy_proof_without_ml_profile() {
    let root = workspace_root();
    let example = root.join("target/test-roundtrip-legacy-no-ml.rev");
    let proof = root.join("target/test-roundtrip-legacy-no-ml.json");
    std::fs::write(
        &example,
        r#"
        global trace: witness<tensor<int, 2>>;
        global x: tensor<int, 2> = [7, 8];
        trace += x
        "#,
    )
    .expect("example can be written");

    let assert = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("roundtrip")
        .arg(&example)
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&assert.get_output().stdout).expect("roundtrip JSON output is UTF-8");
    let mut proof_json: serde_json::Value =
        serde_json::from_str(stdout).expect("roundtrip emits valid JSON");

    proof_json["payload"]
        .as_object_mut()
        .expect("payload is an object")
        .remove("ml_profile");
    let legacy_fingerprint = sha256_json(&proof_json["payload"]);
    proof_json["fingerprint"] = json!(legacy_fingerprint.clone());
    proof_json["fingerprints"]["payload"] = json!(legacy_fingerprint);
    std::fs::write(
        &proof,
        serde_json::to_string_pretty(&proof_json).expect("legacy proof serializes"),
    )
    .expect("legacy proof is written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("verify-roundtrip")
        .arg(&proof)
        .assert()
        .success()
        .stdout(predicate::str::contains("roundtrip verification: ok"))
        .stdout(predicate::str::contains("replay_fingerprint_matches: true"));
}

#[test]
fn verify_roundtrip_rejects_tampered_proof() {
    let root = workspace_root();
    let example = root.join("target/test-roundtrip-proof-tamper.rev");
    let proof = root.join("target/test-roundtrip-proof-tamper.json");
    let tampered = root.join("target/test-roundtrip-proof-tampered.json");
    std::fs::write(
        &example,
        r#"
        global trace: witness<tensor<int, 2>>;
        global x: tensor<int, 2> = [5, 6];
        trace += x
        "#,
    )
    .expect("example can be written");
    let _ = std::fs::remove_file(&proof);
    let _ = std::fs::remove_file(&tampered);

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("roundtrip")
        .arg(&example)
        .arg("--proof-output")
        .arg(&proof)
        .assert()
        .success();

    let mut proof_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&proof).expect("proof is written"))
            .expect("proof file is valid JSON");
    proof_json["payload"]["forward"]["store"]["trace"][0] = json!(99);
    std::fs::write(
        &tampered,
        serde_json::to_string_pretty(&proof_json).expect("tampered proof serializes"),
    )
    .expect("tampered proof can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("verify-roundtrip")
        .arg(&tampered)
        .assert()
        .failure()
        .stderr(predicate::str::contains("failed verification"))
        .stdout(predicate::str::contains("roundtrip verification: failed"))
        .stdout(predicate::str::contains("payload_fingerprint: false"));
}

#[test]
fn run_and_reverse_json_emit_dataset_loop_metadata() {
    let root = workspace_root();
    let example = root.join("target/test-run-json-dataset-loop.rev");
    std::fs::write(
        &example,
        r#"
        global labels: tensor<int, 2> = [0, 1];
        global trace: witness<tensor<int, 2>>;

        procedure main() {
          iterate int sample = 0 to len(labels) - 1
            trace[sample] += labels[sample]
          end
        }
        "#,
    )
    .expect("example can be written");

    let assert = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&assert.get_output().stdout).expect("run JSON output is UTF-8");
    let result: serde_json::Value = serde_json::from_str(stdout).expect("run emits valid JSON");
    assert_eq!(result["kind"].as_str(), Some("reverie_run_result"));
    assert_eq!(
        result["dataset_loops"],
        json!([{"index": "sample", "size_sources": ["labels"]}])
    );
    assert_eq!(
        result["store"]["trace"].as_array().unwrap(),
        &[json!(0), json!(1)]
    );
    assert_eq!(
        result["witness_store"].as_array().unwrap(),
        &[json!("trace")]
    );
    assert_eq!(result["witness_metrics"]["known_cells"].as_u64(), Some(2));
    assert_eq!(
        result["witness_metrics"]["known_payload_bytes"].as_u64(),
        Some(16)
    );
    assert_eq!(
        result["witness_proof"]["fingerprint"].as_str(),
        Some(sha256_json(&result["witness_proof"]["payload"]).as_str())
    );

    let assert = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(&example)
        .args(["--var", "trace=[0,1]"])
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&assert.get_output().stdout).expect("reverse JSON output is UTF-8");
    let result: serde_json::Value = serde_json::from_str(stdout).expect("reverse emits valid JSON");
    assert_eq!(result["kind"].as_str(), Some("reverie_reverse_result"));
    assert_eq!(
        result["dataset_loops"],
        json!([{"index": "sample", "size_sources": ["labels"]}])
    );
    assert_eq!(
        result["store"]["trace"].as_array().unwrap(),
        &[json!(0), json!(0)]
    );
    assert_eq!(
        result["witness_proof"]["fingerprint"].as_str(),
        Some(sha256_json(&result["witness_proof"]["payload"]).as_str())
    );
}

#[test]
fn reverse_accepts_vars_json_seed_file() {
    let root = workspace_root();
    let example = root.join("target/test-vars-json-reverse.rev");
    let seeds = root.join("target/test-vars-json-reverse.json");
    std::fs::write(&example, "x += delta").expect("example can be written");
    std::fs::write(&seeds, r#"{"x":5,"delta":4}"#).expect("seed JSON can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .arg("--vars-json")
        .arg(seeds)
        .assert()
        .success()
        .stdout("{delta = 4, x = 1}\n");
}

#[test]
fn reverse_json_emits_machine_readable_store() {
    let root = workspace_root();
    let example = root.join("target/test-reverse-json.rev");
    std::fs::write(&example, "x += delta").expect("example can be written");

    let assert = Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "x=5", "--var", "delta=4"])
        .arg("--json")
        .assert()
        .success();
    let stdout =
        std::str::from_utf8(&assert.get_output().stdout).expect("reverse JSON output is UTF-8");
    let result: serde_json::Value = serde_json::from_str(stdout).expect("reverse emits valid JSON");
    assert_eq!(result["kind"].as_str(), Some("reverie_reverse_result"));
    assert_eq!(result["store"]["x"].as_i64(), Some(1));
    assert_eq!(result["store"]["delta"].as_i64(), Some(4));
}

#[test]
fn vars_json_accepts_nested_arrays_and_stacks() {
    let root = workspace_root();
    let example = root.join("examples/skip.rev");
    let seeds = root.join("target/test-vars-json-structured.json");
    std::fs::write(&seeds, r#"{"matrix":[[1,2],[3,4]],"s":{"stack":[3,2,1]}}"#)
        .expect("seed JSON can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .arg("--vars-json")
        .arg(seeds)
        .args(["--type", "matrix=array<array<int>>", "--type", "s=stack"])
        .assert()
        .success()
        .stdout("{matrix = [[1, 2], [3, 4]], s = stack[3, 2, 1]}\n");
}

#[test]
fn vars_json_rejects_duplicates_with_var() {
    let root = workspace_root();
    let example = root.join("examples/skip.rev");
    let seeds = root.join("target/test-vars-json-duplicate.json");
    std::fs::write(&seeds, r#"{"x":1}"#).expect("seed JSON can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .arg("--vars-json")
        .arg(seeds)
        .args(["--var", "x=2"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "duplicate --var annotation for `x`",
        ));
}

#[test]
fn vars_json_rejects_float_seed_value() {
    let root = workspace_root();
    let example = root.join("examples/skip.rev");
    let seeds = root.join("target/test-vars-json-float.json");
    std::fs::write(&seeds, r#"{"x":1.5}"#).expect("seed JSON can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(example)
        .arg("--vars-json")
        .arg(seeds)
        .assert()
        .failure()
        .stderr(predicate::str::contains("--vars-json"))
        .stderr(predicate::str::contains(
            "floating-point JSON numbers are not supported",
        ));
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
fn reversible_preprocess_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/reversible_preprocess.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "features = [1073741824, -1073741824, 1610612736, 536870912]",
        ))
        .stdout(predicate::str::contains(
            "raw = [2147483648, 0, 1073741824, -536870912]",
        ));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "features=[1073741824,-1073741824,1610612736,536870912]",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("features = [0, 0, 0, 0]"))
        .stdout(predicate::str::contains(
            "raw = [2147483648, 0, 1073741824, -536870912]",
        ));
}

#[test]
fn reversible_normalize_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/reversible_normalize.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "features = [536870912, 1073741824, 536870912, 1073741824]",
        ));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "features=[536870912,1073741824,536870912,1073741824]",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("features = [0, 0, 0, 0]"))
        .stdout(predicate::str::contains(
            "raw = [2147483648, 1073741824, -1073741824, 0]",
        ));
}

#[test]
fn reversible_clamp_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/reversible_clamp.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "clipped = [-1073741824, -268435456, 536870912, 1073741824]",
        ))
        .stdout(predicate::str::contains(
            "residual = [-1073741824, 0, 0, 1073741824]",
        ));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "clipped=[-1073741824,-268435456,536870912,1073741824]",
            "--var",
            "residual=[-1073741824,0,0,1073741824]",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("clipped = [0, 0, 0, 0]"))
        .stdout(predicate::str::contains("residual = [0, 0, 0, 0]"))
        .stdout(predicate::str::contains(
            "raw = [-2147483648, -268435456, 536870912, 2147483648]",
        ));
}

#[test]
fn reversible_pack_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/reversible_pack.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains("packed = 77"))
        .stdout(predicate::str::contains(
            "unpacked = [1, 0, 1, 1, 0, 0, 1, 0]",
        ));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args(["--var", "packed=77", "--var", "unpacked=[1,0,1,1,0,0,1,0]"])
        .assert()
        .success()
        .stdout(predicate::str::contains("packed = 0"))
        .stdout(predicate::str::contains(
            "unpacked = [0, 0, 0, 0, 0, 0, 0, 0]",
        ))
        .stdout(predicate::str::contains("flags = [1, 0, 1, 1, 0, 0, 1, 0]"));
}

#[test]
fn reversible_inference_trace_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/reversible_inference_trace.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "features = [1073741824, -1073741824, 1610612736, 536870912]",
        ))
        .stdout(predicate::str::contains(
            "logits = [1073741824, 1610612736, 536870912]",
        ))
        .stdout(predicate::str::contains("top_classes = [1, 0, 2]"))
        .stdout(predicate::str::contains(
            "top_logit_values = [1610612736, 1073741824, 536870912]",
        ))
        .stdout(predicate::str::contains("prediction = 1"))
        .stdout(predicate::str::contains("runner_up_class = 0"))
        .stdout(predicate::str::contains("margin = 536870912"))
        .stdout(predicate::str::contains("label_rank = 1"))
        .stdout(predicate::str::contains("correct = 1"))
        .stdout(predicate::str::contains("top2_correct = 1"));

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("reverse")
        .arg(example)
        .args([
            "--var",
            "features=[1073741824,-1073741824,1610612736,536870912]",
            "--var",
            "logits=[1073741824,1610612736,536870912]",
            "--var",
            "top_classes=[1,0,2]",
            "--var",
            "top_logit_values=[1610612736,1073741824,536870912]",
            "--var",
            "prediction=1",
            "--var",
            "runner_up_class=0",
            "--var",
            "margin=536870912",
            "--var",
            "label_rank=1",
            "--var",
            "correct=1",
            "--var",
            "top2_correct=1",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("features = [0, 0, 0, 0]"))
        .stdout(predicate::str::contains("logits = [0, 0, 0]"))
        .stdout(predicate::str::contains("top_classes = [0, 0, 0]"))
        .stdout(predicate::str::contains("top_logit_values = [0, 0, 0]"))
        .stdout(predicate::str::contains("prediction = 0"))
        .stdout(predicate::str::contains("runner_up_class = 0"))
        .stdout(predicate::str::contains("margin = 0"))
        .stdout(predicate::str::contains("label_rank = 0"))
        .stdout(predicate::str::contains("correct = 0"))
        .stdout(predicate::str::contains("top2_correct = 0"))
        .stdout(predicate::str::contains(
            "raw = [2147483648, 0, 1073741824, -536870912]",
        ));
}

#[test]
fn janus_factor_example_runs_forward_and_backward() {
    let example = workspace_root().join("examples/janus_factor.rev");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("run")
        .arg(&example)
        .args(["--var", "num=840"])
        .assert()
        .success()
        .stdout("{fact = [0, 2, 2, 2, 3, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], i = 0, num = 0, try = 0, z = 0}\n");

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
            "fact=[0,2,2,2,3,5,7,0,0,0,0,0,0,0,0,0,0,0,0,0]",
        ])
        .assert()
        .success()
        .stdout("{fact = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], i = 0, num = 840, try = 0, z = 0}\n");
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
fn scrub_dump_accepts_vars_json_seed_file() {
    let root = workspace_root();
    let example = root.join("examples/fib.rev");
    let seeds = root.join("target/test-vars-json-scrub.json");
    std::fs::write(&seeds, r#"{"n":2,"i":0,"a":0,"b":1}"#).expect("seed JSON can be written");

    Command::cargo_bin("reverie")
        .expect("binary exists")
        .arg("scrub")
        .arg(example)
        .arg("--vars-json")
        .arg(seeds)
        .args(["--watch", "a", "--watch", "i", "--dump"])
        .assert()
        .success()
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

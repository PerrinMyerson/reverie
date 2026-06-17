use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand, ValueEnum};
use reverie_interp::{ExecutionOptions, IoState, RuntimeError, State, Value};
use reverie_syntax::{
    BinaryOp, Expr, ParseOptions, Program, Spanned, SpannedExpr, SpannedStmt, SpannedType, Stmt,
    SyntaxDiagnostic, TypeExpr, UnaryOp, format_program, format_type_expr,
    parse_program_with_options, parse_type_expr,
};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Parser)]
#[command(name = "reverie")]
#[command(about = "A tiny reversible language you can scrub like a tape.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Parse and check a Reverie source file.
    Check {
        file: PathBuf,

        /// Annotate an external variable for static checking, e.g. --type distance=int<m>.
        #[arg(long = "type", value_name = "NAME=TYPE", value_parser = parse_type_arg)]
        types: Vec<TypeArg>,

        /// Use legacy Janus parsing and compatibility update semantics.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Print the mechanically derived inverse source program.
    Invert {
        file: PathBuf,

        /// Annotate an external variable for static checking, e.g. --type distance=int<m>.
        #[arg(long = "type", value_name = "NAME=TYPE", value_parser = parse_type_arg)]
        types: Vec<TypeArg>,

        /// Use legacy Janus parsing and compatibility update semantics.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Explain the reversible constructs used by a source file.
    Explain {
        file: PathBuf,

        /// Annotate an external variable for static checking, e.g. --type distance=int<m>.
        #[arg(long = "type", value_name = "NAME=TYPE", value_parser = parse_type_arg)]
        types: Vec<TypeArg>,

        /// Include an ML-oriented tensor/witness/Q31 audit profile.
        #[arg(long)]
        ml: bool,

        /// Emit a machine-readable JSON summary.
        #[arg(long)]
        json: bool,

        /// Use legacy Janus parsing and compatibility update semantics.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Print canonical parser-compatible Reverie source.
    Fmt {
        file: PathBuf,

        /// Exit non-zero if the source is not already canonical.
        #[arg(long)]
        check: bool,

        /// Rewrite the source file in canonical form.
        #[arg(long)]
        write: bool,

        /// Use legacy Janus parsing and compatibility update semantics.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Run a Reverie source file.
    Run {
        file: PathBuf,

        /// Seed a variable before execution, e.g. --var n=7 or --var xs=[1,2,3].
        #[arg(long = "var", value_name = "NAME=VALUE", value_parser = parse_var)]
        vars: Vec<VarArg>,

        /// Seed variables from a JSON object, e.g. --vars-json seeds.json.
        #[arg(long = "vars-json", value_name = "PATH")]
        vars_json: Vec<PathBuf>,

        /// Annotate an external variable for static checking, e.g. --type distance=int<m>.
        #[arg(long = "type", value_name = "NAME=TYPE", value_parser = parse_type_arg)]
        types: Vec<TypeArg>,

        /// Select the execution engine.
        #[arg(long, value_enum, default_value_t = Engine::Slot)]
        engine: Engine,

        /// Emit the final store and tape state as JSON.
        #[arg(long)]
        json: bool,

        /// Seed a reversible input tape value for `read`.
        #[arg(long = "input", value_name = "VALUE", value_parser = parse_value)]
        inputs: Vec<Value>,

        /// Seed a reversible output tape value for inverse I/O operations.
        #[arg(long = "output", value_name = "VALUE", value_parser = parse_value)]
        outputs: Vec<Value>,

        /// Use legacy Janus parsing and compatibility update semantics.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Run a Reverie source file backward from a seeded store.
    Reverse {
        file: PathBuf,

        /// Seed a variable before backward execution, e.g. --var n=7 or --var xs=[1,2,3].
        #[arg(long = "var", value_name = "NAME=VALUE", value_parser = parse_var)]
        vars: Vec<VarArg>,

        /// Seed variables from a JSON object, e.g. --vars-json seeds.json.
        #[arg(long = "vars-json", value_name = "PATH")]
        vars_json: Vec<PathBuf>,

        /// Annotate an external variable for static checking, e.g. --type distance=int<m>.
        #[arg(long = "type", value_name = "NAME=TYPE", value_parser = parse_type_arg)]
        types: Vec<TypeArg>,

        /// Select the execution engine.
        #[arg(long, value_enum, default_value_t = Engine::Slot)]
        engine: Engine,

        /// Emit the final store and tape state as JSON.
        #[arg(long)]
        json: bool,

        /// Seed a reversible input tape value for inverse I/O operations.
        #[arg(long = "input", value_name = "VALUE", value_parser = parse_value)]
        inputs: Vec<Value>,

        /// Seed a reversible output tape value for `unread`/`unwrite`.
        #[arg(long = "output", value_name = "VALUE", value_parser = parse_value)]
        outputs: Vec<Value>,

        /// Use legacy Janus parsing and compatibility update semantics.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Run forward, then backward, and emit a fingerprinted restoration proof.
    Roundtrip {
        file: PathBuf,

        /// Seed a variable before execution, e.g. --var n=7 or --var xs=[1,2,3].
        #[arg(long = "var", value_name = "NAME=VALUE", value_parser = parse_var)]
        vars: Vec<VarArg>,

        /// Seed variables from a JSON object, e.g. --vars-json seeds.json.
        #[arg(long = "vars-json", value_name = "PATH")]
        vars_json: Vec<PathBuf>,

        /// Annotate an external variable for static checking, e.g. --type distance=int<m>.
        #[arg(long = "type", value_name = "NAME=TYPE", value_parser = parse_type_arg)]
        types: Vec<TypeArg>,

        /// Select the execution engine.
        #[arg(long, value_enum, default_value_t = Engine::Slot)]
        engine: Engine,

        /// Emit a machine-readable roundtrip proof.
        #[arg(long)]
        json: bool,

        /// Write the roundtrip proof JSON to a file.
        #[arg(long = "proof-output", value_name = "PATH")]
        proof_output: Option<PathBuf>,

        /// Seed a reversible input tape value for `read`.
        #[arg(long = "input", value_name = "VALUE", value_parser = parse_value)]
        inputs: Vec<Value>,

        /// Seed a reversible output tape value for inverse I/O operations.
        #[arg(long = "output", value_name = "VALUE", value_parser = parse_value)]
        outputs: Vec<Value>,

        /// Use legacy Janus parsing and compatibility update semantics.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Verify and replay a saved roundtrip proof.
    VerifyRoundtrip {
        proof: PathBuf,

        /// Emit a machine-readable verification report.
        #[arg(long)]
        json: bool,

        /// Write a human-readable Markdown verification card.
        #[arg(long = "markdown-output", value_name = "PATH")]
        markdown_output: Option<PathBuf>,
    },

    /// Scrub a Reverie source file through its forward timeline.
    Scrub {
        file: PathBuf,

        /// Seed a variable before execution, e.g. --var n=7 or --var xs=[1,2,3].
        #[arg(long = "var", value_name = "NAME=VALUE", value_parser = parse_var)]
        vars: Vec<VarArg>,

        /// Seed variables from a JSON object, e.g. --vars-json seeds.json.
        #[arg(long = "vars-json", value_name = "PATH")]
        vars_json: Vec<PathBuf>,

        /// Annotate an external variable for static checking, e.g. --type distance=int<m>.
        #[arg(long = "type", value_name = "NAME=TYPE", value_parser = parse_type_arg)]
        types: Vec<TypeArg>,

        /// Watch a variable's value timeline. May be repeated.
        #[arg(long = "watch", value_name = "NAME")]
        watches: Vec<String>,

        /// Print the timeline instead of opening the interactive TUI.
        #[arg(long)]
        dump: bool,

        /// Seed a reversible input tape value for `read`.
        #[arg(long = "input", value_name = "VALUE", value_parser = parse_value)]
        inputs: Vec<Value>,

        /// Seed a reversible output tape value for inverse I/O operations.
        #[arg(long = "output", value_name = "VALUE", value_parser = parse_value)]
        outputs: Vec<Value>,

        /// Use legacy Janus parsing and compatibility update semantics.
        #[arg(long)]
        legacy_janus: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VarArg {
    name: String,
    value: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TypeArg {
    name: String,
    ty: SpannedType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Engine {
    /// Correctness-first AST interpreter.
    Tree,

    /// Slot-indexed compiled interpreter.
    Slot,
}

#[derive(Debug, Error)]
enum CliError {
    #[error("failed to read `{path}`: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("failed to write `{path}`: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("failed to report diagnostics: {0}")]
    Report(#[from] io::Error),

    #[error("source contains reported diagnostics")]
    ReportedDiagnostics,

    #[error(transparent)]
    Runtime(#[from] reverie_interp::RuntimeError),

    #[error(transparent)]
    Tui(#[from] reverie_tui::TuiError),

    #[error("{0}")]
    SeedType(String),
}

fn main() -> ExitCode {
    match run_cli(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::ReportedDiagnostics) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run_cli(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Command::Check {
            file,
            mut types,
            legacy_janus,
        } => {
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            normalize_legacy_seed_types(legacy_janus, &mut types);
            validate_type_args(&types)?;
            validate_external_types(&file, &program, &types)?;
            check_or_report(&file, &source, &program, &types, legacy_janus)?;
            println!("ok: {}", file.display());
            Ok(())
        }
        Command::Invert {
            file,
            mut types,
            legacy_janus,
        } => {
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            normalize_legacy_seed_types(legacy_janus, &mut types);
            validate_type_args(&types)?;
            validate_external_types(&file, &program, &types)?;
            check_or_report(&file, &source, &program, &types, legacy_janus)?;
            let inverse = reverie_core::invert_program(&program);
            print!("{}", format_program(&inverse));
            Ok(())
        }
        Command::Explain {
            file,
            mut types,
            ml,
            json,
            legacy_janus,
        } => {
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            normalize_legacy_seed_types(legacy_janus, &mut types);
            validate_type_args(&types)?;
            validate_external_types(&file, &program, &types)?;
            check_or_report(&file, &source, &program, &types, legacy_janus)?;
            if json {
                print_explanation_json(&file, &program, &types, ml);
            } else {
                print_explanation(&file, &program, &types, ml);
            }
            Ok(())
        }
        Command::Fmt {
            file,
            check,
            write,
            legacy_janus,
        } => {
            if check && write {
                return Err(CliError::SeedType(
                    "`reverie fmt` cannot combine --check and --write".to_owned(),
                ));
            }
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            let formatted = format_program(&program);
            if check {
                if source == formatted {
                    println!("ok: {}", file.display());
                    Ok(())
                } else {
                    Err(CliError::SeedType(format!(
                        "{} is not canonical; run `reverie fmt {}`",
                        file.display(),
                        shell_quote_arg(&file.display().to_string())
                    )))
                }
            } else if write {
                if source != formatted {
                    fs::write(&file, formatted).map_err(|source| CliError::Write {
                        path: file.clone(),
                        source,
                    })?;
                }
                println!("ok: {}", file.display());
                Ok(())
            } else {
                print!("{formatted}");
                Ok(())
            }
        }
        Command::Run {
            file,
            vars: cli_vars,
            vars_json,
            types,
            engine,
            json,
            inputs,
            outputs,
            legacy_janus,
        } => {
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            let mut vars = load_vars_json_files(&vars_json)?;
            vars.extend(cli_vars);
            let mut types = types;
            normalize_legacy_seed_args(legacy_janus, &mut vars, &mut types);
            validate_seed_args(&vars, &types)?;
            validate_tape_value_shapes("--input", &inputs)?;
            validate_tape_value_shapes("--output", &outputs)?;
            validate_external_types(&file, &program, &types)?;
            validate_external_seeds(&file, &program, &vars)?;
            let types = types_with_seed_inference(&vars, &types);
            validate_seed_types(&vars, &types)?;
            check_or_report(&file, &source, &program, &types, legacy_janus)?;
            let initial_state = state_from_vars(vars);
            let initial_io = IoState::with_output(initial_state, inputs, outputs);
            let options = execution_options(legacy_janus);
            let state = runtime_or_report(&file, &source, || match engine {
                Engine::Tree => {
                    reverie_interp::execute_io_with_options(&program, initial_io, options)
                }
                Engine::Slot => {
                    reverie_interp::execute_compiled_io_with_options(&program, initial_io, options)
                }
            })?;
            let metadata = json.then(|| {
                run_result_metadata(
                    RunProvenance {
                        file: &file,
                        source: &source,
                        direction: "run",
                        engine,
                        legacy_janus,
                    },
                    &program,
                    &types,
                    &state,
                )
            });
            print_io_state(&state, json, "reverie_run_result", metadata);
            Ok(())
        }
        Command::Reverse {
            file,
            vars: cli_vars,
            vars_json,
            types,
            engine,
            json,
            inputs,
            outputs,
            legacy_janus,
        } => {
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            let mut vars = load_vars_json_files(&vars_json)?;
            vars.extend(cli_vars);
            let mut types = types;
            normalize_legacy_seed_args(legacy_janus, &mut vars, &mut types);
            validate_seed_args(&vars, &types)?;
            validate_tape_value_shapes("--input", &inputs)?;
            validate_tape_value_shapes("--output", &outputs)?;
            validate_external_types(&file, &program, &types)?;
            validate_external_seeds(&file, &program, &vars)?;
            let types = types_with_seed_inference(&vars, &types);
            validate_seed_types(&vars, &types)?;
            check_or_report(&file, &source, &program, &types, legacy_janus)?;
            let initial_state = state_from_vars(vars);
            let initial_io = IoState::with_output(initial_state, inputs, outputs);
            let options = execution_options(legacy_janus);
            let state = runtime_or_report(&file, &source, || match engine {
                Engine::Tree => {
                    reverie_interp::execute_io_backward_with_options(&program, initial_io, options)
                }
                Engine::Slot => reverie_interp::execute_compiled_io_backward_with_options(
                    &program, initial_io, options,
                ),
            })?;
            let metadata = json.then(|| {
                run_result_metadata(
                    RunProvenance {
                        file: &file,
                        source: &source,
                        direction: "reverse",
                        engine,
                        legacy_janus,
                    },
                    &program,
                    &types,
                    &state,
                )
            });
            print_io_state(&state, json, "reverie_reverse_result", metadata);
            Ok(())
        }
        Command::Roundtrip {
            file,
            vars: cli_vars,
            vars_json,
            types,
            engine,
            json,
            proof_output,
            inputs,
            outputs,
            legacy_janus,
        } => {
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            let mut vars = load_vars_json_files(&vars_json)?;
            vars.extend(cli_vars);
            let mut types = types;
            normalize_legacy_seed_args(legacy_janus, &mut vars, &mut types);
            validate_seed_args(&vars, &types)?;
            validate_tape_value_shapes("--input", &inputs)?;
            validate_tape_value_shapes("--output", &outputs)?;
            validate_external_types(&file, &program, &types)?;
            validate_external_seeds(&file, &program, &vars)?;
            let types = types_with_seed_inference(&vars, &types);
            validate_seed_types(&vars, &types)?;
            check_or_report(&file, &source, &program, &types, legacy_janus)?;

            let initial_state = state_from_vars(vars);
            let initial_io = IoState::with_output(initial_state, inputs, outputs);
            let options = execution_options(legacy_janus);
            let baseline = runtime_or_report(&file, &source, || {
                execute_initialization_io(&program, initial_io.clone(), engine, options)
            })?;
            let forward = runtime_or_report(&file, &source, || match engine {
                Engine::Tree => {
                    reverie_interp::execute_io_with_options(&program, initial_io.clone(), options)
                }
                Engine::Slot => reverie_interp::execute_compiled_io_with_options(
                    &program,
                    initial_io.clone(),
                    options,
                ),
            })?;
            let restored = runtime_or_report(&file, &source, || match engine {
                Engine::Tree => reverie_interp::execute_io_backward_with_options(
                    &program,
                    forward.clone(),
                    options,
                ),
                Engine::Slot => reverie_interp::execute_compiled_io_backward_with_options(
                    &program,
                    forward.clone(),
                    options,
                ),
            })?;

            let summary = summarize_program(&program);
            let witness_report = witness_report(&summary, &types);
            let check = roundtrip_check(&baseline, &restored, &summary);
            let proof = roundtrip_proof_json(RoundtripProofInput {
                file: &file,
                source: &source,
                engine,
                legacy_janus,
                types: &types,
                summary: &summary,
                witness_report: &witness_report,
                baseline: &baseline,
                forward: &forward,
                restored: &restored,
                check: &check,
                include_ml_profile: true,
            });

            if let Some(path) = &proof_output {
                write_json_file(path, &proof)?;
            }

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&proof).expect("roundtrip proof JSON serializes")
                );
            } else {
                print_roundtrip_proof_text(RoundtripProofText {
                    file: &file,
                    engine,
                    legacy_janus,
                    baseline: &baseline,
                    forward: &forward,
                    restored: &restored,
                    witness_report: &witness_report,
                    proof: &proof,
                    check: &check,
                    proof_output: proof_output.as_deref(),
                });
            }

            if check.passed {
                Ok(())
            } else {
                Err(CliError::SeedType(
                    "roundtrip proof failed to restore the entry store or tapes".to_owned(),
                ))
            }
        }
        Command::VerifyRoundtrip {
            proof,
            json,
            markdown_output,
        } => {
            let proof_json = read_json_file(&proof)?;
            let verification = verify_roundtrip_proof(&proof, &proof_json)?;
            if let Some(path) = &markdown_output {
                write_text_file(
                    path,
                    &render_roundtrip_verification_markdown(&proof_json, &verification),
                )?;
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&verification)
                        .expect("roundtrip verification JSON serializes")
                );
            } else {
                print_roundtrip_verification_text(
                    &proof,
                    &verification,
                    markdown_output.as_deref(),
                );
            }
            if verification["passed"].as_bool() == Some(true) {
                Ok(())
            } else {
                Err(CliError::SeedType(format!(
                    "roundtrip proof {} failed verification",
                    proof.display()
                )))
            }
        }
        Command::Scrub {
            file,
            vars: cli_vars,
            vars_json,
            types,
            watches,
            dump,
            inputs,
            outputs,
            legacy_janus,
        } => {
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            let mut vars = load_vars_json_files(&vars_json)?;
            vars.extend(cli_vars);
            let mut types = types;
            normalize_legacy_seed_args(legacy_janus, &mut vars, &mut types);
            validate_seed_args(&vars, &types)?;
            validate_watch_args(&watches)?;
            validate_tape_value_shapes("--input", &inputs)?;
            validate_tape_value_shapes("--output", &outputs)?;
            validate_external_types(&file, &program, &types)?;
            validate_external_seeds(&file, &program, &vars)?;
            let types = types_with_seed_inference(&vars, &types);
            validate_seed_types(&vars, &types)?;
            check_or_report(&file, &source, &program, &types, legacy_janus)?;
            let initial_state = state_from_vars(vars);
            let initial_io = IoState::with_output(initial_state, inputs, outputs);
            let options = execution_options(legacy_janus);
            let timeline = runtime_or_report(&file, &source, || {
                reverie_interp::build_timeline_io_with_options(&program, initial_io, options)
            })?;
            validate_watches_exist(&timeline, &watches)?;

            if dump {
                println!(
                    "{}",
                    reverie_tui::dump_timeline_with_watches(&timeline, &watches)
                );
            } else {
                reverie_tui::run_scrubber(file.display().to_string(), source, timeline, watches)?;
            }

            Ok(())
        }
    }
}

#[derive(Debug, Default)]
struct ProgramSummary {
    statements: usize,
    expressions: usize,
    features: BTreeSet<&'static str>,
    safety_checks: BTreeSet<&'static str>,
    safety_check_counts: BTreeMap<&'static str, usize>,
    dataset_loops: Vec<DatasetLoopSummary>,
    declared_names: BTreeSet<String>,
    declared_templates: BTreeMap<String, String>,
    tensor_names: BTreeSet<String>,
    tensor_measures: BTreeMap<String, WitnessMeasure>,
    witness_names: BTreeSet<String>,
    witness_measures: BTreeMap<String, WitnessMeasure>,
    external_names: BTreeSet<String>,
    update_targets: BTreeMap<String, usize>,
    tensor_builtin_counts: BTreeMap<String, usize>,
    lossy_signal_counts: BTreeMap<String, usize>,
    q31_builtin_calls: usize,
    fixed_point_binary_ops: usize,
}

#[derive(Debug, Clone)]
struct DatasetLoopSummary {
    index: String,
    size_sources: BTreeSet<String>,
}

#[derive(Debug, Clone, Copy)]
struct WitnessMeasure {
    cells: Option<usize>,
    payload_bytes: Option<usize>,
}

#[derive(Debug)]
struct WitnessReport {
    names: BTreeSet<String>,
    measures: BTreeMap<String, WitnessMeasure>,
    known_cells: usize,
    known_payload_bytes: usize,
    unknown_names: Vec<String>,
}

fn print_explanation(file: &Path, program: &Program, types: &[TypeArg], ml: bool) {
    let summary = summarize_program(program);
    let witness_report = witness_report(&summary, types);
    let features = features_with_witness(&summary, &witness_report);

    println!("file: {}", file.display());
    println!("status: reversible program checks");
    println!("globals: {}", program.globals.len());
    println!("procedures: {}", program.procedures.len());
    println!("statements: {}", summary.statements);
    println!("expressions: {}", summary.expressions);
    println!("features:");
    if features.is_empty() {
        println!("- skip-only core");
    } else {
        for feature in &features {
            println!("- {feature}");
        }
    }
    println!("safety checks:");
    if summary.safety_checks.is_empty() {
        println!("- no additional indexed or arithmetic runtime checks");
    } else {
        for check in &summary.safety_checks {
            println!("- {check}");
        }
    }
    println!("dataset loops:");
    if summary.dataset_loops.is_empty() {
        println!("- none");
    } else {
        for loop_summary in &summary.dataset_loops {
            println!(
                "- {}: bound by {}",
                loop_summary.index,
                loop_summary
                    .size_sources
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
    }
    println!("external store:");
    if summary.external_names.is_empty() {
        println!("- none");
    } else {
        for name in &summary.external_names {
            let annotation = types
                .iter()
                .find(|ty| ty.name == *name)
                .map(|ty| format_type_expr(&ty.ty.node))
                .unwrap_or_else(|| {
                    format!("inferred at runtime; add --type {name}=TYPE for units")
                });
            println!("- {name}: {annotation}");
        }
        println!(
            "run template: reverie run {} {}",
            shell_quote_arg(&file.display().to_string()),
            run_template_args(&summary.external_names, types)
        );
    }
    println!("declared store:");
    if summary.declared_names.is_empty() {
        println!("- none");
    } else {
        for name in &summary.declared_names {
            let value = summary
                .declared_templates
                .get(name)
                .map(String::as_str)
                .unwrap_or("0");
            println!("- {name}: optional --var override; default template {value}");
        }
        println!(
            "declared override template: reverie run {} {}",
            shell_quote_arg(&file.display().to_string()),
            declared_template_args(&summary)
        );
    }
    println!("witness store:");
    if witness_report.names.is_empty() {
        println!("- none");
    } else {
        for name in &witness_report.names {
            let measure = witness_report.measures.get(name);
            println!(
                "- {name}: cells={} payload_bytes={}",
                format_optional_usize(measure.and_then(|measure| measure.cells)),
                format_optional_usize(measure.and_then(|measure| measure.payload_bytes))
            );
        }
    }
    println!(
        "witness proof cost: variables={} known_cells={} known_payload_bytes={} unknown_variables={}",
        witness_report.names.len(),
        witness_report.known_cells,
        witness_report.known_payload_bytes,
        if witness_report.unknown_names.is_empty() {
            "none".to_owned()
        } else {
            witness_report.unknown_names.join(",")
        }
    );
    if ml {
        print_ml_profile(&summary, &witness_report, types);
    }
    println!(
        "inverse: reverie invert {}",
        shell_quote_arg(&file.display().to_string())
    );
}

fn print_explanation_json(file: &Path, program: &Program, types: &[TypeArg], ml: bool) {
    let summary = summarize_program(program);
    let witness_report = witness_report(&summary, types);
    let features = features_with_witness(&summary, &witness_report);
    let file = file.display().to_string();
    let external_store = summary
        .external_names
        .iter()
        .map(|name| {
            let annotation = types
                .iter()
                .find(|ty| ty.name == *name)
                .map(|ty| format_type_expr(&ty.ty.node));
            format!(
                "{{\"name\":{},\"type\":{},\"template\":{}}}",
                json_string(name),
                json_option_string(annotation.as_deref()),
                json_string(
                    &types
                        .iter()
                        .find(|ty| ty.name == *name)
                        .map(|ty| template_value_for_type(&ty.ty.node))
                        .unwrap_or_else(|| "0".to_owned())
                )
            )
        })
        .collect::<Vec<_>>();
    let declared_store = summary
        .declared_names
        .iter()
        .map(|name| {
            let value = summary
                .declared_templates
                .get(name)
                .map(String::as_str)
                .unwrap_or("0");
            format!(
                "{{\"name\":{},\"template\":{}}}",
                json_string(name),
                json_string(value)
            )
        })
        .collect::<Vec<_>>();
    let witness_store = witness_report
        .names
        .iter()
        .map(|name| json_string(name))
        .collect::<Vec<_>>();
    let witness_entries = witness_report
        .names
        .iter()
        .map(|name| {
            let measure = witness_report.measures.get(name);
            format!(
                "{{\"name\":{},\"cells\":{},\"payload_bytes\":{}}}",
                json_string(name),
                json_option_usize(measure.and_then(|measure| measure.cells)),
                json_option_usize(measure.and_then(|measure| measure.payload_bytes))
            )
        })
        .collect::<Vec<_>>();

    println!("{{");
    println!("  \"file\": {},", json_string(&file));
    println!("  \"status\": \"reversible program checks\",");
    println!("  \"globals\": {},", program.globals.len());
    println!("  \"procedures\": {},", program.procedures.len());
    println!("  \"statements\": {},", summary.statements);
    println!("  \"expressions\": {},", summary.expressions);
    println!(
        "  \"features\": {},",
        json_string_array(features.iter().copied())
    );
    println!(
        "  \"safety_checks\": {},",
        json_string_array(summary.safety_checks.iter().copied())
    );
    println!(
        "  \"safety_check_counts\": {},",
        json_usize_object(
            summary
                .safety_check_counts
                .iter()
                .map(|(key, value)| (*key, *value))
        )
    );
    println!(
        "  \"dataset_loops\": {},",
        json_dataset_loops(&summary.dataset_loops)
    );
    println!("  \"external_store\": [{}],", external_store.join(","));
    println!("  \"declared_store\": [{}],", declared_store.join(","));
    println!("  \"witness_store\": [{}],", witness_store.join(","));
    println!(
        "  \"witness_metrics\": {{\"variables\":{},\"known_cells\":{},\"known_payload_bytes\":{},\"unknown_variables\":{},\"entries\":[{}]}},",
        witness_report.names.len(),
        witness_report.known_cells,
        witness_report.known_payload_bytes,
        json_string_array(witness_report.unknown_names.iter().map(String::as_str)),
        witness_entries.join(",")
    );
    if ml {
        println!(
            "  \"ml_profile\": {},",
            ml_profile_json(&summary, &witness_report, types)
        );
    }
    println!(
        "  \"run_template\": {},",
        json_string(&command_template(
            &format!("reverie run {}", shell_quote_arg(&file)),
            &run_template_args(&summary.external_names, types)
        ))
    );
    println!(
        "  \"declared_override_template\": {},",
        json_string(&command_template(
            &format!("reverie run {}", shell_quote_arg(&file)),
            &declared_template_args(&summary)
        ))
    );
    println!(
        "  \"inverse_template\": {}",
        json_string(&format!("reverie invert {}", shell_quote_arg(&file)))
    );
    println!("}}");
}

fn command_template(prefix: &str, args: &str) -> String {
    if args.is_empty() {
        prefix.to_owned()
    } else {
        format!("{prefix} {args}")
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for char in value.chars() {
        match char {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            char if char.is_control() => out.push_str(&format!("\\u{:04x}", char as u32)),
            char => out.push(char),
        }
    }
    out.push('"');
    out
}

fn json_option_string(value: Option<&str>) -> String {
    value.map(json_string).unwrap_or_else(|| "null".to_owned())
}

fn json_option_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_owned())
}

fn format_optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn json_string_array<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(json_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_usize_object<'a>(entries: impl IntoIterator<Item = (&'a str, usize)>) -> String {
    format!(
        "{{{}}}",
        entries
            .into_iter()
            .map(|(key, value)| format!("{}:{}", json_string(key), value))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_dataset_loops(loops: &[DatasetLoopSummary]) -> String {
    dataset_loops_json(loops).to_string()
}

fn dataset_loops_json(loops: &[DatasetLoopSummary]) -> JsonValue {
    json!(
        loops
            .iter()
            .map(|loop_summary| {
                json!({
                    "index": loop_summary.index.clone(),
                    "size_sources": loop_summary
                        .size_sources
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>()
    )
}

fn summarize_program(program: &Program) -> ProgramSummary {
    let mut summary = ProgramSummary::default();
    let global_names = program
        .globals
        .iter()
        .map(|global| global.name.clone())
        .collect::<BTreeSet<_>>();
    if !program.globals.is_empty() {
        summary.features.insert("source globals");
    }
    for global in &program.globals {
        summary.declared_names.insert(global.name.clone());
        summary.declared_templates.insert(
            global.name.clone(),
            declared_template_value(&global.dims, global.ty.as_ref().map(|ty| &ty.node)),
        );
        if let Some(ty) = &global.ty {
            let dims = storage_dims(global.len, &global.dims);
            record_tensor_measure(&mut summary, &global.name, &dims, &ty.node);
            record_witness_measure(&mut summary, &global.name, &dims, &ty.node);
        }
        if global.len > 1 || !global.dims.is_empty() {
            summary.features.insert("fixed-size arrays");
        }
        if global.init.is_some() {
            summary.features.insert("source declarations");
        }
    }
    for procedure in &program.procedures {
        let mut scope = global_names.clone();
        for param in &procedure.params {
            if param
                .ty
                .as_ref()
                .is_some_and(|ty| type_contains_witness(&ty.node))
            {
                summary.features.insert("witness tapes");
            }
        }
        scope.extend(procedure.params.iter().map(|param| param.name.clone()));
        visit_stmt(&procedure.body, &mut scope, &mut summary);
    }
    let mut scope = global_names;
    visit_stmt(&program.body, &mut scope, &mut summary);

    summary
}

fn features_with_witness<'a>(
    summary: &'a ProgramSummary,
    witness_report: &WitnessReport,
) -> BTreeSet<&'a str> {
    let mut features = summary.features.iter().copied().collect::<BTreeSet<_>>();
    if !witness_report.names.is_empty() {
        features.insert("witness tapes");
    }
    features
}

fn run_template_args(external_names: &BTreeSet<String>, types: &[TypeArg]) -> String {
    let var_args = external_names
        .iter()
        .map(|name| {
            let value = types
                .iter()
                .find(|ty| ty.name == *name)
                .map(|ty| template_value_for_type(&ty.ty.node))
                .unwrap_or_else(|| "0".to_owned());
            format!("--var {name}={}", shell_quote_arg(&value))
        })
        .collect::<Vec<_>>();
    let type_args = types
        .iter()
        .filter(|ty| external_names.contains(&ty.name))
        .map(|ty| {
            format!(
                "--type {}",
                shell_quote_arg(&format!("{}={}", ty.name, format_type_expr(&ty.ty.node)))
            )
        })
        .collect::<Vec<_>>();

    var_args
        .into_iter()
        .chain(type_args)
        .collect::<Vec<_>>()
        .join(" ")
}

fn template_value_for_type(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Int { .. } => "0".to_owned(),
        TypeExpr::Bool => "false".to_owned(),
        TypeExpr::Stack => "nil".to_owned(),
        TypeExpr::Array { element } => format!("[{}]", template_value_for_type(&element.node)),
        TypeExpr::Tensor { element, shape } => tensor_template_value(shape, &element.node),
        TypeExpr::Witness { inner } => template_value_for_type(&inner.node),
    }
}

fn tensor_template_value(shape: &[usize], element: &TypeExpr) -> String {
    let Some((len, rest)) = shape.split_first() else {
        return template_value_for_type(element);
    };
    let element = tensor_template_value(rest, element);
    format!("[{}]", vec![element; *len].join(","))
}

fn declared_template_args(summary: &ProgramSummary) -> String {
    summary
        .declared_names
        .iter()
        .map(|name| {
            let value = summary
                .declared_templates
                .get(name)
                .map(String::as_str)
                .unwrap_or("0");
            format!("--var {name}={}", shell_quote_arg(value))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn storage_dims(len: usize, dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() && len > 1 {
        vec![len]
    } else {
        dims.to_vec()
    }
}

fn declared_template_value(dims: &[usize], ty: Option<&TypeExpr>) -> String {
    let Some((len, rest)) = dims.split_first() else {
        return ty
            .map(template_value_for_type)
            .unwrap_or_else(|| "0".to_owned());
    };
    let element_ty = match ty {
        Some(TypeExpr::Array { element }) => Some(&element.node),
        Some(TypeExpr::Tensor { element, .. }) => Some(&element.node),
        Some(TypeExpr::Witness { inner }) => Some(&inner.node),
        other => other,
    };
    let element = declared_template_value(rest, element_ty);
    format!("[{}]", vec![element; *len].join(","))
}

fn shell_quote_arg(value: &str) -> String {
    if value.chars().all(|char| {
        char.is_ascii_alphanumeric() || matches!(char, '_' | '-' | '.' | '/' | ':' | '=')
    }) {
        return value.to_owned();
    }

    format!("'{}'", value.replace('\'', "'\\''"))
}

fn visit_stmt(stmt: &SpannedStmt, scope: &mut BTreeSet<String>, summary: &mut ProgramSummary) {
    summary.statements += 1;
    match &stmt.node {
        Stmt::Skip => {}
        Stmt::Seq(statements) => {
            for statement in statements {
                visit_stmt(statement, scope, summary);
            }
        }
        Stmt::Assert { condition } => {
            summary.features.insert("assertions");
            visit_expr(condition, scope, summary);
        }
        Stmt::Update { target, expr, .. } => {
            summary.features.insert("reversible updates");
            *summary
                .update_targets
                .entry(target.name.clone())
                .or_insert(0) += 1;
            if target.is_indexed() {
                summary.features.insert("indexed mutation");
                record_safety_check(summary, "same-root update aliases rejected before runtime");
                if expr_reads_root(expr, &target.name) {
                    record_safety_check(
                        summary,
                        "same-root update reads proven disjoint before runtime",
                    );
                }
            }
            visit_place(target, scope, summary);
            visit_expr(expr, scope, summary);
        }
        Stmt::Swap { left, right } => {
            summary.features.insert("swaps");
            if left.is_indexed() || right.is_indexed() {
                summary.features.insert("indexed mutation");
            }
            visit_place(left, scope, summary);
            visit_place(right, scope, summary);
        }
        Stmt::Push { source, stack } => {
            summary.features.insert("reversible stacks");
            visit_place(source, scope, summary);
            visit_name(stack, scope, summary);
        }
        Stmt::Pop { target, stack } => {
            summary.features.insert("reversible stacks");
            visit_place(target, scope, summary);
            visit_name(stack, scope, summary);
        }
        Stmt::If {
            entry,
            then_branch,
            else_branch,
            exit,
        } => {
            summary.features.insert("reversible conditionals");
            visit_expr(entry, scope, summary);
            let mut then_scope = scope.clone();
            visit_stmt(then_branch, &mut then_scope, summary);
            let mut else_scope = scope.clone();
            visit_stmt(else_branch, &mut else_scope, summary);
            visit_expr(exit, scope, summary);
        }
        Stmt::Loop {
            entry,
            body,
            step,
            exit,
        } => {
            summary.features.insert("reversible loops");
            visit_expr(entry, scope, summary);
            let mut body_scope = scope.clone();
            visit_stmt(body, &mut body_scope, summary);
            let mut step_scope = scope.clone();
            visit_stmt(step, &mut step_scope, summary);
            visit_expr(exit, scope, summary);
        }
        Stmt::Iterate {
            name,
            start,
            step,
            end,
            body,
        } => {
            summary.features.insert("Janus iterate loops");
            let size_sources = iterate_size_sources(start, step, end);
            if !size_sources.is_empty() {
                summary.features.insert("dataset-shaped iterate loops");
                summary.dataset_loops.push(DatasetLoopSummary {
                    index: name.clone(),
                    size_sources,
                });
            }
            visit_expr(start, scope, summary);
            visit_expr(step, scope, summary);
            visit_expr(end, scope, summary);
            let mut body_scope = scope.clone();
            body_scope.insert(name.clone());
            visit_stmt(body, &mut body_scope, summary);
        }
        Stmt::Call { args, .. } | Stmt::Uncall { args, .. } => {
            summary.features.insert("procedures");
            if args.iter().any(|arg| arg.is_indexed()) {
                record_safety_check(
                    summary,
                    "same-root element argument aliases rejected before runtime",
                );
            }
            if args
                .iter()
                .any(|arg| arg.is_indexed() && !all_indices_constant(&arg.indices))
            {
                record_safety_check(
                    summary,
                    "dynamic element argument locations resolved at call entry",
                );
            }
            for arg in args {
                visit_place(arg, scope, summary);
            }
        }
        Stmt::Read { target } | Stmt::Unread { target } => {
            summary.features.insert("reversible tape I/O");
            visit_place(target, scope, summary);
        }
        Stmt::Write { source } | Stmt::Unwrite { source } => {
            summary.features.insert("reversible tape I/O");
            visit_place(source, scope, summary);
        }
        Stmt::Show { targets } => {
            summary.features.insert("observations");
            for target in targets {
                visit_name(target, scope, summary);
            }
        }
        Stmt::Printf { args, .. } => {
            summary.features.insert("observations");
            for arg in args {
                visit_expr(arg, scope, summary);
            }
        }
        Stmt::Local {
            name,
            ty,
            init,
            body,
            delocal,
            ..
        } => {
            summary.features.insert("local/delocal");
            if ty
                .as_ref()
                .is_some_and(|ty| type_contains_witness(&ty.node))
            {
                summary.features.insert("witness tapes");
            }
            visit_expr(init, scope, summary);
            let mut body_scope = scope.clone();
            body_scope.insert(name.clone());
            visit_stmt(body, &mut body_scope, summary);
            visit_expr(delocal, &mut body_scope, summary);
        }
        Stmt::Declare {
            name,
            init,
            len,
            dims,
            ty,
            ..
        } => {
            summary.features.insert("source declarations");
            if *len > 1 || !dims.is_empty() {
                summary.features.insert("fixed-size arrays");
            }
            if let Some(init) = init {
                visit_expr(init, scope, summary);
            }
            summary.declared_names.insert(name.clone());
            let storage_dims = storage_dims(*len, dims);
            summary.declared_templates.insert(
                name.clone(),
                declared_template_value(&storage_dims, Some(&ty.node)),
            );
            record_tensor_measure(summary, name, &storage_dims, &ty.node);
            record_witness_measure(summary, name, &storage_dims, &ty.node);
            scope.insert(name.clone());
        }
    }
}

fn iterate_size_sources(
    start: &SpannedExpr,
    step: &SpannedExpr,
    end: &SpannedExpr,
) -> BTreeSet<String> {
    let mut sources = BTreeSet::new();
    collect_size_sources(start, &mut sources);
    collect_size_sources(step, &mut sources);
    collect_size_sources(end, &mut sources);
    sources
}

fn collect_size_sources(expr: &SpannedExpr, sources: &mut BTreeSet<String>) {
    match &expr.node {
        Expr::Size { target } => {
            sources.insert(target.clone());
        }
        Expr::Array(elements) => {
            for element in elements {
                collect_size_sources(element, sources);
            }
        }
        Expr::Index { indices, .. } => {
            for index in indices {
                collect_size_sources(index, sources);
            }
        }
        Expr::Unary { expr, .. } => collect_size_sources(expr, sources),
        Expr::Binary { left, right, .. } => {
            collect_size_sources(left, sources);
            collect_size_sources(right, sources);
        }
        Expr::Call { args, .. } => {
            for arg in args {
                collect_size_sources(arg, sources);
            }
        }
        Expr::Int { .. }
        | Expr::Bool(_)
        | Expr::Nil
        | Expr::Var(_)
        | Expr::Empty { .. }
        | Expr::Top { .. } => {}
    }
}

fn record_tensor_measure(summary: &mut ProgramSummary, name: &str, dims: &[usize], ty: &TypeExpr) {
    if !type_contains_tensor(ty) {
        return;
    }
    summary.features.insert("array-backed tensors");
    summary.tensor_names.insert(name.to_owned());
    summary
        .tensor_measures
        .insert(name.to_owned(), witness_measure_for_storage(dims, ty));
}

fn record_witness_measure(summary: &mut ProgramSummary, name: &str, dims: &[usize], ty: &TypeExpr) {
    if !type_contains_witness(ty) {
        return;
    }
    summary.features.insert("witness tapes");
    summary.witness_names.insert(name.to_owned());
    summary
        .witness_measures
        .insert(name.to_owned(), witness_measure_for_storage(dims, ty));
}

fn witness_report(summary: &ProgramSummary, types: &[TypeArg]) -> WitnessReport {
    let mut names = summary.witness_names.clone();
    let mut measures = summary.witness_measures.clone();
    for ty in types {
        if summary.external_names.contains(&ty.name) && type_contains_witness(&ty.ty.node) {
            names.insert(ty.name.clone());
            measures.insert(
                ty.name.clone(),
                witness_measure_for_storage(&[], &ty.ty.node),
            );
        }
    }

    let mut known_cells = 0_usize;
    let mut known_payload_bytes = 0_usize;
    let mut unknown_names = Vec::new();
    for name in &names {
        let Some(measure) = measures.get(name) else {
            unknown_names.push(name.clone());
            continue;
        };
        match (measure.cells, measure.payload_bytes) {
            (Some(cells), Some(payload_bytes)) => {
                known_cells = known_cells.saturating_add(cells);
                known_payload_bytes = known_payload_bytes.saturating_add(payload_bytes);
            }
            _ => unknown_names.push(name.clone()),
        }
    }

    WitnessReport {
        names,
        measures,
        known_cells,
        known_payload_bytes,
        unknown_names,
    }
}

fn print_ml_profile(summary: &ProgramSummary, witness_report: &WitnessReport, types: &[TypeArg]) {
    let profile = ml_profile_json(summary, witness_report, types);
    let tensor_metrics = profile.get("tensor_metrics").unwrap_or(&JsonValue::Null);
    let update_counts = profile.get("update_counts").unwrap_or(&JsonValue::Null);
    let replay_cost = profile.get("replay_cost").unwrap_or(&JsonValue::Null);
    let witness_to_state_ratio = replay_cost
        .get("witness_to_state_payload_ratio")
        .and_then(JsonValue::as_f64)
        .map(|ratio| format!("{ratio:.6}"))
        .unwrap_or_else(|| "n/a".to_owned());

    println!("ml profile:");
    println!("- goal_fit: {}", json_str(&profile, "goal_fit"));
    println!(
        "- capabilities: {}",
        json_string_array_markdown(profile.get("capabilities"))
    );
    println!(
        "- tensors: variables={} state_variables={} witness_variables={} known_state_payload_bytes={} known_witness_payload_bytes={}",
        json_u64(tensor_metrics, "variables"),
        json_u64(tensor_metrics, "state_variables"),
        json_u64(tensor_metrics, "witness_variables"),
        json_u64(tensor_metrics, "known_state_payload_bytes"),
        witness_report.known_payload_bytes,
    );
    println!(
        "- updates: tensor={} witness={} dataset_loops={}",
        json_u64(update_counts, "tensor_update_statements"),
        json_u64(update_counts, "witness_update_statements"),
        json_u64(&profile, "dataset_loop_count"),
    );
    println!(
        "- builtins: tensor_calls={} q31_calls={} fixed_point_binary_ops={}",
        json_u64(&profile, "tensor_builtin_calls"),
        json_u64(&profile, "q31_builtin_calls"),
        json_u64(&profile, "fixed_point_binary_ops"),
    );
    println!(
        "- replay_cost: forward_statements={} inverse_statements={} roundtrip_statements={} replay_payload_bytes={} witness_to_state_ratio={}",
        json_u64(replay_cost, "forward_statement_count"),
        json_u64(replay_cost, "inverse_statement_count"),
        json_u64(replay_cost, "roundtrip_statement_count"),
        json_u64(replay_cost, "known_replay_payload_bytes"),
        witness_to_state_ratio,
    );
    let lossy = json_object_counts_markdown(profile.get("non_injective_signal_calls"));
    if lossy != "none" {
        println!("- non_injective_signal_calls: {lossy}");
    }
}

fn ml_profile_json(
    summary: &ProgramSummary,
    witness_report: &WitnessReport,
    types: &[TypeArg],
) -> JsonValue {
    let (tensor_names, tensor_measures) = tensor_inventory(summary, types);
    let state_tensor_names = tensor_names
        .iter()
        .filter(|name| !witness_report.names.contains(*name))
        .cloned()
        .collect::<BTreeSet<_>>();
    let (known_tensor_cells, known_tensor_payload_bytes, unknown_tensor_variables) =
        sum_known_measures(&tensor_names, &tensor_measures);
    let (
        known_state_tensor_cells,
        known_state_tensor_payload_bytes,
        unknown_state_tensor_variables,
    ) = sum_known_measures(&state_tensor_names, &tensor_measures);
    let tensor_update_statements = update_count_for_names(&summary.update_targets, &tensor_names);
    let witness_update_statements =
        update_count_for_names(&summary.update_targets, &witness_report.names);
    let tensor_builtin_calls = summary.tensor_builtin_counts.values().sum::<usize>();
    let q31_calls = summary.q31_builtin_calls;
    let uses_q31 = q31_calls > 0 || summary.fixed_point_binary_ops > 0;
    let known_replay_payload_bytes =
        known_state_tensor_payload_bytes.saturating_add(witness_report.known_payload_bytes);

    let mut capabilities = BTreeSet::new();
    capabilities.insert("static_reversibility_check");
    if !tensor_names.is_empty() {
        capabilities.insert("static_tensor_shapes");
    }
    if tensor_update_statements > 0 {
        capabilities.insert("reversible_tensor_accumulation");
    }
    if tensor_builtin_calls > 0 {
        capabilities.insert("ml_linear_algebra_builtins");
    }
    if uses_q31 {
        capabilities.insert("deterministic_q31_fixed_point");
    }
    if !witness_report.names.is_empty() {
        capabilities.insert("explicit_witness_tapes");
    }
    if !witness_report.names.is_empty() && witness_report.unknown_names.is_empty() {
        capabilities.insert("known_witness_budget");
    }
    if !summary.dataset_loops.is_empty() {
        capabilities.insert("dataset_shaped_iteration");
    }
    if !summary.lossy_signal_counts.is_empty() && witness_update_statements > 0 {
        capabilities.insert("non_injective_signals_captured_as_state");
    }
    capabilities.insert("static_replay_cost_model");
    capabilities.insert("roundtrip_proof_ready");

    let goal_fit = if capabilities.contains("static_tensor_shapes")
        && capabilities.contains("reversible_tensor_accumulation")
        && capabilities.contains("deterministic_q31_fixed_point")
        && capabilities.contains("explicit_witness_tapes")
    {
        "auditable_ml_kernel"
    } else if capabilities.contains("static_tensor_shapes")
        || capabilities.contains("ml_linear_algebra_builtins")
    {
        "tensor_reversible_kernel"
    } else {
        "general_reversible_program"
    };

    json!({
        "schema": "reverie_explain_ml_profile_v1",
        "goal_fit": goal_fit,
        "capabilities": capabilities.into_iter().collect::<Vec<_>>(),
        "tensor_variables": tensor_names.iter().cloned().collect::<Vec<_>>(),
        "tensor_entries": tensor_names.iter().map(|name| {
            let measure = tensor_measures.get(name);
            json!({
                "name": name,
                "role": if witness_report.names.contains(name) { "witness" } else { "state" },
                "cells": measure.and_then(|measure| measure.cells),
                "payload_bytes": measure.and_then(|measure| measure.payload_bytes),
            })
        }).collect::<Vec<_>>(),
        "tensor_metrics": {
            "variables": tensor_names.len(),
            "state_variables": state_tensor_names.len(),
            "witness_variables": witness_report.names.len(),
            "known_cells": known_tensor_cells,
            "known_payload_bytes": known_tensor_payload_bytes,
            "known_state_cells": known_state_tensor_cells,
            "known_state_payload_bytes": known_state_tensor_payload_bytes,
            "known_witness_cells": witness_report.known_cells,
            "known_witness_payload_bytes": witness_report.known_payload_bytes,
            "unknown_variables": unknown_tensor_variables,
            "unknown_state_variables": unknown_state_tensor_variables,
            "unknown_witness_variables": &witness_report.unknown_names,
        },
        "replay_cost": {
            "forward_statement_count": summary.statements,
            "inverse_statement_count": summary.statements,
            "roundtrip_statement_count": summary.statements.saturating_mul(2),
            "forward_expression_count": summary.expressions,
            "inverse_expression_count": summary.expressions,
            "roundtrip_expression_count": summary.expressions.saturating_mul(2),
            "known_state_tensor_payload_bytes": known_state_tensor_payload_bytes,
            "known_witness_payload_bytes": witness_report.known_payload_bytes,
            "known_replay_payload_bytes": known_replay_payload_bytes,
            "unknown_state_tensor_variables": unknown_state_tensor_variables,
            "unknown_witness_variables": &witness_report.unknown_names,
            "witness_to_state_payload_ratio": ratio_json(
                witness_report.known_payload_bytes,
                known_state_tensor_payload_bytes,
            ),
        },
        "update_counts": {
            "tensor_update_statements": tensor_update_statements,
            "witness_update_statements": witness_update_statements,
            "all_update_targets": &summary.update_targets,
        },
        "builtin_counts": &summary.tensor_builtin_counts,
        "tensor_builtin_calls": tensor_builtin_calls,
        "q31_builtin_calls": q31_calls,
        "fixed_point_binary_ops": summary.fixed_point_binary_ops,
        "non_injective_signal_calls": &summary.lossy_signal_counts,
        "dataset_loop_count": summary.dataset_loops.len(),
        "roundtrip_template": "reverie roundtrip <FILE> --json",
    })
}

fn tensor_inventory(
    summary: &ProgramSummary,
    types: &[TypeArg],
) -> (BTreeSet<String>, BTreeMap<String, WitnessMeasure>) {
    let mut names = summary.tensor_names.clone();
    let mut measures = summary.tensor_measures.clone();
    for ty in types {
        if summary.external_names.contains(&ty.name) && type_contains_tensor(&ty.ty.node) {
            names.insert(ty.name.clone());
            measures.insert(
                ty.name.clone(),
                witness_measure_for_storage(&[], &ty.ty.node),
            );
        }
    }
    (names, measures)
}

fn sum_known_measures(
    names: &BTreeSet<String>,
    measures: &BTreeMap<String, WitnessMeasure>,
) -> (usize, usize, Vec<String>) {
    let mut cells = 0_usize;
    let mut payload_bytes = 0_usize;
    let mut unknown = Vec::new();
    for name in names {
        match measures.get(name) {
            Some(WitnessMeasure {
                cells: Some(entry_cells),
                payload_bytes: Some(entry_payload_bytes),
            }) => {
                cells = cells.saturating_add(*entry_cells);
                payload_bytes = payload_bytes.saturating_add(*entry_payload_bytes);
            }
            _ => unknown.push(name.clone()),
        }
    }
    (cells, payload_bytes, unknown)
}

fn update_count_for_names(
    update_targets: &BTreeMap<String, usize>,
    names: &BTreeSet<String>,
) -> usize {
    names
        .iter()
        .filter_map(|name| update_targets.get(name))
        .sum::<usize>()
}

#[derive(Debug, Clone, Copy)]
struct RunProvenance<'a> {
    file: &'a Path,
    source: &'a str,
    direction: &'static str,
    engine: Engine,
    legacy_janus: bool,
}

#[derive(Debug)]
struct RoundtripCheck {
    passed: bool,
    baseline_store_restored: bool,
    input_restored: bool,
    output_restored: bool,
    missing_store_entries: Vec<String>,
    changed_store_entries: Vec<String>,
    declared_store_entries: Vec<String>,
    unexpected_store_entries: Vec<String>,
}

#[derive(Debug)]
struct RoundtripProofInput<'a> {
    file: &'a Path,
    source: &'a str,
    engine: Engine,
    legacy_janus: bool,
    types: &'a [TypeArg],
    summary: &'a ProgramSummary,
    witness_report: &'a WitnessReport,
    baseline: &'a IoState,
    forward: &'a IoState,
    restored: &'a IoState,
    check: &'a RoundtripCheck,
    include_ml_profile: bool,
}

fn run_result_metadata(
    provenance: RunProvenance<'_>,
    program: &Program,
    types: &[TypeArg],
    state: &IoState,
) -> JsonValue {
    let summary = summarize_program(program);
    let witness_report = witness_report(&summary, types);
    json!({
        "program": {
            "file": provenance.file.display().to_string(),
            "source_sha256": sha256_bytes(provenance.source.as_bytes()),
            "direction": provenance.direction,
            "engine": provenance.engine.as_str(),
            "legacy_janus": provenance.legacy_janus,
        },
        "dataset_loops": dataset_loops_json(&summary.dataset_loops),
        "store_metadata": store_metadata_json(program, types, &witness_report, state),
        "witness_store": witness_report.names.iter().cloned().collect::<Vec<_>>(),
        "witness_metrics": witness_metrics_json(&witness_report),
        "ml_profile": ml_profile_json(&summary, &witness_report, types),
        "witness_proof": witness_proof_json(&witness_report, state),
    })
}

fn execute_initialization_io(
    program: &Program,
    state: IoState,
    engine: Engine,
    options: ExecutionOptions,
) -> Result<IoState, RuntimeError> {
    let initialization = Program::with_globals_and_procedures(
        program.globals.clone(),
        Vec::new(),
        Spanned::new(Stmt::Skip, 0..0),
    );
    match engine {
        Engine::Tree => reverie_interp::execute_io_with_options(&initialization, state, options),
        Engine::Slot => {
            reverie_interp::execute_compiled_io_with_options(&initialization, state, options)
        }
    }
}

fn roundtrip_check(
    baseline: &IoState,
    restored: &IoState,
    summary: &ProgramSummary,
) -> RoundtripCheck {
    let baseline_store = baseline.store().store();
    let restored_store = restored.store().store();
    let missing_store_entries = baseline_store
        .keys()
        .filter(|name| !restored_store.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    let changed_store_entries = baseline_store
        .iter()
        .filter(|(name, value)| restored_store.get(*name) != Some(*value))
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let extra_store_entries = restored_store
        .keys()
        .filter(|name| !baseline_store.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    let declared_store_entries = extra_store_entries
        .iter()
        .filter(|name| summary.declared_names.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    let unexpected_store_entries = extra_store_entries
        .into_iter()
        .filter(|name| !summary.declared_names.contains(name))
        .collect::<Vec<_>>();
    let baseline_store_restored = missing_store_entries.is_empty()
        && changed_store_entries.is_empty()
        && unexpected_store_entries.is_empty();
    let input_restored = baseline.input() == restored.input();
    let output_restored = baseline.output() == restored.output();
    let passed = baseline_store_restored && input_restored && output_restored;

    RoundtripCheck {
        passed,
        baseline_store_restored,
        input_restored,
        output_restored,
        missing_store_entries,
        changed_store_entries,
        declared_store_entries,
        unexpected_store_entries,
    }
}

fn roundtrip_proof_json(input: RoundtripProofInput<'_>) -> JsonValue {
    let baseline = io_state_json(input.baseline);
    let forward = io_state_json(input.forward);
    let restored = io_state_json(input.restored);
    let forward_witness_proof = witness_proof_json(input.witness_report, input.forward);
    let restored_witness_proof = witness_proof_json(input.witness_report, input.restored);
    let check = roundtrip_check_json(input.check);
    let mut payload = json!({
        "schema": "reverie_roundtrip_proof_v1",
        "program": {
            "file": input.file.display().to_string(),
            "source_sha256": sha256_bytes(input.source.as_bytes()),
            "direction": "roundtrip",
            "engine": input.engine.as_str(),
            "legacy_janus": input.legacy_janus,
        },
        "type_annotations": type_annotations_json(input.types),
        "dataset_loops": dataset_loops_json(&input.summary.dataset_loops),
        "baseline": baseline,
        "forward": forward,
        "restored": restored,
        "check": check,
        "witness_store": input.witness_report.names.iter().cloned().collect::<Vec<_>>(),
        "witness_metrics": witness_metrics_json(input.witness_report),
        "forward_witness_proof": forward_witness_proof,
        "restored_witness_proof": restored_witness_proof,
    });
    if input.include_ml_profile {
        payload
            .as_object_mut()
            .expect("roundtrip payload is an object")
            .insert(
                "ml_profile".to_owned(),
                ml_profile_json(input.summary, input.witness_report, input.types),
            );
    }
    let payload_fingerprint = sha256_json(&payload);
    let fingerprints = json!({
        "algorithm": "sha256",
        "payload": payload_fingerprint,
        "baseline": sha256_json(&payload["baseline"]),
        "forward": sha256_json(&payload["forward"]),
        "restored": sha256_json(&payload["restored"]),
        "forward_witness_proof": sha256_json(&payload["forward_witness_proof"]),
        "restored_witness_proof": sha256_json(&payload["restored_witness_proof"]),
    });
    json!({
        "kind": "reverie_roundtrip_result",
        "schema": "reverie_roundtrip_proof_v1",
        "passed": input.check.passed,
        "fingerprint": payload_fingerprint,
        "fingerprints": fingerprints,
        "payload": payload,
    })
}

fn type_annotations_json(types: &[TypeArg]) -> Vec<JsonValue> {
    types
        .iter()
        .map(|ty| {
            json!({
                "name": ty.name,
                "type": format_type_expr(&ty.ty.node),
            })
        })
        .collect()
}

fn roundtrip_check_json(check: &RoundtripCheck) -> JsonValue {
    json!({
        "passed": check.passed,
        "baseline_store_restored": check.baseline_store_restored,
        "input_restored": check.input_restored,
        "output_restored": check.output_restored,
        "missing_store_entries": check.missing_store_entries,
        "changed_store_entries": check.changed_store_entries,
        "declared_store_entries": check.declared_store_entries,
        "unexpected_store_entries": check.unexpected_store_entries,
    })
}

fn io_state_json(state: &IoState) -> JsonValue {
    let store = state
        .store()
        .store()
        .iter()
        .map(|(name, value)| (name.clone(), json_from_value(value)))
        .collect::<JsonMap<_, _>>();
    json!({
        "store": store,
        "input": state.input().iter().map(json_from_value).collect::<Vec<_>>(),
        "output": state.output().iter().map(json_from_value).collect::<Vec<_>>(),
        "observations": state.observations(),
    })
}

struct RoundtripProofText<'a> {
    file: &'a Path,
    engine: Engine,
    legacy_janus: bool,
    baseline: &'a IoState,
    forward: &'a IoState,
    restored: &'a IoState,
    witness_report: &'a WitnessReport,
    proof: &'a JsonValue,
    check: &'a RoundtripCheck,
    proof_output: Option<&'a Path>,
}

fn print_roundtrip_proof_text(input: RoundtripProofText<'_>) {
    let RoundtripProofText {
        file,
        engine,
        legacy_janus,
        baseline,
        forward,
        restored,
        witness_report,
        proof,
        check,
        proof_output,
    } = input;

    println!("roundtrip: {}", if check.passed { "ok" } else { "failed" });
    println!("file: {}", file.display());
    println!("engine: {}", engine.as_str());
    println!("legacy_janus: {legacy_janus}");
    println!(
        "checks: store={} input={} output={}",
        check.baseline_store_restored, check.input_restored, check.output_restored
    );
    if !check.missing_store_entries.is_empty() {
        println!(
            "missing store entries: {}",
            check.missing_store_entries.join(",")
        );
    }
    if !check.changed_store_entries.is_empty() {
        println!(
            "changed store entries: {}",
            check.changed_store_entries.join(",")
        );
    }
    if !check.declared_store_entries.is_empty() {
        println!(
            "declared store entries: {}",
            check.declared_store_entries.join(",")
        );
    }
    if !check.unexpected_store_entries.is_empty() {
        println!(
            "unexpected store entries: {}",
            check.unexpected_store_entries.join(",")
        );
    }
    println!("baseline: {}", baseline.store());
    println!("forward: {}", forward.store());
    println!("restored: {}", restored.store());
    println!(
        "tapes: input={} output={} observations_forward={} observations_restored={}",
        baseline.input().len(),
        baseline.output().len(),
        forward.observations().len(),
        restored.observations().len()
    );
    println!(
        "witness proof cost: variables={} known_cells={} known_payload_bytes={} unknown_variables={}",
        witness_report.names.len(),
        witness_report.known_cells,
        witness_report.known_payload_bytes,
        if witness_report.unknown_names.is_empty() {
            "none".to_owned()
        } else {
            witness_report.unknown_names.join(",")
        }
    );
    if let Some(profile) = proof.pointer("/payload/ml_profile") {
        print_roundtrip_ml_profile_text(profile);
    }
    println!(
        "fingerprint: {}",
        proof["fingerprint"]
            .as_str()
            .expect("roundtrip proof fingerprint is a string")
    );
    if let Some(path) = proof_output {
        println!("proof: {}", path.display());
    }
}

fn print_roundtrip_ml_profile_text(profile: &JsonValue) {
    let tensor_metrics = profile.get("tensor_metrics").unwrap_or(&JsonValue::Null);
    let replay_cost = profile.get("replay_cost").unwrap_or(&JsonValue::Null);
    println!("ml profile: {}", json_str(profile, "goal_fit"));
    println!(
        "ml witness payload bytes: {}",
        json_u64(tensor_metrics, "known_witness_payload_bytes")
    );
    println!(
        "ml q31 builtin calls: {}",
        json_u64(profile, "q31_builtin_calls")
    );
    println!(
        "ml replay roundtrip statements: {}",
        json_u64(replay_cost, "roundtrip_statement_count")
    );
    println!(
        "ml replay payload bytes: {}",
        json_u64(replay_cost, "known_replay_payload_bytes")
    );
}

fn read_json_file(path: &Path) -> Result<JsonValue, CliError> {
    let text = fs::read_to_string(path).map_err(|source| CliError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| {
        CliError::SeedType(format!("failed to parse {}: {source}", path.display()))
    })
}

fn write_json_file(path: &Path, value: &JsonValue) -> Result<(), CliError> {
    let text = serde_json::to_string_pretty(value).expect("JSON proof serializes");
    fs::write(path, format!("{text}\n")).map_err(|source| CliError::Write {
        path: path.to_path_buf(),
        source,
    })
}

fn write_text_file(path: &Path, text: &str) -> Result<(), CliError> {
    fs::write(path, text).map_err(|source| CliError::Write {
        path: path.to_path_buf(),
        source,
    })
}

fn verify_roundtrip_proof(path: &Path, proof: &JsonValue) -> Result<JsonValue, CliError> {
    let payload = proof.get("payload").unwrap_or(&JsonValue::Null);
    let schema_ok = proof.get("schema").and_then(JsonValue::as_str)
        == Some("reverie_roundtrip_proof_v1")
        && payload.get("schema").and_then(JsonValue::as_str) == Some("reverie_roundtrip_proof_v1");
    let kind_ok = proof.get("kind").and_then(JsonValue::as_str) == Some("reverie_roundtrip_result");
    let artifact_passed = proof.get("passed").and_then(JsonValue::as_bool) == Some(true)
        && payload
            .pointer("/check/passed")
            .and_then(JsonValue::as_bool)
            == Some(true);
    let fingerprint = proof
        .get("fingerprint")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_owned();
    let payload_fingerprint = sha256_json(payload);
    let payload_fingerprint_ok = fingerprint == payload_fingerprint;
    let declared_payload_fingerprint_ok = proof
        .pointer("/fingerprints/payload")
        .and_then(JsonValue::as_str)
        == Some(fingerprint.as_str());
    let baseline_fingerprint_ok = subpayload_fingerprint_ok(proof, payload, "baseline");
    let forward_fingerprint_ok = subpayload_fingerprint_ok(proof, payload, "forward");
    let restored_fingerprint_ok = subpayload_fingerprint_ok(proof, payload, "restored");
    let forward_witness_fingerprint_ok =
        subpayload_fingerprint_ok(proof, payload, "forward_witness_proof")
            && witness_proof_fingerprint_ok(&payload["forward_witness_proof"]);
    let restored_witness_fingerprint_ok =
        subpayload_fingerprint_ok(proof, payload, "restored_witness_proof")
            && witness_proof_fingerprint_ok(&payload["restored_witness_proof"]);
    let ml_profile = payload
        .get("ml_profile")
        .cloned()
        .unwrap_or(JsonValue::Null);
    let ml_profile_present = ml_profile.is_object();
    let ml_profile_schema_ok = !ml_profile_present
        || ml_profile.get("schema").and_then(JsonValue::as_str)
            == Some("reverie_explain_ml_profile_v1");

    let source_path = payload
        .pointer("/program/file")
        .and_then(JsonValue::as_str)
        .map(PathBuf::from);
    let expected_source_hash = payload
        .pointer("/program/source_sha256")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_owned();
    let mut source_readable = false;
    let mut source_hash_matches = false;
    let mut replayed = false;
    let mut replay_fingerprint = JsonValue::Null;
    let mut replay_fingerprint_matches = false;
    let mut replay_restoration_passed = false;
    let mut replay_ml_profile = JsonValue::Null;
    let mut replay_ml_profile_matches = !ml_profile_present;

    if let Some(source_path) = source_path {
        if let Ok(source) = fs::read_to_string(&source_path) {
            source_readable = true;
            source_hash_matches = sha256_bytes(source.as_bytes()) == expected_source_hash;
            if source_hash_matches {
                let replay =
                    replay_roundtrip_proof_from_saved_payload(&source_path, &source, payload)?;
                replayed = true;
                replay_fingerprint = replay["fingerprint"].clone();
                replay_fingerprint_matches =
                    replay["fingerprint"].as_str() == Some(fingerprint.as_str());
                replay_restoration_passed = replay
                    .pointer("/payload/check/passed")
                    .and_then(JsonValue::as_bool)
                    == Some(true);
                replay_ml_profile = replay
                    .pointer("/payload/ml_profile")
                    .cloned()
                    .unwrap_or(JsonValue::Null);
                replay_ml_profile_matches = !ml_profile_present || replay_ml_profile == ml_profile;
            }
        }
    }

    let checks = json!({
        "schema": schema_ok,
        "kind": kind_ok,
        "artifact_passed": artifact_passed,
        "payload_fingerprint": payload_fingerprint_ok,
        "declared_payload_fingerprint": declared_payload_fingerprint_ok,
        "baseline_fingerprint": baseline_fingerprint_ok,
        "forward_fingerprint": forward_fingerprint_ok,
        "restored_fingerprint": restored_fingerprint_ok,
        "forward_witness_fingerprint": forward_witness_fingerprint_ok,
        "restored_witness_fingerprint": restored_witness_fingerprint_ok,
        "ml_profile_schema": ml_profile_schema_ok,
        "ml_profile_replayed": replay_ml_profile_matches,
        "source_readable": source_readable,
        "source_hash_matches": source_hash_matches,
        "replayed": replayed,
        "replay_fingerprint_matches": replay_fingerprint_matches,
        "replay_restoration_passed": replay_restoration_passed,
    });
    let passed = checks
        .as_object()
        .expect("checks JSON is an object")
        .values()
        .all(|value| value.as_bool() == Some(true));

    Ok(json!({
        "kind": "reverie_roundtrip_verification",
        "schema": "reverie_roundtrip_verification_v1",
        "passed": passed,
        "proof_path": path.display().to_string(),
        "proof_fingerprint": fingerprint,
        "payload_fingerprint": payload_fingerprint,
        "replay_fingerprint": replay_fingerprint,
        "ml_profile_present": ml_profile_present,
        "ml_profile_fingerprint": if ml_profile_present { json!(sha256_json(&ml_profile)) } else { JsonValue::Null },
        "replay_ml_profile_fingerprint": if replay_ml_profile.is_object() { json!(sha256_json(&replay_ml_profile)) } else { JsonValue::Null },
        "ml_profile": ml_profile,
        "checks": checks,
    }))
}

fn render_roundtrip_verification_markdown(proof: &JsonValue, verification: &JsonValue) -> String {
    let payload = proof.get("payload").unwrap_or(&JsonValue::Null);
    let program = payload.get("program").unwrap_or(&JsonValue::Null);
    let metrics = payload.get("witness_metrics").unwrap_or(&JsonValue::Null);
    let ml_profile = payload.get("ml_profile").unwrap_or(&JsonValue::Null);
    let fingerprints = proof.get("fingerprints").unwrap_or(&JsonValue::Null);
    let checks = verification.get("checks").unwrap_or(&JsonValue::Null);
    let passed = verification["passed"].as_bool() == Some(true);
    let mut lines = vec![
        "# Reverie Roundtrip Verification".to_owned(),
        "".to_owned(),
        format!("**Verdict:** {}", if passed { "pass" } else { "fail" }),
        format!(
            "**Proof fingerprint:** `{}`",
            json_str(verification, "proof_fingerprint")
        ),
        format!(
            "**Replay fingerprint:** `{}`",
            json_str(verification, "replay_fingerprint")
        ),
        "".to_owned(),
        "## Source".to_owned(),
        "".to_owned(),
        "| File | Engine | Legacy Janus | Source SHA-256 |".to_owned(),
        "| --- | --- | --- | --- |".to_owned(),
        format!(
            "| `{}` | `{}` | {} | `{}` |",
            json_str(program, "file"),
            json_str(program, "engine"),
            json_bool(program, "legacy_janus"),
            json_str(program, "source_sha256")
        ),
        "".to_owned(),
        "## Witness Budget".to_owned(),
        "".to_owned(),
        "| Variables | Known cells | Known payload bytes | Unknown variables |".to_owned(),
        "| ---: | ---: | ---: | --- |".to_owned(),
        format!(
            "| {} | {} | {} | {} |",
            json_u64(metrics, "variables"),
            json_u64(metrics, "known_cells"),
            json_u64(metrics, "known_payload_bytes"),
            json_string_array_markdown(metrics.get("unknown_variables"))
        ),
    ];
    if ml_profile.is_object() {
        let tensor_metrics = ml_profile.get("tensor_metrics").unwrap_or(&JsonValue::Null);
        lines.extend([
            "".to_owned(),
            "## ML Profile".to_owned(),
            "".to_owned(),
            "| Goal fit | Capabilities | State tensor bytes | Witness tensor bytes | Replay payload bytes | Roundtrip statements | Tensor updates | Witness updates | Q31 calls | Non-injective signals |".to_owned(),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |".to_owned(),
            format!(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                json_str(ml_profile, "goal_fit"),
                json_string_array_markdown(ml_profile.get("capabilities")),
                json_u64(tensor_metrics, "known_state_payload_bytes"),
                json_u64(tensor_metrics, "known_witness_payload_bytes"),
                json_u64(
                    ml_profile
                        .get("replay_cost")
                        .unwrap_or(&JsonValue::Null),
                    "known_replay_payload_bytes"
                ),
                json_u64(
                    ml_profile
                        .get("replay_cost")
                        .unwrap_or(&JsonValue::Null),
                    "roundtrip_statement_count"
                ),
                json_u64(
                    ml_profile
                        .get("update_counts")
                        .unwrap_or(&JsonValue::Null),
                    "tensor_update_statements"
                ),
                json_u64(
                    ml_profile
                        .get("update_counts")
                        .unwrap_or(&JsonValue::Null),
                    "witness_update_statements"
                ),
                json_u64(ml_profile, "q31_builtin_calls"),
                json_object_counts_markdown(ml_profile.get("non_injective_signal_calls"))
            ),
        ]);
    }
    lines.extend([
        "".to_owned(),
        "## Signed Payloads".to_owned(),
        "".to_owned(),
        "| Payload | SHA-256 |".to_owned(),
        "| --- | --- |".to_owned(),
    ]);
    for key in [
        "payload",
        "baseline",
        "forward",
        "restored",
        "forward_witness_proof",
        "restored_witness_proof",
    ] {
        lines.push(format!(
            "| {} | `{}` |",
            key.replace('_', " "),
            json_str(fingerprints, key)
        ));
    }
    lines.extend([
        "".to_owned(),
        "## Verification Checks".to_owned(),
        "".to_owned(),
        "| Check | Passed |".to_owned(),
        "| --- | --- |".to_owned(),
    ]);
    if let Some(checks) = checks.as_object() {
        for (name, value) in checks {
            lines.push(format!(
                "| {} | {} |",
                name.replace('_', " "),
                value.as_bool().unwrap_or(false)
            ));
        }
    }
    lines.push("".to_owned());
    lines.join("\n")
}

fn json_str<'a>(value: &'a JsonValue, key: &str) -> &'a str {
    value.get(key).and_then(JsonValue::as_str).unwrap_or("")
}

fn json_bool(value: &JsonValue, key: &str) -> bool {
    value.get(key).and_then(JsonValue::as_bool).unwrap_or(false)
}

fn json_u64(value: &JsonValue, key: &str) -> u64 {
    value.get(key).and_then(JsonValue::as_u64).unwrap_or(0)
}

fn ratio_json(numerator: usize, denominator: usize) -> JsonValue {
    if denominator == 0 {
        JsonValue::Null
    } else {
        json!(numerator as f64 / denominator as f64)
    }
}

fn json_string_array_markdown(value: Option<&JsonValue>) -> String {
    let Some(values) = value.and_then(JsonValue::as_array) else {
        return "none".to_owned();
    };
    if values.is_empty() {
        return "none".to_owned();
    }
    values
        .iter()
        .filter_map(JsonValue::as_str)
        .collect::<Vec<_>>()
        .join(", ")
}

fn json_object_counts_markdown(value: Option<&JsonValue>) -> String {
    let Some(entries) = value.and_then(JsonValue::as_object) else {
        return "none".to_owned();
    };
    if entries.is_empty() {
        return "none".to_owned();
    }
    entries
        .iter()
        .map(|(name, count)| format!("{name}={}", count.as_u64().unwrap_or(0)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn subpayload_fingerprint_ok(proof: &JsonValue, payload: &JsonValue, key: &str) -> bool {
    let Some(value) = payload.get(key) else {
        return false;
    };
    proof
        .pointer(&format!("/fingerprints/{key}"))
        .and_then(JsonValue::as_str)
        == Some(sha256_json(value).as_str())
}

fn witness_proof_fingerprint_ok(proof: &JsonValue) -> bool {
    let Some(payload) = proof.get("payload") else {
        return false;
    };
    proof.get("fingerprint").and_then(JsonValue::as_str) == Some(sha256_json(payload).as_str())
}

fn replay_roundtrip_proof_from_saved_payload(
    file: &Path,
    source: &str,
    payload: &JsonValue,
) -> Result<JsonValue, CliError> {
    let engine = engine_from_roundtrip_payload(payload)?;
    let legacy_janus = payload
        .pointer("/program/legacy_janus")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    let mut types = type_args_from_roundtrip_payload(payload)?;
    normalize_legacy_seed_types(legacy_janus, &mut types);
    validate_type_args(&types)?;
    let baseline = io_state_from_roundtrip_payload(payload.get("baseline"), "baseline")?;
    let program = parse_or_report(file, source, legacy_janus)?;
    validate_external_types(file, &program, &types)?;
    check_or_report(file, source, &program, &types, legacy_janus)?;
    let options = execution_options(legacy_janus);
    let forward = runtime_or_report(file, source, || match engine {
        Engine::Tree => {
            reverie_interp::execute_io_with_options(&program, baseline.clone(), options)
        }
        Engine::Slot => {
            reverie_interp::execute_compiled_io_with_options(&program, baseline.clone(), options)
        }
    })?;
    let restored = runtime_or_report(file, source, || match engine {
        Engine::Tree => {
            reverie_interp::execute_io_backward_with_options(&program, forward.clone(), options)
        }
        Engine::Slot => reverie_interp::execute_compiled_io_backward_with_options(
            &program,
            forward.clone(),
            options,
        ),
    })?;
    let summary = summarize_program(&program);
    let witness_report = witness_report(&summary, &types);
    let check = roundtrip_check(&baseline, &restored, &summary);
    let include_ml_profile = payload.get("ml_profile").is_some();
    Ok(roundtrip_proof_json(RoundtripProofInput {
        file,
        source,
        engine,
        legacy_janus,
        types: &types,
        summary: &summary,
        witness_report: &witness_report,
        baseline: &baseline,
        forward: &forward,
        restored: &restored,
        check: &check,
        include_ml_profile,
    }))
}

fn engine_from_roundtrip_payload(payload: &JsonValue) -> Result<Engine, CliError> {
    match payload
        .pointer("/program/engine")
        .and_then(JsonValue::as_str)
    {
        Some("tree") => Ok(Engine::Tree),
        Some("slot") => Ok(Engine::Slot),
        Some(other) => Err(CliError::SeedType(format!(
            "roundtrip proof has unsupported engine `{other}`"
        ))),
        None => Err(CliError::SeedType(
            "roundtrip proof is missing payload.program.engine".to_owned(),
        )),
    }
}

fn type_args_from_roundtrip_payload(payload: &JsonValue) -> Result<Vec<TypeArg>, CliError> {
    let Some(values) = payload.get("type_annotations") else {
        return Ok(Vec::new());
    };
    let values = values.as_array().ok_or_else(|| {
        CliError::SeedType("roundtrip proof type_annotations must be an array".to_owned())
    })?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let name = value
                .get("name")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| {
                    CliError::SeedType(format!(
                        "roundtrip proof type_annotations[{index}] is missing name"
                    ))
                })?;
            if !is_identifier(name) {
                return Err(CliError::SeedType(format!(
                    "roundtrip proof type annotation `{name}` is not a valid Reverie identifier"
                )));
            }
            let ty = value
                .get("type")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| {
                    CliError::SeedType(format!(
                        "roundtrip proof type_annotations[{index}] is missing type"
                    ))
                })?;
            let ty = parse_type_expr(ty).map_err(|diagnostics| {
                CliError::SeedType(
                    diagnostics
                        .into_iter()
                        .map(|diagnostic| diagnostic.message)
                        .collect::<Vec<_>>()
                        .join("; "),
                )
            })?;
            Ok(TypeArg {
                name: name.to_owned(),
                ty,
            })
        })
        .collect()
}

fn io_state_from_roundtrip_payload(
    value: Option<&JsonValue>,
    context: &str,
) -> Result<IoState, CliError> {
    let value = value.ok_or_else(|| {
        CliError::SeedType(format!("roundtrip proof is missing payload.{context}"))
    })?;
    let object = value.as_object().ok_or_else(|| {
        CliError::SeedType(format!(
            "roundtrip proof payload.{context} must be an object"
        ))
    })?;
    let store = object
        .get("store")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            CliError::SeedType(format!(
                "roundtrip proof payload.{context}.store must be an object"
            ))
        })?;
    let mut bindings = Vec::new();
    for (name, value) in store {
        if !is_identifier(name) {
            return Err(CliError::SeedType(format!(
                "roundtrip proof payload.{context}.store contains invalid name `{name}`"
            )));
        }
        bindings.push((
            name.clone(),
            value_from_json_seed(value, &format!("payload.{context}.store.{name}"))?,
        ));
    }
    let input = values_from_roundtrip_array(object.get("input"), &format!("{context}.input"))?;
    let output = values_from_roundtrip_array(object.get("output"), &format!("{context}.output"))?;
    if !object
        .get("observations")
        .and_then(JsonValue::as_array)
        .is_some_and(Vec::is_empty)
    {
        return Err(CliError::SeedType(format!(
            "roundtrip proof payload.{context}.observations must be an empty array for replay seeding"
        )));
    }
    Ok(IoState::with_output(
        State::from_bindings(bindings),
        input,
        output,
    ))
}

fn values_from_roundtrip_array(
    value: Option<&JsonValue>,
    context: &str,
) -> Result<Vec<Value>, CliError> {
    let values = value.and_then(JsonValue::as_array).ok_or_else(|| {
        CliError::SeedType(format!(
            "roundtrip proof payload.{context} must be an array"
        ))
    })?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| value_from_json_seed(value, &format!("payload.{context}[{index}]")))
        .collect()
}

fn print_roundtrip_verification_text(
    path: &Path,
    verification: &JsonValue,
    markdown_output: Option<&Path>,
) {
    println!(
        "roundtrip verification: {}",
        if verification["passed"].as_bool() == Some(true) {
            "ok"
        } else {
            "failed"
        }
    );
    println!("proof: {}", path.display());
    println!(
        "fingerprint: {}",
        verification["proof_fingerprint"]
            .as_str()
            .unwrap_or("<missing>")
    );
    println!(
        "replay fingerprint: {}",
        verification["replay_fingerprint"]
            .as_str()
            .unwrap_or("<missing>")
    );
    if let Some(checks) = verification["checks"].as_object() {
        for (name, value) in checks {
            println!("{name}: {}", value.as_bool().unwrap_or(false));
        }
    }
    if let Some(path) = markdown_output {
        println!("markdown: {}", path.display());
    }
}

impl Engine {
    fn as_str(self) -> &'static str {
        match self {
            Engine::Tree => "tree",
            Engine::Slot => "slot",
        }
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_lower(&digest)
}

fn sha256_json(value: &JsonValue) -> String {
    let bytes = serde_json::to_vec(value).expect("JSON fingerprint payload serializes");
    let normalized: JsonValue =
        serde_json::from_slice(&bytes).expect("JSON fingerprint payload normalizes");
    let normalized_bytes =
        serde_json::to_vec(&normalized).expect("normalized JSON fingerprint payload serializes");
    sha256_bytes(&normalized_bytes)
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

fn witness_metrics_json(report: &WitnessReport) -> JsonValue {
    json!({
        "variables": report.names.len(),
        "known_cells": report.known_cells,
        "known_payload_bytes": report.known_payload_bytes,
        "unknown_variables": report.unknown_names,
        "entries": report.names.iter().map(|name| {
            let measure = report.measures.get(name);
            json!({
                "name": name,
                "cells": measure.and_then(|measure| measure.cells),
                "payload_bytes": measure.and_then(|measure| measure.payload_bytes),
            })
        }).collect::<Vec<_>>(),
    })
}

fn witness_proof_json(report: &WitnessReport, state: &IoState) -> JsonValue {
    let store = state.store().store();
    let entries = report
        .names
        .iter()
        .map(|name| {
            let value = store.get(name).map(json_from_value);
            let value_fingerprint = sha256_json(value.as_ref().unwrap_or(&JsonValue::Null));
            let measure = report.measures.get(name);
            json!({
                "name": name,
                "present": value.is_some(),
                "cells": measure.and_then(|measure| measure.cells),
                "payload_bytes": measure.and_then(|measure| measure.payload_bytes),
                "value_fingerprint": value_fingerprint,
            })
        })
        .collect::<Vec<_>>();
    let payload = json!({
        "schema": "reverie_witness_store_proof_v1",
        "algorithm": "sha256",
        "variables": report.names.len(),
        "known_cells": report.known_cells,
        "known_payload_bytes": report.known_payload_bytes,
        "unknown_variables": report.unknown_names,
        "entries": entries,
    });
    json!({
        "schema": "reverie_witness_store_proof_v1",
        "algorithm": "sha256",
        "fingerprint": sha256_json(&payload),
        "payload": payload,
    })
}

fn store_metadata_json(
    program: &Program,
    types: &[TypeArg],
    witness_report: &WitnessReport,
    state: &IoState,
) -> Vec<JsonValue> {
    let mut entries = state
        .store()
        .store()
        .keys()
        .map(|name| {
            (
                name.clone(),
                json!({
                    "name": name,
                    "type": JsonValue::Null,
                    "role": "state",
                    "source": JsonValue::Null,
                    "cells": JsonValue::Null,
                    "payload_bytes": JsonValue::Null,
                }),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let declared_names = program
        .globals
        .iter()
        .map(|global| global.name.clone())
        .collect::<BTreeSet<_>>();

    for global in &program.globals {
        let ty = global.ty.as_ref().map(|ty| &ty.node);
        upsert_store_metadata(
            &mut entries,
            &global.name,
            ty,
            "declaration",
            witness_report,
        );
    }
    for ty in types {
        if declared_names.contains(&ty.name) {
            continue;
        }
        upsert_store_metadata(
            &mut entries,
            &ty.name,
            Some(&ty.ty.node),
            "type_annotation",
            witness_report,
        );
    }

    entries.into_values().collect()
}

fn upsert_store_metadata(
    entries: &mut BTreeMap<String, JsonValue>,
    name: &str,
    ty: Option<&TypeExpr>,
    source: &str,
    witness_report: &WitnessReport,
) {
    let role = if witness_report.names.contains(name) {
        "witness"
    } else {
        "state"
    };
    let measure = witness_report.measures.get(name);
    let entry = json!({
        "name": name,
        "type": ty.map(format_type_expr),
        "role": role,
        "source": source,
        "cells": measure.and_then(|measure| measure.cells),
        "payload_bytes": measure.and_then(|measure| measure.payload_bytes),
    });
    entries.insert(name.to_owned(), entry);
}

fn witness_measure_for_storage(dims: &[usize], ty: &TypeExpr) -> WitnessMeasure {
    let mut measure = witness_measure_for_type(ty);
    if let Some(factor) = dims
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
    {
        measure = multiply_witness_measure(measure, factor);
    } else {
        measure = WitnessMeasure {
            cells: None,
            payload_bytes: None,
        };
    }
    measure
}

fn witness_measure_for_type(ty: &TypeExpr) -> WitnessMeasure {
    match ty {
        TypeExpr::Witness { inner } => witness_measure_for_type(&inner.node),
        TypeExpr::Tensor { element, shape } => {
            let measure = witness_measure_for_type(&element.node);
            let Some(factor) = shape
                .iter()
                .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
            else {
                return WitnessMeasure {
                    cells: None,
                    payload_bytes: None,
                };
            };
            multiply_witness_measure(measure, factor)
        }
        TypeExpr::Int { .. } => WitnessMeasure {
            cells: Some(1),
            payload_bytes: Some(size_of::<i64>()),
        },
        TypeExpr::Bool => WitnessMeasure {
            cells: Some(1),
            payload_bytes: Some(size_of::<bool>()),
        },
        TypeExpr::Array { .. } | TypeExpr::Stack => WitnessMeasure {
            cells: None,
            payload_bytes: None,
        },
    }
}

fn multiply_witness_measure(measure: WitnessMeasure, factor: usize) -> WitnessMeasure {
    WitnessMeasure {
        cells: measure.cells.and_then(|cells| cells.checked_mul(factor)),
        payload_bytes: measure
            .payload_bytes
            .and_then(|payload_bytes| payload_bytes.checked_mul(factor)),
    }
}

fn type_contains_witness(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Witness { .. } => true,
        TypeExpr::Array { element } | TypeExpr::Tensor { element, .. } => {
            type_contains_witness(&element.node)
        }
        TypeExpr::Int { .. } | TypeExpr::Bool | TypeExpr::Stack => false,
    }
}

fn type_contains_tensor(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Tensor { .. } => true,
        TypeExpr::Array { element } | TypeExpr::Witness { inner: element } => {
            type_contains_tensor(&element.node)
        }
        TypeExpr::Int { .. } | TypeExpr::Bool | TypeExpr::Stack => false,
    }
}

fn is_lossy_signal_builtin(name: &str) -> bool {
    matches!(
        name,
        "argmax"
            | "argmax_eq"
            | "clamp"
            | "clamp_q31"
            | "normalize_q31"
            | "relu"
            | "sum"
            | "runner_up"
            | "top2_margin"
            | "rank_of"
            | "top_k_indices"
            | "top_k_values"
            | "top_k_contains"
    )
}

fn visit_name(name: &str, scope: &BTreeSet<String>, summary: &mut ProgramSummary) {
    if !scope.contains(name) {
        summary.external_names.insert(name.to_owned());
    }
}

fn visit_place(
    place: &reverie_syntax::Place,
    scope: &mut BTreeSet<String>,
    summary: &mut ProgramSummary,
) {
    visit_name(&place.name, scope, summary);
    visit_place_indices(&place.indices, scope, summary);
}

fn visit_place_indices(
    indices: &[SpannedExpr],
    scope: &mut BTreeSet<String>,
    summary: &mut ProgramSummary,
) {
    for index in indices {
        summary.features.insert("fixed-size arrays");
        if const_int_value(index).is_some() {
            record_safety_check(summary, "constant array indexes checked before runtime");
        } else {
            record_safety_check(summary, "dynamic array indexes checked at runtime");
        }
        visit_expr(index, scope, summary);
    }
}

fn visit_expr(expr: &SpannedExpr, scope: &mut BTreeSet<String>, summary: &mut ProgramSummary) {
    summary.expressions += 1;
    match &expr.node {
        Expr::Int { .. } | Expr::Bool(_) | Expr::Nil => {}
        Expr::Var(name) => visit_name(name, scope, summary),
        Expr::Array(elements) => {
            summary.features.insert("array literals");
            for element in elements {
                visit_expr(element, scope, summary);
            }
        }
        Expr::Index { target, indices } => {
            summary.features.insert("indexed reads");
            summary.features.insert("fixed-size arrays");
            visit_name(target, scope, summary);
            visit_place_indices(indices, scope, summary);
        }
        Expr::Empty { target } | Expr::Top { target } => {
            summary.features.insert("stack reads");
            visit_name(target, scope, summary);
        }
        Expr::Size { target } => {
            summary.features.insert("size reads");
            visit_name(target, scope, summary);
        }
        Expr::Unary { expr, .. } => {
            visit_expr(expr, scope, summary);
        }
        Expr::Binary { op, left, right } => {
            match op {
                BinaryOp::Div | BinaryOp::Rem => {
                    if const_int_value(right).is_some() {
                        record_safety_check(
                            summary,
                            "constant zero divisors rejected before runtime",
                        );
                    } else {
                        record_safety_check(summary, "dynamic zero divisors checked at runtime");
                    }
                }
                _ => {}
            }
            if *op == reverie_syntax::BinaryOp::FixedMul {
                summary.features.insert("Janus fixed-point multiply");
                summary.fixed_point_binary_ops += 1;
            }
            visit_expr(left, scope, summary);
            visit_expr(right, scope, summary);
        }
        Expr::Call { name, args } => {
            summary.features.insert("tensor builtins");
            *summary
                .tensor_builtin_counts
                .entry(name.clone())
                .or_insert(0) += 1;
            if name.ends_with("_q31") {
                summary.features.insert("Janus fixed-point multiply");
                summary.q31_builtin_calls += 1;
            }
            if is_lossy_signal_builtin(name) {
                *summary.lossy_signal_counts.entry(name.clone()).or_insert(0) += 1;
            }
            for arg in args {
                visit_expr(arg, scope, summary);
            }
        }
    }
}

fn record_safety_check(summary: &mut ProgramSummary, check: &'static str) {
    summary.safety_checks.insert(check);
    *summary.safety_check_counts.entry(check).or_insert(0) += 1;
}

fn all_indices_constant(indices: &[SpannedExpr]) -> bool {
    indices.iter().all(|index| const_int_value(index).is_some())
}

fn expr_reads_root(expr: &SpannedExpr, root: &str) -> bool {
    match &expr.node {
        Expr::Int { .. } | Expr::Bool(_) | Expr::Nil => false,
        Expr::Array(elements) => elements
            .iter()
            .any(|element| expr_reads_root(element, root)),
        Expr::Var(name) | Expr::Empty { target: name } | Expr::Top { target: name } => name == root,
        Expr::Size { .. } => false,
        Expr::Index { target, indices } => {
            target == root || indices.iter().any(|index| expr_reads_root(index, root))
        }
        Expr::Unary { expr, .. } => expr_reads_root(expr, root),
        Expr::Binary { left, right, .. } => {
            expr_reads_root(left, root) || expr_reads_root(right, root)
        }
        Expr::Call { args, .. } => args.iter().any(|arg| expr_reads_root(arg, root)),
    }
}

fn const_int_value(expr: &SpannedExpr) -> Option<i64> {
    match &expr.node {
        Expr::Int { value, .. } => Some(*value),
        Expr::Unary {
            op: UnaryOp::Neg,
            expr,
        } => const_int_value(expr).map(i64::wrapping_neg),
        Expr::Binary { op, left, right } => {
            let left = const_int_value(left)?;
            let right = const_int_value(right)?;
            match op {
                BinaryOp::Add => Some(left.wrapping_add(right)),
                BinaryOp::Sub => Some(left.wrapping_sub(right)),
                BinaryOp::Mul => Some(left.wrapping_mul(right)),
                BinaryOp::FixedMul => Some(((i128::from(left) * i128::from(right)) >> 31) as i64),
                BinaryOp::Div if right != 0 => Some(left.wrapping_div(right)),
                BinaryOp::Rem if right != 0 => Some(left.wrapping_rem(right)),
                BinaryOp::Shl => Some(left.wrapping_shl(right as u32)),
                BinaryOp::Shr => Some(left.wrapping_shr(right as u32)),
                BinaryOp::BitAnd => Some(left & right),
                BinaryOp::BitXor => Some(left ^ right),
                BinaryOp::BitOr => Some(left | right),
                BinaryOp::Div
                | BinaryOp::Rem
                | BinaryOp::Eq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::LtEq
                | BinaryOp::Gt
                | BinaryOp::GtEq
                | BinaryOp::And
                | BinaryOp::Or => None,
            }
        }
        _ => None,
    }
}

fn state_from_vars(vars: Vec<VarArg>) -> State {
    State::from_bindings(vars.into_iter().map(|var| (var.name, var.value)))
}

fn load_vars_json_files(paths: &[PathBuf]) -> Result<Vec<VarArg>, CliError> {
    let mut vars = Vec::new();
    for path in paths {
        vars.extend(load_vars_json_file(path)?);
    }
    Ok(vars)
}

fn load_vars_json_file(path: &Path) -> Result<Vec<VarArg>, CliError> {
    let text = fs::read_to_string(path).map_err(|source| CliError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let json: JsonValue = serde_json::from_str(&text).map_err(|source| {
        CliError::SeedType(format!(
            "failed to parse --vars-json {}: {source}",
            path.display()
        ))
    })?;
    let object = json.as_object().ok_or_else(|| {
        CliError::SeedType(format!(
            "--vars-json {} must contain a JSON object mapping names to values",
            path.display()
        ))
    })?;

    let mut vars = Vec::new();
    for (name, value) in object {
        if !is_identifier(name) {
            return Err(CliError::SeedType(format!(
                "`{name}` from --vars-json {} is not a valid Reverie identifier",
                path.display()
            )));
        }
        vars.push(VarArg {
            name: name.clone(),
            value: value_from_json_seed(
                value,
                &format!("seed `{name}` in --vars-json {}", path.display()),
            )?,
        });
    }
    Ok(vars)
}

fn value_from_json_seed(value: &JsonValue, context: &str) -> Result<Value, CliError> {
    match value {
        JsonValue::Bool(value) => Ok(Value::Bool(*value)),
        JsonValue::Number(number) => Ok(Value::Int(i64_from_json_number(number, context)?)),
        JsonValue::Array(values) => values
            .iter()
            .enumerate()
            .map(|(index, value)| value_from_json_seed(value, &format!("{context}[{index}]")))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        JsonValue::Object(object) => stack_from_json_object(object, context),
        JsonValue::Null | JsonValue::String(_) => Err(CliError::SeedType(format!(
            "{context} must be an integer, boolean, array, or {{\"stack\": [...]}} object"
        ))),
    }
}

fn i64_from_json_number(number: &serde_json::Number, context: &str) -> Result<i64, CliError> {
    if let Some(value) = number.as_i64() {
        return Ok(value);
    }
    if let Some(value) = number.as_u64() {
        return i64::try_from(value)
            .map_err(|_| CliError::SeedType(format!("{context} is outside signed i64 range")));
    }
    Err(CliError::SeedType(format!(
        "{context} must be a signed i64 integer; floating-point JSON numbers are not supported"
    )))
}

fn stack_from_json_object(
    object: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<Value, CliError> {
    let Some(stack) = object.get("stack").filter(|_| object.len() == 1) else {
        return Err(CliError::SeedType(format!(
            "{context} object values are only supported as {{\"stack\": [...]}}"
        )));
    };
    let Some(values) = stack.as_array() else {
        return Err(CliError::SeedType(format!(
            "{context}.stack must be an array of signed i64 integers"
        )));
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            JsonValue::Number(number) => {
                i64_from_json_number(number, &format!("{context}.stack[{index}]"))
            }
            _ => Err(CliError::SeedType(format!(
                "{context}.stack[{index}] must be a signed i64 integer"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Stack)
}

fn normalize_legacy_seed_args(legacy_janus: bool, vars: &mut [VarArg], types: &mut [TypeArg]) {
    if !legacy_janus {
        return;
    }

    for var in vars {
        var.name = var.name.to_ascii_lowercase();
    }
    normalize_legacy_seed_types(legacy_janus, types);
}

fn normalize_legacy_seed_types(legacy_janus: bool, types: &mut [TypeArg]) {
    if !legacy_janus {
        return;
    }

    for ty in types {
        ty.name = ty.name.to_ascii_lowercase();
    }
}

fn print_io_state(state: &IoState, emit_json: bool, kind: &str, metadata: Option<JsonValue>) {
    if emit_json {
        print_io_state_json(state, kind, metadata);
        return;
    }

    for observation in state.observations() {
        print!("{observation}");
        if !observation.ends_with('\n') {
            println!();
        }
    }
    println!("{}", state.store());
    if !state.input().is_empty() {
        println!("input: {}", format_values(state.input().iter()));
    }
    if !state.output().is_empty() {
        println!("output: {}", format_values(state.output().iter()));
    }
}

fn print_io_state_json(state: &IoState, kind: &str, metadata: Option<JsonValue>) {
    let store = state
        .store()
        .store()
        .iter()
        .map(|(name, value)| (name.clone(), json_from_value(value)))
        .collect::<JsonMap<_, _>>();
    let mut result = json!({
        "kind": kind,
        "store": store,
        "input": state.input().iter().map(json_from_value).collect::<Vec<_>>(),
        "output": state.output().iter().map(json_from_value).collect::<Vec<_>>(),
        "observations": state.observations(),
    });
    if let Some(JsonValue::Object(metadata)) = metadata {
        let object = result
            .as_object_mut()
            .expect("run result JSON root is an object");
        for (key, value) in metadata {
            object.insert(key, value);
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&result).expect("run result JSON serializes")
    );
}

fn json_from_value(value: &Value) -> JsonValue {
    match value {
        Value::Int(value) => json!(value),
        Value::Bool(value) => json!(value),
        Value::Array(values) => JsonValue::Array(values.iter().map(json_from_value).collect()),
        Value::Stack(values) => json!({ "stack": values }),
    }
}

fn format_values<'a>(values: impl IntoIterator<Item = &'a Value>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn parse_var(input: &str) -> Result<VarArg, String> {
    let (name, value) = input
        .split_once('=')
        .ok_or_else(|| "expected NAME=VALUE".to_owned())?;

    if !is_identifier(name) {
        return Err(format!("`{name}` is not a valid Reverie identifier"));
    }

    let value = parse_value(value)?;

    Ok(VarArg {
        name: name.to_owned(),
        value,
    })
}

fn parse_type_arg(input: &str) -> Result<TypeArg, String> {
    let (name, ty) = input
        .split_once('=')
        .ok_or_else(|| "expected NAME=TYPE".to_owned())?;

    if !is_identifier(name) {
        return Err(format!("`{name}` is not a valid Reverie identifier"));
    }

    let ty = parse_type_expr(ty).map_err(|diagnostics| {
        diagnostics
            .into_iter()
            .map(|diagnostic| diagnostic.message)
            .collect::<Vec<_>>()
            .join("; ")
    })?;

    Ok(TypeArg {
        name: name.to_owned(),
        ty,
    })
}

fn parse_value(input: &str) -> Result<Value, String> {
    let input = input.trim();
    if input == "true" {
        return Ok(Value::Bool(true));
    }
    if input == "false" {
        return Ok(Value::Bool(false));
    }
    if input == "nil" {
        return Ok(Value::Stack(Vec::new()));
    }
    if input.starts_with("stack[") && input.ends_with(']') {
        return parse_stack_value(input);
    }
    if input.starts_with('[') || input.ends_with(']') {
        return parse_array_value(input);
    }

    input
        .parse()
        .map(Value::Int)
        .map_err(|_| format!("`{input}` is not a signed i64 integer, boolean, array, or stack"))
}

fn parse_array_value(input: &str) -> Result<Value, String> {
    let inner = input
        .strip_prefix('[')
        .and_then(|input| input.strip_suffix(']'))
        .ok_or_else(|| format!("`{input}` is not an array literal; expected [VALUE,...]"))?;

    if inner.trim().is_empty() {
        return Ok(Value::Array(Vec::new()));
    }

    split_top_level_commas(inner)?
        .into_iter()
        .map(parse_value)
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn parse_stack_value(input: &str) -> Result<Value, String> {
    let inner = input
        .strip_prefix("stack[")
        .and_then(|input| input.strip_suffix(']'))
        .ok_or_else(|| format!("`{input}` is not a stack literal; expected stack[INT,...]"))?;

    parse_int_list(inner).map(Value::Stack)
}

fn parse_int_list(inner: &str) -> Result<Vec<i64>, String> {
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }

    inner
        .split(',')
        .map(|part| {
            let part = part.trim();
            part.parse()
                .map_err(|_| format!("`{part}` is not a signed i64 integer"))
        })
        .collect()
}

fn split_top_level_commas(input: &str) -> Result<Vec<&str>, String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0_i32;

    for (index, char) in input.char_indices() {
        match char {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth < 0 {
                    return Err("array literal has unmatched `]`".to_owned());
                }
            }
            ',' if depth == 0 => {
                parts.push(input[start..index].trim());
                start = index + char.len_utf8();
            }
            _ => {}
        }
    }

    if depth != 0 {
        return Err("array literal has unmatched `[`".to_owned());
    }

    parts.push(input[start..].trim());
    if let Some(empty) = parts.iter().find(|part| part.is_empty()) {
        return Err(format!("empty array element near `{empty}`"));
    }

    Ok(parts)
}

fn validate_seed_types(vars: &[VarArg], types: &[TypeArg]) -> Result<(), CliError> {
    for var in vars {
        let Some(ty) = types.iter().find(|ty| ty.name == var.name) else {
            continue;
        };

        if !value_matches_type(&var.value, &ty.ty.node) {
            return Err(CliError::SeedType(format!(
                "seed `{}` has value `{}` but type annotation expects `{}`",
                var.name,
                var.value,
                format_type_expr(&ty.ty.node)
            )));
        }
    }

    Ok(())
}

fn validate_seed_args(vars: &[VarArg], types: &[TypeArg]) -> Result<(), CliError> {
    reject_duplicate_names(vars.iter().map(|var| var.name.as_str()), "--var")?;
    validate_seed_value_shapes(vars)?;
    validate_type_args(types)
}

fn validate_watch_args(watches: &[String]) -> Result<(), CliError> {
    reject_duplicate_names(watches.iter().map(String::as_str), "--watch")?;
    for watch in watches {
        if !is_identifier(watch) {
            return Err(CliError::SeedType(format!(
                "`{watch}` is not a valid Reverie identifier"
            )));
        }
    }
    Ok(())
}

fn validate_watches_exist(
    timeline: &reverie_interp::Timeline,
    watches: &[String],
) -> Result<(), CliError> {
    let missing = watches
        .iter()
        .filter(|watch| {
            !timeline
                .frames()
                .iter()
                .any(|frame| frame.state.get(watch).is_some())
        })
        .cloned()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    Err(CliError::SeedType(format!(
        "unknown --watch name(s) `{}`; watched names must appear in the scrub timeline",
        missing.join("`, `")
    )))
}

fn validate_seed_value_shapes(vars: &[VarArg]) -> Result<(), CliError> {
    for var in vars {
        validate_value_shape(&format!("seed `{}`", var.name), &var.value)?;
    }
    Ok(())
}

fn validate_tape_value_shapes(flag: &str, values: &[Value]) -> Result<(), CliError> {
    for value in values {
        validate_value_shape(flag, value)?;
    }
    Ok(())
}

fn validate_value_shape(context: &str, value: &Value) -> Result<(), CliError> {
    let Value::Array(values) = value else {
        return Ok(());
    };
    for value in values {
        validate_value_shape(context, value)?;
    }
    if !values.is_empty() && seed_array_element_type(value).is_none() {
        return Err(CliError::SeedType(format!(
            "{context} has a heterogeneous array value; arrays must be homogeneous"
        )));
    }
    validate_rectangular_array_value(context, values)?;
    Ok(())
}

fn validate_rectangular_array_value(context: &str, values: &[Value]) -> Result<(), CliError> {
    let Some(Value::Array(first)) = values.first() else {
        return Ok(());
    };

    for value in &values[1..] {
        let Value::Array(row) = value else {
            continue;
        };
        if row.len() != first.len() {
            return Err(CliError::SeedType(format!(
                "{context} has a ragged array value; nested array rows must have matching lengths (expected {}, found {})",
                first.len(),
                row.len()
            )));
        }
    }

    Ok(())
}

fn types_with_seed_inference(vars: &[VarArg], explicit_types: &[TypeArg]) -> Vec<TypeArg> {
    let mut types = explicit_types.to_vec();
    for var in vars {
        if explicit_types.iter().any(|ty| ty.name == var.name) {
            continue;
        }
        let Some(ty) = type_from_seed_value(&var.value) else {
            continue;
        };
        types.push(TypeArg {
            name: var.name.clone(),
            ty: Spanned::new(ty, 0..0),
        });
    }
    types
}

fn type_from_seed_value(value: &Value) -> Option<TypeExpr> {
    match value {
        Value::Int(_) => Some(TypeExpr::Int { unit: None }),
        Value::Bool(_) => Some(TypeExpr::Bool),
        Value::Stack(_) => Some(TypeExpr::Stack),
        Value::Array(_) => {
            let element = seed_array_element_type(value)?;
            Some(TypeExpr::Array {
                element: Box::new(Spanned::new(element, 0..0)),
            })
        }
    }
}

fn seed_array_element_type(value: &Value) -> Option<TypeExpr> {
    let Value::Array(values) = value else {
        return type_from_seed_value(value);
    };
    let (first, rest) = values.split_first()?;
    let element = type_from_seed_value(first)?;
    for value in rest {
        if type_from_seed_value(value)? != element {
            return None;
        }
    }
    Some(element)
}

fn validate_external_seeds(
    file: &Path,
    program: &Program,
    vars: &[VarArg],
) -> Result<(), CliError> {
    let summary = summarize_program(program);
    let allowed = allowed_seed_names(&summary);
    if allowed.is_empty() {
        return Ok(());
    }

    let provided = vars
        .iter()
        .map(|var| var.name.as_str())
        .collect::<BTreeSet<_>>();
    let missing = summary
        .external_names
        .iter()
        .filter(|name| !provided.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let unexpected = provided
        .iter()
        .filter(|name| !allowed.contains(**name))
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    if missing.is_empty() && unexpected.is_empty() {
        return Ok(());
    }

    let mut parts = Vec::new();
    if !missing.is_empty() {
        parts.push(format!(
            "missing --var seed(s) for `{}`",
            missing.join("`, `")
        ));
    }
    if !unexpected.is_empty() {
        parts.push(format!(
            "unexpected --var seed(s) for `{}`",
            unexpected.join("`, `")
        ));
    }
    Err(CliError::SeedType(format!(
        "{}; run `reverie explain {}` for a seed template",
        parts.join("; "),
        shell_quote_arg(&file.display().to_string())
    )))
}

fn validate_external_types(
    file: &Path,
    program: &Program,
    types: &[TypeArg],
) -> Result<(), CliError> {
    let summary = summarize_program(program);
    let allowed = allowed_seed_names(&summary);
    if allowed.is_empty() {
        return Ok(());
    }

    let unexpected = types
        .iter()
        .filter(|ty| !allowed.contains(ty.name.as_str()))
        .map(|ty| ty.name.clone())
        .collect::<Vec<_>>();
    if unexpected.is_empty() {
        return Ok(());
    }

    Err(CliError::SeedType(format!(
        "unexpected --type annotation(s) for `{}`; run `reverie explain {}` for a type template",
        unexpected.join("`, `"),
        shell_quote_arg(&file.display().to_string())
    )))
}

fn allowed_seed_names(summary: &ProgramSummary) -> BTreeSet<String> {
    summary
        .external_names
        .iter()
        .cloned()
        .chain(summary.declared_names.iter().cloned())
        .collect()
}

fn validate_type_args(types: &[TypeArg]) -> Result<(), CliError> {
    reject_duplicate_names(types.iter().map(|ty| ty.name.as_str()), "--type")
}

fn reject_duplicate_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
    flag: &str,
) -> Result<(), CliError> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(CliError::SeedType(format!(
                "duplicate {flag} annotation for `{name}`"
            )));
        }
    }
    Ok(())
}

fn value_matches_type(value: &Value, ty: &TypeExpr) -> bool {
    match (value, ty) {
        (Value::Int(_), TypeExpr::Int { .. }) => true,
        (Value::Bool(_), TypeExpr::Bool) => true,
        (Value::Stack(_), TypeExpr::Stack) => true,
        (Value::Array(values), TypeExpr::Array { element }) => values
            .iter()
            .all(|value| value_matches_type(value, &element.node)),
        (value, TypeExpr::Tensor { element, shape }) => {
            tensor_value_matches_type(value, shape, &element.node)
        }
        (value, TypeExpr::Witness { inner }) => value_matches_type(value, &inner.node),
        _ => false,
    }
}

fn tensor_value_matches_type(value: &Value, shape: &[usize], element: &TypeExpr) -> bool {
    let Some((len, rest)) = shape.split_first() else {
        return value_matches_type(value, element);
    };
    let Value::Array(values) = value else {
        return false;
    };
    values.len() == *len
        && values
            .iter()
            .all(|value| tensor_value_matches_type(value, rest, element))
}

fn is_identifier(input: &str) -> bool {
    let mut chars = input.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|char| char == '_' || char.is_ascii_alphanumeric())
}

fn read_source(path: &Path) -> Result<String, CliError> {
    fs::read_to_string(path).map_err(|source| CliError::Read {
        path: path.to_path_buf(),
        source,
    })
}

fn parse_or_report(
    path: &Path,
    source: &str,
    legacy_janus: bool,
) -> Result<reverie_syntax::Program, CliError> {
    match parse_program_with_options(source, ParseOptions { legacy_janus }) {
        Ok(program) => Ok(program),
        Err(diagnostics) => {
            emit_diagnostics(path, source, &diagnostics)?;
            Err(CliError::ReportedDiagnostics)
        }
    }
}

fn check_or_report(
    path: &Path,
    source: &str,
    program: &reverie_syntax::Program,
    types: &[TypeArg],
    legacy_janus: bool,
) -> Result<(), CliError> {
    let types = types.iter().map(|ty| (ty.name.clone(), ty.ty.clone()));
    let options = reverie_core::CheckOptions {
        allow_update_aliases: legacy_janus,
    };
    match reverie_core::check_program_with_types_and_options(program, types, options) {
        Ok(()) => Ok(()),
        Err(error) => {
            emit_core_diagnostic(path, source, &error)?;
            Err(CliError::ReportedDiagnostics)
        }
    }
}

fn execution_options(legacy_janus: bool) -> ExecutionOptions {
    ExecutionOptions {
        allow_update_aliases: legacy_janus,
    }
}

fn runtime_or_report<T>(
    path: &Path,
    source: &str,
    run: impl FnOnce() -> Result<T, RuntimeError>,
) -> Result<T, CliError> {
    match run() {
        Ok(value) => Ok(value),
        Err(error) => {
            if error.span().is_some() || !error.labels().is_empty() {
                emit_runtime_diagnostic(path, source, &error)?;
                Err(CliError::ReportedDiagnostics)
            } else {
                Err(CliError::Runtime(error))
            }
        }
    }
}

fn emit_diagnostics(path: &Path, source: &str, diagnostics: &[SyntaxDiagnostic]) -> io::Result<()> {
    let file_name = path.display().to_string();

    for diagnostic in diagnostics {
        let span = diagnostic.span.clone();
        Report::build(ReportKind::Error, (file_name.clone(), span.clone()))
            .with_message(&diagnostic.message)
            .with_label(
                Label::new((file_name.clone(), span))
                    .with_message(&diagnostic.label)
                    .with_color(Color::Red),
            )
            .finish()
            .eprint((file_name.clone(), Source::from(source)))?;
    }

    Ok(())
}

fn emit_core_diagnostic(
    path: &Path,
    source: &str,
    error: &reverie_core::CoreError,
) -> io::Result<()> {
    let file_name = path.display().to_string();
    let span = error.span();
    let mut report = Report::build(ReportKind::Error, (file_name.clone(), span.clone()))
        .with_message(error.to_string());

    if error.labels().is_empty() {
        report = report.with_label(
            Label::new((file_name.clone(), span))
                .with_message("semantic check failed here")
                .with_color(Color::Red),
        );
    } else {
        for label in error.labels() {
            report = report.with_label(
                Label::new((file_name.clone(), label.span()))
                    .with_message(label.message())
                    .with_color(Color::Red),
            );
        }
    }

    report
        .finish()
        .eprint((file_name.clone(), Source::from(source)))
}

fn emit_runtime_diagnostic(path: &Path, source: &str, error: &RuntimeError) -> io::Result<()> {
    let file_name = path.display().to_string();
    let Some(span) = error
        .span()
        .or_else(|| error.labels().first().map(|label| label.span()))
    else {
        return Ok(());
    };
    let mut report = Report::build(ReportKind::Error, (file_name.clone(), span.clone()))
        .with_message(error.to_string())
        .with_label(
            Label::new((file_name.clone(), span))
                .with_message("runtime failed here")
                .with_color(Color::Red),
        );

    for label in error.labels() {
        report = report.with_label(
            Label::new((file_name.clone(), label.span()))
                .with_message(label.message())
                .with_color(Color::Red),
        );
    }

    report
        .finish()
        .eprint((file_name.clone(), Source::from(source)))
}

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand, ValueEnum};
use reverie_interp::{IoState, RuntimeError, State, Value};
use reverie_syntax::{
    BinaryOp, Expr, ParseOptions, Program, Spanned, SpannedExpr, SpannedStmt, SpannedType, Stmt,
    SyntaxDiagnostic, TypeExpr, UnaryOp, format_program, format_type_expr,
    parse_program_with_options, parse_type_expr,
};
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

        /// Parse with stricter legacy Janus surface rules.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Print the mechanically derived inverse source program.
    Invert {
        file: PathBuf,

        /// Annotate an external variable for static checking, e.g. --type distance=int<m>.
        #[arg(long = "type", value_name = "NAME=TYPE", value_parser = parse_type_arg)]
        types: Vec<TypeArg>,

        /// Parse with stricter legacy Janus surface rules.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Explain the reversible constructs used by a source file.
    Explain {
        file: PathBuf,

        /// Annotate an external variable for static checking, e.g. --type distance=int<m>.
        #[arg(long = "type", value_name = "NAME=TYPE", value_parser = parse_type_arg)]
        types: Vec<TypeArg>,

        /// Emit a machine-readable JSON summary.
        #[arg(long)]
        json: bool,

        /// Parse with stricter legacy Janus surface rules.
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

        /// Parse with stricter legacy Janus surface rules.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Run a Reverie source file.
    Run {
        file: PathBuf,

        /// Seed a variable before execution, e.g. --var n=7 or --var xs=[1,2,3].
        #[arg(long = "var", value_name = "NAME=VALUE", value_parser = parse_var)]
        vars: Vec<VarArg>,

        /// Annotate an external variable for static checking, e.g. --type distance=int<m>.
        #[arg(long = "type", value_name = "NAME=TYPE", value_parser = parse_type_arg)]
        types: Vec<TypeArg>,

        /// Select the execution engine.
        #[arg(long, value_enum, default_value_t = Engine::Slot)]
        engine: Engine,

        /// Seed a reversible input tape value for `read`.
        #[arg(long = "input", value_name = "VALUE", value_parser = parse_value)]
        inputs: Vec<Value>,

        /// Seed a reversible output tape value for inverse I/O operations.
        #[arg(long = "output", value_name = "VALUE", value_parser = parse_value)]
        outputs: Vec<Value>,

        /// Parse with stricter legacy Janus surface rules.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Run a Reverie source file backward from a seeded store.
    Reverse {
        file: PathBuf,

        /// Seed a variable before backward execution, e.g. --var n=7 or --var xs=[1,2,3].
        #[arg(long = "var", value_name = "NAME=VALUE", value_parser = parse_var)]
        vars: Vec<VarArg>,

        /// Annotate an external variable for static checking, e.g. --type distance=int<m>.
        #[arg(long = "type", value_name = "NAME=TYPE", value_parser = parse_type_arg)]
        types: Vec<TypeArg>,

        /// Select the execution engine.
        #[arg(long, value_enum, default_value_t = Engine::Slot)]
        engine: Engine,

        /// Seed a reversible input tape value for inverse I/O operations.
        #[arg(long = "input", value_name = "VALUE", value_parser = parse_value)]
        inputs: Vec<Value>,

        /// Seed a reversible output tape value for `unread`/`unwrite`.
        #[arg(long = "output", value_name = "VALUE", value_parser = parse_value)]
        outputs: Vec<Value>,

        /// Parse with stricter legacy Janus surface rules.
        #[arg(long)]
        legacy_janus: bool,
    },

    /// Scrub a Reverie source file through its forward timeline.
    Scrub {
        file: PathBuf,

        /// Seed a variable before execution, e.g. --var n=7 or --var xs=[1,2,3].
        #[arg(long = "var", value_name = "NAME=VALUE", value_parser = parse_var)]
        vars: Vec<VarArg>,

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

        /// Parse with stricter legacy Janus surface rules.
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
            check_or_report(&file, &source, &program, &types)?;
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
            check_or_report(&file, &source, &program, &types)?;
            let inverse = reverie_core::invert_program(&program);
            print!("{}", format_program(&inverse));
            Ok(())
        }
        Command::Explain {
            file,
            mut types,
            json,
            legacy_janus,
        } => {
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            normalize_legacy_seed_types(legacy_janus, &mut types);
            validate_type_args(&types)?;
            validate_external_types(&file, &program, &types)?;
            check_or_report(&file, &source, &program, &types)?;
            if json {
                print_explanation_json(&file, &program, &types);
            } else {
                print_explanation(&file, &program, &types);
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
            vars,
            types,
            engine,
            inputs,
            outputs,
            legacy_janus,
        } => {
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            let mut vars = vars;
            let mut types = types;
            normalize_legacy_seed_args(legacy_janus, &mut vars, &mut types);
            validate_seed_args(&vars, &types)?;
            validate_tape_value_shapes("--input", &inputs)?;
            validate_tape_value_shapes("--output", &outputs)?;
            validate_external_types(&file, &program, &types)?;
            validate_external_seeds(&file, &program, &vars)?;
            let types = types_with_seed_inference(&vars, &types);
            validate_seed_types(&vars, &types)?;
            check_or_report(&file, &source, &program, &types)?;
            let initial_state = state_from_vars(vars);
            let initial_io = IoState::with_output(initial_state, inputs, outputs);
            let state = runtime_or_report(&file, &source, || match engine {
                Engine::Tree => reverie_interp::execute_io(&program, initial_io),
                Engine::Slot => reverie_interp::execute_compiled_io(&program, initial_io),
            })?;
            print_io_state(&state);
            Ok(())
        }
        Command::Reverse {
            file,
            vars,
            types,
            engine,
            inputs,
            outputs,
            legacy_janus,
        } => {
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            let mut vars = vars;
            let mut types = types;
            normalize_legacy_seed_args(legacy_janus, &mut vars, &mut types);
            validate_seed_args(&vars, &types)?;
            validate_tape_value_shapes("--input", &inputs)?;
            validate_tape_value_shapes("--output", &outputs)?;
            validate_external_types(&file, &program, &types)?;
            validate_external_seeds(&file, &program, &vars)?;
            let types = types_with_seed_inference(&vars, &types);
            validate_seed_types(&vars, &types)?;
            check_or_report(&file, &source, &program, &types)?;
            let initial_state = state_from_vars(vars);
            let initial_io = IoState::with_output(initial_state, inputs, outputs);
            let state = runtime_or_report(&file, &source, || match engine {
                Engine::Tree => reverie_interp::execute_io_backward(&program, initial_io),
                Engine::Slot => reverie_interp::execute_compiled_io_backward(&program, initial_io),
            })?;
            print_io_state(&state);
            Ok(())
        }
        Command::Scrub {
            file,
            vars,
            types,
            watches,
            dump,
            inputs,
            outputs,
            legacy_janus,
        } => {
            let source = read_source(&file)?;
            let program = parse_or_report(&file, &source, legacy_janus)?;
            let mut vars = vars;
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
            check_or_report(&file, &source, &program, &types)?;
            let initial_state = state_from_vars(vars);
            let initial_io = IoState::with_output(initial_state, inputs, outputs);
            let timeline = runtime_or_report(&file, &source, || {
                reverie_interp::build_timeline_io(&program, initial_io)
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
    declared_names: BTreeSet<String>,
    declared_templates: BTreeMap<String, String>,
    external_names: BTreeSet<String>,
}

fn print_explanation(file: &Path, program: &Program, types: &[TypeArg]) {
    let summary = summarize_program(program);

    println!("file: {}", file.display());
    println!("status: reversible program checks");
    println!("globals: {}", program.globals.len());
    println!("procedures: {}", program.procedures.len());
    println!("statements: {}", summary.statements);
    println!("expressions: {}", summary.expressions);
    println!("features:");
    if summary.features.is_empty() {
        println!("- skip-only core");
    } else {
        for feature in &summary.features {
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
    println!(
        "inverse: reverie invert {}",
        shell_quote_arg(&file.display().to_string())
    );
}

fn print_explanation_json(file: &Path, program: &Program, types: &[TypeArg]) {
    let summary = summarize_program(program);
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

    println!("{{");
    println!("  \"file\": {},", json_string(&file));
    println!("  \"status\": \"reversible program checks\",");
    println!("  \"globals\": {},", program.globals.len());
    println!("  \"procedures\": {},", program.procedures.len());
    println!("  \"statements\": {},", summary.statements);
    println!("  \"expressions\": {},", summary.expressions);
    println!(
        "  \"features\": {},",
        json_string_array(summary.features.iter().copied())
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
    println!("  \"external_store\": [{}],", external_store.join(","));
    println!("  \"declared_store\": [{}],", declared_store.join(","));
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
        if global.len > 1 || !global.dims.is_empty() {
            summary.features.insert("fixed-size arrays");
        }
        if global.init.is_some() {
            summary.features.insert("source declarations");
        }
    }
    for procedure in &program.procedures {
        let mut scope = global_names.clone();
        scope.extend(procedure.params.iter().map(|param| param.name.clone()));
        visit_stmt(&procedure.body, &mut scope, &mut summary);
    }
    let mut scope = global_names;
    visit_stmt(&program.body, &mut scope, &mut summary);

    summary
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
    }
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
            init,
            body,
            delocal,
            ..
        } => {
            summary.features.insert("local/delocal");
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
            scope.insert(name.clone());
        }
    }
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
            }
            visit_expr(left, scope, summary);
            visit_expr(right, scope, summary);
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

fn print_io_state(state: &IoState) {
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
        _ => false,
    }
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
) -> Result<(), CliError> {
    let types = types.iter().map(|ty| (ty.name.clone(), ty.ty.clone()));
    match reverie_core::check_program_with_types(program, types) {
        Ok(()) => Ok(()),
        Err(error) => {
            emit_core_diagnostic(path, source, &error)?;
            Err(CliError::ReportedDiagnostics)
        }
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

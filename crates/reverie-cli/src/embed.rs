//! A small, stable surface for embedding Reverie's CLI behavior in other
//! front ends (the Python package uses it).
//!
//! Everything here reuses the exact code paths the `reverie` binary runs, so a
//! roundtrip proof produced through [`roundtrip_proof`] is byte-for-byte the
//! JSON that `reverie roundtrip --proof-output` writes for the same file path,
//! seeds and engine, and [`verify_roundtrip_proof`] is the check
//! `reverie verify-roundtrip` performs.

use std::fmt;
use std::path::Path;

use reverie_core::CoreError;
use reverie_interp::{IoState, RuntimeError, Value};
use reverie_syntax::{Program, SpannedType};
use serde_json::Value as JsonValue;

use crate::{
    Cli, CliError, Engine, TypeArg, VarArg, allowed_seed_names, execute_initialization_io,
    execution_options, explanation_json, normalize_legacy_seed_args, roundtrip_check,
    roundtrip_proof_json, run_cli, state_from_vars, summarize_program, types_with_seed_inference,
    validate_seed_types, validate_seed_value_shapes, witness_report,
};

/// Execution engine, matching the CLI's `--engine` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProofEngine {
    /// Correctness-first AST interpreter (`--engine tree`).
    Tree,
    /// Slot-indexed compiled interpreter (`--engine slot`, the CLI default).
    #[default]
    Slot,
}

impl From<ProofEngine> for Engine {
    fn from(engine: ProofEngine) -> Self {
        match engine {
            ProofEngine::Tree => Engine::Tree,
            ProofEngine::Slot => Engine::Slot,
        }
    }
}

/// Why an embedded operation failed.
#[derive(Debug)]
pub enum EmbedError {
    /// The seed store is malformed or does not fit the program's externals.
    Seed(String),
    /// Static checking failed once seed types were inferred.
    Check(CoreError),
    /// Execution failed.
    Runtime(RuntimeError),
    /// Anything else (I/O while replaying a proof, malformed proof fields).
    Other(String),
}

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Seed(message) | Self::Other(message) => f.write_str(message),
            Self::Check(error) => write!(f, "{error}"),
            Self::Runtime(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for EmbedError {}

impl From<CliError> for EmbedError {
    fn from(error: CliError) -> Self {
        match error {
            CliError::SeedType(message) => Self::Seed(message),
            CliError::Runtime(error) => Self::Runtime(error),
            other => Self::Other(other.to_string()),
        }
    }
}

/// How a program is checked and run, mirroring the CLI's flags.
#[derive(Debug, Clone, Default)]
pub struct EmbedOptions {
    /// `--engine`.
    pub engine: ProofEngine,
    /// `--legacy-janus`: Janus parsing and update-alias semantics.
    pub legacy_janus: bool,
    /// `--type NAME=TYPE` annotations for external variables.
    pub types: Vec<(String, SpannedType)>,
}

impl EmbedOptions {
    fn type_args(&self) -> Vec<TypeArg> {
        self.types
            .iter()
            .map(|(name, ty)| TypeArg {
                name: name.clone(),
                ty: ty.clone(),
            })
            .collect()
    }

    /// Execution options for the interpreter.
    pub fn execution(&self) -> reverie_interp::ExecutionOptions {
        execution_options(self.legacy_janus)
    }

    /// Check options for the checker.
    pub fn check(&self) -> reverie_core::CheckOptions {
        reverie_core::CheckOptions {
            allow_update_aliases: self.legacy_janus,
        }
    }
}

/// Validate a seed store the way `reverie run --var ... --type ...` does and
/// return the full set of external type annotations (explicit ones plus those
/// inferred from seeds), ready for
/// [`reverie_core::check_program_with_types_and_options`].
///
/// Seeds must be homogeneous, rectangular values; when the program reads
/// external variables, every one must be seeded and no unknown names may be
/// supplied.
pub fn seed_types(
    program: &Program,
    seeds: &[(String, Value)],
    options: &EmbedOptions,
) -> Result<Vec<(String, SpannedType)>, EmbedError> {
    let (_, types) = validated(program, seeds, options)?;
    Ok(types.into_iter().map(|ty| (ty.name, ty.ty)).collect())
}

/// Run `program` forward then backward from `seeds` and build the
/// `reverie_roundtrip_proof_v1` JSON document the CLI writes with
/// `--proof-output`. `file` is recorded as the proof's source path (it is not
/// read here); `source` must be the text `program` was parsed from.
///
/// A program that fails to restore its entry store still yields a proof, with
/// `"passed": false`.
pub fn roundtrip_proof(
    file: &Path,
    source: &str,
    program: &Program,
    seeds: &[(String, Value)],
    options: &EmbedOptions,
) -> Result<JsonValue, EmbedError> {
    let (vars, types) = validated(program, seeds, options)?;
    reverie_core::check_program_with_types_and_options(
        program,
        types.iter().map(|ty| (ty.name.clone(), ty.ty.clone())),
        options.check(),
    )
    .map_err(EmbedError::Check)?;

    let engine = Engine::from(options.engine);
    let execution = options.execution();
    let initial_io = IoState::with_output(state_from_vars(vars), [], []);
    let baseline = execute_initialization_io(program, initial_io.clone(), engine, execution)
        .map_err(EmbedError::Runtime)?;
    let forward = match engine {
        Engine::Tree => reverie_interp::execute_io_with_options(program, initial_io, execution),
        Engine::Slot => {
            reverie_interp::execute_compiled_io_with_options(program, initial_io, execution)
        }
    }
    .map_err(EmbedError::Runtime)?;
    let restored = match engine {
        Engine::Tree => {
            reverie_interp::execute_io_backward_with_options(program, forward.clone(), execution)
        }
        Engine::Slot => reverie_interp::execute_compiled_io_backward_with_options(
            program,
            forward.clone(),
            execution,
        ),
    }
    .map_err(EmbedError::Runtime)?;

    let summary = summarize_program(program);
    let witness_report = witness_report(&summary, &types);
    let check = roundtrip_check(&baseline, &restored, &summary);
    Ok(roundtrip_proof_json(crate::RoundtripProofInput {
        file,
        source,
        engine,
        legacy_janus: options.legacy_janus,
        types: &types,
        summary: &summary,
        witness_report: &witness_report,
        baseline: &baseline,
        forward: &forward,
        restored: &restored,
        check: &check,
        include_ml_profile: true,
    }))
}

/// The `reverie explain --json` document for a checked program.
pub fn explain_json(file: &Path, program: &Program, options: &EmbedOptions, ml: bool) -> String {
    explanation_json(file, program, &options.type_args(), ml)
}

/// Run the `reverie` command line with `args` (including the program name in
/// `args[0]`) and return its exit status. Help and usage errors are printed
/// as the binary prints them, without exiting the host process.
pub fn run_command_line(args: Vec<String>) -> i32 {
    use clap::Parser as _;
    match Cli::try_parse_from(args) {
        Ok(cli) => match run_cli(cli) {
            Ok(()) => 0,
            Err(CliError::ReportedDiagnostics) => 1,
            Err(error) => {
                eprintln!("error: {error}");
                1
            }
        },
        Err(error) => {
            let _ = error.print();
            error.exit_code()
        }
    }
}

/// Verify a roundtrip proof exactly as `reverie verify-roundtrip --json` does:
/// check every fingerprint, then, when the recorded source file is readable and
/// its hash matches, replay the program and compare fingerprints.
/// `proof_label` is recorded as the report's `proof_path`.
///
/// When `source` is given it is replayed instead of the recorded file (the
/// recorded `source_sha256` must still match), which lets callers verify
/// proofs of programs that never lived on disk.
pub fn verify_roundtrip_proof(
    proof: &JsonValue,
    proof_label: &str,
    source: Option<&str>,
) -> Result<JsonValue, EmbedError> {
    crate::verify_roundtrip_proof_with_source(Path::new(proof_label), proof, source)
        .map_err(EmbedError::from)
}

fn var_args(seeds: &[(String, Value)]) -> Vec<VarArg> {
    seeds
        .iter()
        .map(|(name, value)| VarArg {
            name: name.clone(),
            value: value.clone(),
        })
        .collect()
}

fn validated(
    program: &Program,
    seeds: &[(String, Value)],
    options: &EmbedOptions,
) -> Result<(Vec<VarArg>, Vec<TypeArg>), EmbedError> {
    let mut vars = var_args(seeds);
    let mut explicit = options.type_args();
    normalize_legacy_seed_args(options.legacy_janus, &mut vars, &mut explicit);
    validate_seed_value_shapes(&vars)?;
    validate_seed_names(program, &vars)?;
    validate_type_names(program, &explicit)?;
    let types = types_with_seed_inference(&vars, &explicit);
    validate_seed_types(&vars, &types)?;
    Ok((vars, types))
}

fn validate_type_names(program: &Program, types: &[TypeArg]) -> Result<(), EmbedError> {
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
        Ok(())
    } else {
        Err(EmbedError::Seed(format!(
            "type annotation(s) for unknown variable(s) `{}`",
            unexpected.join("`, `")
        )))
    }
}

/// Same rule as the CLI's `validate_external_seeds`, worded for callers that
/// pass a store rather than `--var` flags.
fn validate_seed_names(program: &Program, vars: &[VarArg]) -> Result<(), EmbedError> {
    let summary = summarize_program(program);
    let allowed = allowed_seed_names(&summary);
    if allowed.is_empty() {
        return Ok(());
    }
    let provided = vars
        .iter()
        .map(|var| var.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
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
    let mut parts = Vec::new();
    if !missing.is_empty() {
        parts.push(format!("missing seed(s) for `{}`", missing.join("`, `")));
    }
    if !unexpected.is_empty() {
        parts.push(format!(
            "unexpected seed(s) for `{}`",
            unexpected.join("`, `")
        ));
    }
    if parts.is_empty() {
        Ok(())
    } else {
        Err(EmbedError::Seed(parts.join("; ")))
    }
}

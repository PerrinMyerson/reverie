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
    CliError, Engine, TypeArg, VarArg, allowed_seed_names, execute_initialization_io,
    roundtrip_check, roundtrip_proof_json, state_from_vars, summarize_program,
    types_with_seed_inference, validate_seed_types, validate_seed_value_shapes, witness_report,
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

/// Validate a seed store the way `reverie run --var ...` does and return the
/// external type annotations inferred from it, ready for
/// [`reverie_core::check_program_with_types`].
///
/// Seeds must be homogeneous, rectangular values; when the program reads
/// external variables, every one must be seeded and no unknown names may be
/// supplied.
pub fn seed_types(
    program: &Program,
    seeds: &[(String, Value)],
) -> Result<Vec<(String, SpannedType)>, EmbedError> {
    let vars = var_args(seeds);
    let types = validated_types(program, &vars)?;
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
    engine: ProofEngine,
) -> Result<JsonValue, EmbedError> {
    let vars = var_args(seeds);
    let types = validated_types(program, &vars)?;
    reverie_core::check_program_with_types(
        program,
        types.iter().map(|ty| (ty.name.clone(), ty.ty.clone())),
    )
    .map_err(EmbedError::Check)?;

    let engine = Engine::from(engine);
    let options = reverie_interp::ExecutionOptions::default();
    let initial_io = IoState::with_output(state_from_vars(vars), [], []);
    let baseline = execute_initialization_io(program, initial_io.clone(), engine, options)
        .map_err(EmbedError::Runtime)?;
    let forward = match engine {
        Engine::Tree => reverie_interp::execute_io_with_options(program, initial_io, options),
        Engine::Slot => {
            reverie_interp::execute_compiled_io_with_options(program, initial_io, options)
        }
    }
    .map_err(EmbedError::Runtime)?;
    let restored = match engine {
        Engine::Tree => {
            reverie_interp::execute_io_backward_with_options(program, forward.clone(), options)
        }
        Engine::Slot => reverie_interp::execute_compiled_io_backward_with_options(
            program,
            forward.clone(),
            options,
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
        legacy_janus: false,
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

fn validated_types(program: &Program, vars: &[VarArg]) -> Result<Vec<TypeArg>, EmbedError> {
    validate_seed_value_shapes(vars)?;
    validate_seed_names(program, vars)?;
    let types = types_with_seed_inference(vars, &[]);
    validate_seed_types(vars, &types)?;
    Ok(types)
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

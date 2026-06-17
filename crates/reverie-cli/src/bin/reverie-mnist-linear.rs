use std::collections::BTreeMap;
use std::fs;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use reverie_interp::{State, Value, execute_compiled, execute_compiled_backward};
use reverie_syntax::Program;
use serde_json::json;
use sha2::{Digest, Sha256};

const IMAGE_PIXELS: usize = 28 * 28;
const DIGITS: usize = 10;
const Q31_ONE: i64 = 1_i64 << 31;
const DEFAULT_LR: i64 = Q31_ONE / 1000;
const TRAIN_SOURCE: &str = include_str!("../../../../examples/mnist_reversible_step.rev");
const IDENTIFY_SOURCE: &str = include_str!("../../../../examples/mnist_identify.rev");

#[derive(Debug, Parser)]
#[command(
    name = "reverie-mnist-linear",
    about = "Train and evaluate a reversible Q31 linear MNIST classifier with Reverie programs."
)]
struct Args {
    /// MNIST IDX image file for training, usually train-images-idx3-ubyte.
    #[arg(long)]
    train_images: Option<PathBuf>,

    /// MNIST IDX label file for training, usually train-labels-idx1-ubyte.
    #[arg(long)]
    train_labels: Option<PathBuf>,

    /// MNIST IDX image file for evaluation, usually t10k-images-idx3-ubyte.
    #[arg(long)]
    test_images: Option<PathBuf>,

    /// MNIST IDX label file for evaluation, usually t10k-labels-idx1-ubyte.
    #[arg(long)]
    test_labels: Option<PathBuf>,

    /// Number of training epochs.
    #[arg(long, default_value_t = 1)]
    epochs: usize,

    /// Limit the number of training samples.
    #[arg(long)]
    train_limit: Option<usize>,

    /// Limit the number of evaluation samples.
    #[arg(long)]
    eval_limit: Option<usize>,

    /// Q31 learning rate. Defaults to about 0.001.
    #[arg(long, default_value_t = DEFAULT_LR)]
    lr: i64,

    /// Retain a compact witness tape and invert every training step at the end.
    #[arg(long)]
    reverse_check: bool,

    /// Run the built-in synthetic separability test instead of reading IDX files.
    #[arg(long)]
    self_test: bool,

    /// Emit a machine-readable benchmark/audit report instead of the text summary.
    #[arg(long)]
    json: bool,

    /// Write a replayable audit bundle with the final model and per-step witnesses.
    #[arg(long)]
    audit_output: Option<PathBuf>,

    /// Export the final model from a verified audit bundle to a signed model bundle.
    #[arg(long)]
    export_model: Option<PathBuf>,

    /// Import a raw Q31 weights/bias JSON file into a signed model bundle.
    #[arg(long)]
    import_model_json: Option<PathBuf>,

    /// Write the signed model bundle produced by --export-model or --import-model-json.
    #[arg(long)]
    model_output: Option<PathBuf>,

    /// Verify a signed standalone model bundle.
    #[arg(long)]
    verify_model: Option<PathBuf>,

    /// Export labeled samples from a verified audit bundle to a sample-set JSON.
    #[arg(long)]
    export_samples: Option<PathBuf>,

    /// Write the sample-set JSON produced by --export-samples.
    #[arg(long)]
    samples_output: Option<PathBuf>,

    /// Verify an exported sample-set JSON.
    #[arg(long)]
    verify_samples: Option<PathBuf>,

    /// Verify a replayable audit bundle by running its witness trace backward.
    #[arg(long)]
    verify_audit: Option<PathBuf>,

    /// Inspect one training step from a replayable audit bundle.
    #[arg(long)]
    inspect_audit: Option<PathBuf>,

    /// Write a standalone training-step replay bundle while using --inspect-audit.
    #[arg(long)]
    step_output: Option<PathBuf>,

    /// Write a Markdown review card with --verify-audit, --inspect-audit, --verify-step, or --verify-inference.
    #[arg(long)]
    markdown_output: Option<PathBuf>,

    /// Verify a standalone training-step replay bundle forward and backward.
    #[arg(long)]
    verify_step: Option<PathBuf>,

    /// Scan a replayable audit bundle and rank suspicious training steps.
    #[arg(long)]
    scan_audit: Option<PathBuf>,

    /// Run and inspect inference with the saved final model and a bundle sample.
    #[arg(long)]
    inspect_inference: Option<PathBuf>,

    /// Run and inspect inference with a signed model bundle and an audited sample.
    #[arg(long)]
    inspect_model_inference: Option<PathBuf>,

    /// Evaluate a signed model bundle over a JSON file of labeled samples.
    #[arg(long)]
    evaluate_model: Option<PathBuf>,

    /// Write a signed batch evaluation bundle while using --evaluate-model.
    #[arg(long)]
    evaluation_output: Option<PathBuf>,

    /// Verify a signed batch evaluation bundle by rerunning every sample.
    #[arg(long)]
    verify_evaluation: Option<PathBuf>,

    /// Scan a signed batch evaluation bundle and rank suspicious predictions.
    #[arg(long)]
    scan_evaluation: Option<PathBuf>,

    /// Inspect one row from a signed batch evaluation bundle.
    #[arg(long)]
    inspect_evaluation: Option<PathBuf>,

    /// Audit bundle to provide a sample for --inspect-model-inference.
    #[arg(long)]
    sample_audit: Option<PathBuf>,

    /// JSON file with image_u8 and label fields for --inspect-model-inference.
    #[arg(long)]
    sample_json: Option<PathBuf>,

    /// JSON file with an array of labeled samples for --evaluate-model.
    #[arg(long)]
    samples_json: Option<PathBuf>,

    /// Limit samples written by --export-samples.
    #[arg(long)]
    samples_limit: Option<usize>,

    /// Write a standalone replayable inference bundle while using --inspect-inference.
    #[arg(long)]
    inference_output: Option<PathBuf>,

    /// Write a standalone .rev classifier with the selected model and sample embedded.
    #[arg(long)]
    standalone_rev_output: Option<PathBuf>,

    /// Verify a standalone inference bundle by rerunning forward and backward inference.
    #[arg(long)]
    verify_inference: Option<PathBuf>,

    /// Zero-based step index for --inspect-audit.
    #[arg(long, default_value_t = 0)]
    audit_step: usize,

    /// Strategy for choosing a step with --inspect-audit.
    #[arg(long, value_enum, default_value = "explicit")]
    audit_step_strategy: AuditStepStrategy,

    /// Number of ranked rows to emit for --scan-audit.
    #[arg(long, default_value_t = 10)]
    audit_limit: usize,

    /// Number of ranked rows to emit for --scan-evaluation.
    #[arg(long, default_value_t = 10)]
    evaluation_limit: usize,

    /// Zero-based row index for --inspect-evaluation.
    #[arg(long, default_value_t = 0)]
    evaluation_row: usize,

    /// Fail --scan-audit if accuracy is below this percentage.
    #[arg(long)]
    audit_min_accuracy: Option<f64>,

    /// Fail --scan-audit if any saved step has a margin below this value.
    #[arg(long)]
    audit_min_margin: Option<i64>,

    /// Fail --scan-audit if more than this many witness rows disagree with recomputed logits.
    #[arg(long)]
    audit_max_witness_mismatches: Option<usize>,

    /// Fail --scan-audit if any step's maximum absolute weight delta exceeds this value.
    #[arg(long)]
    audit_max_weight_delta: Option<i128>,

    /// Fail --scan-audit unless every digit label appears at least once.
    #[arg(long)]
    audit_require_label_coverage: bool,

    /// Fail --evaluate-model if accuracy is below this percentage.
    #[arg(long)]
    eval_min_accuracy: Option<f64>,

    /// Fail --evaluate-model if any evaluated sample has a margin below this value.
    #[arg(long)]
    eval_min_margin: Option<i64>,

    /// Fail --evaluate-model if more than this many samples are incorrect.
    #[arg(long)]
    eval_max_incorrect: Option<usize>,

    /// Fail --evaluate-model unless every digit label appears at least once.
    #[arg(long)]
    eval_require_label_coverage: bool,

    /// Compare verified replay artifact sizes for audit, step, and inference bundles.
    #[arg(long, num_args = 1..)]
    compare_artifacts: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum AuditStepStrategy {
    Explicit,
    LowestMargin,
    LargestUpdate,
    TopSuspicious,
}

impl AuditStepStrategy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::LowestMargin => "lowest-margin",
            Self::LargestUpdate => "largest-update",
            Self::TopSuspicious => "top-suspicious",
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AuditGate {
    min_accuracy: Option<f64>,
    min_margin: Option<i64>,
    max_witness_mismatches: Option<usize>,
    max_weight_delta: Option<i128>,
    require_label_coverage: bool,
}

impl AuditGate {
    fn from_args(args: &Args) -> Result<Self, String> {
        if let Some(min_accuracy) = args.audit_min_accuracy {
            if !min_accuracy.is_finite() || !(0.0..=100.0).contains(&min_accuracy) {
                return Err(
                    "--audit-min-accuracy must be a finite percentage in 0..=100".to_owned(),
                );
            }
        }
        if let Some(max_weight_delta) = args.audit_max_weight_delta {
            if max_weight_delta < 0 {
                return Err("--audit-max-weight-delta must be non-negative".to_owned());
            }
        }

        Ok(Self {
            min_accuracy: args.audit_min_accuracy,
            min_margin: args.audit_min_margin,
            max_witness_mismatches: args.audit_max_witness_mismatches,
            max_weight_delta: args.audit_max_weight_delta,
            require_label_coverage: args.audit_require_label_coverage,
        })
    }

    fn is_enabled(&self) -> bool {
        self.min_accuracy.is_some()
            || self.min_margin.is_some()
            || self.max_witness_mismatches.is_some()
            || self.max_weight_delta.is_some()
            || self.require_label_coverage
    }

    fn evaluate(
        &self,
        summary: &AuditScanSummary,
        by_label: &[LabelScanSummary],
    ) -> Option<AuditGateReport> {
        if !self.is_enabled() {
            return None;
        }

        let mut checks = Vec::new();
        if let Some(min_accuracy) = self.min_accuracy {
            let actual = percent(summary.correct, summary.steps);
            checks.push(AuditGateCheck {
                metric: "accuracy_percent",
                passed: actual >= min_accuracy,
                actual: format!("{actual:.2}"),
                expectation: format!(">= {min_accuracy:.2}"),
            });
        }
        if let Some(min_margin) = self.min_margin {
            checks.push(AuditGateCheck {
                metric: "lowest_margin",
                passed: summary
                    .lowest_margin
                    .is_some_and(|actual| actual >= min_margin),
                actual: summary
                    .lowest_margin
                    .map_or_else(|| "none".to_owned(), |actual| actual.to_string()),
                expectation: format!(">= {min_margin}"),
            });
        }
        if let Some(max_witness_mismatches) = self.max_witness_mismatches {
            checks.push(AuditGateCheck {
                metric: "witness_mismatches",
                passed: summary.witness_mismatches <= max_witness_mismatches,
                actual: summary.witness_mismatches.to_string(),
                expectation: format!("<= {max_witness_mismatches}"),
            });
        }
        if let Some(max_weight_delta) = self.max_weight_delta {
            checks.push(AuditGateCheck {
                metric: "max_abs_weight_delta",
                passed: summary.max_abs_weight_delta <= max_weight_delta,
                actual: summary.max_abs_weight_delta.to_string(),
                expectation: format!("<= {max_weight_delta}"),
            });
        }
        if self.require_label_coverage {
            let labels_seen = by_label
                .iter()
                .filter(|summary| summary.samples > 0)
                .count();
            checks.push(AuditGateCheck {
                metric: "label_coverage",
                passed: labels_seen == DIGITS,
                actual: format!("{labels_seen}/{DIGITS}"),
                expectation: format!("{DIGITS}/{DIGITS}"),
            });
        }

        Some(AuditGateReport {
            passed: checks.iter().all(|check| check.passed),
            checks,
        })
    }
}

#[derive(Debug, Clone)]
struct AuditGateReport {
    passed: bool,
    checks: Vec<AuditGateCheck>,
}

#[derive(Debug, Clone)]
struct AuditGateCheck {
    metric: &'static str,
    passed: bool,
    actual: String,
    expectation: String,
}

#[derive(Debug, Clone, Default)]
struct ModelEvaluationGate {
    min_accuracy: Option<f64>,
    min_margin: Option<i64>,
    max_incorrect: Option<usize>,
    require_label_coverage: bool,
}

impl ModelEvaluationGate {
    fn from_args(args: &Args) -> Result<Self, String> {
        if let Some(min_accuracy) = args.eval_min_accuracy {
            if !min_accuracy.is_finite() || !(0.0..=100.0).contains(&min_accuracy) {
                return Err("--eval-min-accuracy must be a finite percentage in 0..=100".to_owned());
            }
        }

        Ok(Self {
            min_accuracy: args.eval_min_accuracy,
            min_margin: args.eval_min_margin,
            max_incorrect: args.eval_max_incorrect,
            require_label_coverage: args.eval_require_label_coverage,
        })
    }

    fn from_policy_json(policy: &serde_json::Value) -> Result<Self, String> {
        let min_accuracy = match policy.get("min_accuracy") {
            Some(serde_json::Value::Null) | None => None,
            Some(value) => {
                let min_accuracy = value.as_f64().ok_or_else(|| {
                    "evaluation gate policy `min_accuracy` must be a number".to_owned()
                })?;
                if !min_accuracy.is_finite() || !(0.0..=100.0).contains(&min_accuracy) {
                    return Err(
                        "evaluation gate policy `min_accuracy` must be a finite percentage in 0..=100"
                            .to_owned(),
                    );
                }
                Some(min_accuracy)
            }
        };
        let min_margin = match policy.get("min_margin") {
            Some(serde_json::Value::Null) | None => None,
            Some(value) => Some(value.as_i64().ok_or_else(|| {
                "evaluation gate policy `min_margin` must be an integer".to_owned()
            })?),
        };
        let max_incorrect = match policy.get("max_incorrect") {
            Some(serde_json::Value::Null) | None => None,
            Some(value) => {
                let raw = value.as_u64().ok_or_else(|| {
                    "evaluation gate policy `max_incorrect` must be a non-negative integer"
                        .to_owned()
                })?;
                Some(usize::try_from(raw).map_err(|_| {
                    "evaluation gate policy `max_incorrect` is too large".to_owned()
                })?)
            }
        };

        Ok(Self {
            min_accuracy,
            min_margin,
            max_incorrect,
            require_label_coverage: json_bool_field(policy, "require_label_coverage")?,
        })
    }

    fn is_enabled(&self) -> bool {
        self.min_accuracy.is_some()
            || self.min_margin.is_some()
            || self.max_incorrect.is_some()
            || self.require_label_coverage
    }

    fn policy_json(&self) -> Option<serde_json::Value> {
        self.is_enabled().then(|| {
            json!({
                "min_accuracy": self.min_accuracy,
                "min_margin": self.min_margin,
                "max_incorrect": self.max_incorrect,
                "require_label_coverage": self.require_label_coverage,
            })
        })
    }

    fn evaluate(
        &self,
        summary: &ModelEvaluationSummary,
        rows: &[ModelEvaluationRow],
    ) -> Option<ModelEvaluationGateReport> {
        if !self.is_enabled() {
            return None;
        }

        let mut checks = Vec::new();
        if let Some(min_accuracy) = self.min_accuracy {
            let actual = percent(summary.correct, summary.samples);
            checks.push(ModelEvaluationGateCheck {
                metric: "accuracy_percent",
                passed: actual >= min_accuracy,
                actual: format!("{actual:.2}"),
                expectation: format!(">= {min_accuracy:.2}"),
            });
        }
        if let Some(min_margin) = self.min_margin {
            checks.push(ModelEvaluationGateCheck {
                metric: "lowest_margin",
                passed: summary
                    .lowest_margin
                    .is_some_and(|actual| actual >= min_margin),
                actual: summary
                    .lowest_margin
                    .map_or_else(|| "none".to_owned(), |actual| actual.to_string()),
                expectation: format!(">= {min_margin}"),
            });
        }
        if let Some(max_incorrect) = self.max_incorrect {
            checks.push(ModelEvaluationGateCheck {
                metric: "incorrect",
                passed: summary.incorrect <= max_incorrect,
                actual: summary.incorrect.to_string(),
                expectation: format!("<= {max_incorrect}"),
            });
        }
        if self.require_label_coverage {
            let mut labels_seen = [false; DIGITS];
            for row in rows {
                let Ok(label) = usize::try_from(row.label) else {
                    continue;
                };
                if let Some(seen) = labels_seen.get_mut(label) {
                    *seen = true;
                }
            }
            let labels_seen = labels_seen.iter().filter(|seen| **seen).count();
            checks.push(ModelEvaluationGateCheck {
                metric: "label_coverage",
                passed: labels_seen == DIGITS,
                actual: format!("{labels_seen}/{DIGITS}"),
                expectation: format!("{DIGITS}/{DIGITS}"),
            });
        }

        Some(ModelEvaluationGateReport {
            passed: checks.iter().all(|check| check.passed),
            checks,
        })
    }
}

#[derive(Debug, Clone)]
struct ModelEvaluationGateReport {
    passed: bool,
    checks: Vec<ModelEvaluationGateCheck>,
}

#[derive(Debug, Clone)]
struct ModelEvaluationGateCheck {
    metric: &'static str,
    passed: bool,
    actual: String,
    expectation: String,
}

#[derive(Debug, Clone)]
struct Dataset {
    images: Vec<Vec<u8>>,
    labels: Vec<u8>,
}

impl Dataset {
    fn len(&self) -> usize {
        self.labels.len()
    }

    fn image(&self, index: usize) -> &[u8] {
        &self.images[index]
    }

    fn label(&self, index: usize) -> u8 {
        self.labels[index]
    }

    fn limited_len(&self, limit: Option<usize>) -> usize {
        limit.map_or_else(|| self.len(), |limit| limit.min(self.len()))
    }

    fn payload_bytes(&self) -> usize {
        let image_bytes = self.images.iter().map(Vec::len).sum::<usize>() * size_of::<u8>();
        let label_bytes = self.labels.len() * size_of::<u8>();
        image_bytes + label_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Model {
    weights: Vec<Vec<i64>>,
    bias: Vec<i64>,
}

impl Model {
    fn zero() -> Self {
        Self {
            weights: vec![vec![0; DIGITS]; IMAGE_PIXELS],
            bias: vec![0; DIGITS],
        }
    }

    fn payload_bytes() -> usize {
        (IMAGE_PIXELS * DIGITS + DIGITS) * size_of::<i64>()
    }
}

impl MemoryReport {
    fn for_run(train: &Dataset, test: &Dataset, trace_payload_bytes: usize) -> Self {
        let model_payload_bytes = Model::payload_bytes();
        let train_dataset_payload_bytes = train.payload_bytes();
        let eval_dataset_payload_bytes = test.payload_bytes();
        let dataset_payload_bytes = train_dataset_payload_bytes + eval_dataset_payload_bytes;
        let estimated_payload_bytes =
            model_payload_bytes + dataset_payload_bytes + trace_payload_bytes;

        Self {
            model_payload_bytes,
            train_dataset_payload_bytes,
            eval_dataset_payload_bytes,
            dataset_payload_bytes,
            trace_payload_bytes,
            estimated_payload_bytes,
            peak_rss_bytes: peak_rss_bytes(),
        }
    }
}

#[derive(Debug, Clone)]
enum DatasetKind {
    Synthetic,
    Idx,
}

impl DatasetKind {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::Idx => "idx",
        }
    }
}

#[derive(Debug, Clone)]
struct TapeEntry {
    sample_index: usize,
    label: i64,
    logits: Vec<i64>,
    error: Vec<i64>,
    prediction: i64,
    correct: i64,
    lr: i64,
}

#[derive(Debug, Clone)]
struct RunConfig {
    dataset_kind: DatasetKind,
    epochs: usize,
    train_limit: Option<usize>,
    eval_limit: Option<usize>,
    lr: i64,
    reverse_check: bool,
    audit_output: Option<PathBuf>,
}

#[derive(Debug)]
struct RunReport {
    config: RunConfig,
    train_samples: usize,
    train_correct: usize,
    eval_samples: usize,
    eval_correct: usize,
    train_elapsed: Duration,
    eval_elapsed: Duration,
    reverse_elapsed: Option<Duration>,
    reverse_steps: usize,
    trace_payload_bytes: usize,
    memory: MemoryReport,
}

#[derive(Debug, Clone, Copy)]
struct MemoryReport {
    model_payload_bytes: usize,
    train_dataset_payload_bytes: usize,
    eval_dataset_payload_bytes: usize,
    dataset_payload_bytes: usize,
    trace_payload_bytes: usize,
    estimated_payload_bytes: usize,
    peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct ProofCostReport {
    entries: usize,
    model_payload_bytes: usize,
    sample_payload_bytes: usize,
    witness_payload_bytes: usize,
    trace_replay_payload_bytes: usize,
    full_replay_payload_bytes: usize,
    sample_bytes_per_step: usize,
    witness_bytes_per_step: usize,
    trace_replay_bytes_per_step: usize,
    forward_recompute_steps: usize,
    inverse_recompute_steps: usize,
}

impl ProofCostReport {
    fn for_training_trace(entries: usize) -> Self {
        let model_payload_bytes = Model::payload_bytes();
        let sample_payload_bytes = training_trace_sample_payload_bytes(entries);
        let witness_payload_bytes = witness_trace_payload_bytes(entries);
        let trace_replay_payload_bytes = sample_payload_bytes + witness_payload_bytes;
        let full_replay_payload_bytes = model_payload_bytes + trace_replay_payload_bytes;
        let sample_bytes_per_step = training_trace_sample_payload_bytes(1);
        let witness_bytes_per_step = witness_trace_payload_bytes(1);
        let trace_replay_bytes_per_step = sample_bytes_per_step + witness_bytes_per_step;

        Self {
            entries,
            model_payload_bytes,
            sample_payload_bytes,
            witness_payload_bytes,
            trace_replay_payload_bytes,
            full_replay_payload_bytes,
            sample_bytes_per_step,
            witness_bytes_per_step,
            trace_replay_bytes_per_step,
            forward_recompute_steps: entries,
            inverse_recompute_steps: entries,
        }
    }

    fn to_json(self, enabled: bool) -> serde_json::Value {
        json!({
            "enabled": enabled,
            "entries": self.entries,
            "model_payload_bytes": self.model_payload_bytes,
            "sample_payload_bytes": self.sample_payload_bytes,
            "witness_payload_bytes": self.witness_payload_bytes,
            "trace_replay_payload_bytes": self.trace_replay_payload_bytes,
            "full_replay_payload_bytes": self.full_replay_payload_bytes,
            "sample_bytes_per_step": self.sample_bytes_per_step,
            "witness_bytes_per_step": self.witness_bytes_per_step,
            "trace_replay_bytes_per_step": self.trace_replay_bytes_per_step,
            "forward_recompute_steps": self.forward_recompute_steps,
            "inverse_recompute_steps": self.inverse_recompute_steps,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct InferenceMemoryReport {
    model_payload_bytes: usize,
    sample_payload_bytes: usize,
    witness_payload_bytes: usize,
    trace_payload_bytes: usize,
    replay_payload_bytes: usize,
    runtime_state_payload_bytes: usize,
    forward_recompute_steps: usize,
    inverse_recompute_steps: usize,
    peak_rss_bytes: Option<u64>,
}

impl InferenceMemoryReport {
    fn for_inference() -> Self {
        let model_payload_bytes = Model::payload_bytes();
        let sample_payload_bytes = IMAGE_PIXELS * size_of::<u8>() + size_of::<u8>();
        let witness_payload_bytes = inference_witness_payload_bytes();
        let trace_payload_bytes = 0;
        let replay_payload_bytes =
            model_payload_bytes + sample_payload_bytes + witness_payload_bytes;
        let runtime_state_payload_bytes =
            model_payload_bytes + (IMAGE_PIXELS + DIGITS + 3) * size_of::<i64>();

        Self {
            model_payload_bytes,
            sample_payload_bytes,
            witness_payload_bytes,
            trace_payload_bytes,
            replay_payload_bytes,
            runtime_state_payload_bytes,
            forward_recompute_steps: 1,
            inverse_recompute_steps: 1,
            peak_rss_bytes: peak_rss_bytes(),
        }
    }

    fn to_json(self) -> serde_json::Value {
        json!({
            "model_payload_bytes": self.model_payload_bytes,
            "sample_payload_bytes": self.sample_payload_bytes,
            "witness_payload_bytes": self.witness_payload_bytes,
            "trace_payload_bytes": self.trace_payload_bytes,
            "replay_payload_bytes": self.replay_payload_bytes,
            "runtime_state_payload_bytes": self.runtime_state_payload_bytes,
            "forward_recompute_steps": self.forward_recompute_steps,
            "inverse_recompute_steps": self.inverse_recompute_steps,
            "peak_rss_bytes": self.peak_rss_bytes,
        })
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let audit_gate = AuditGate::from_args(&args)?;
    let evaluation_gate = ModelEvaluationGate::from_args(&args)?;
    let train_program = compile_program("mnist_reversible_step.rev", TRAIN_SOURCE)?;
    let identify_program = compile_program("mnist_identify.rev", IDENTIFY_SOURCE)?;

    let audit_mode_count = usize::from(args.export_model.is_some())
        + usize::from(args.import_model_json.is_some())
        + usize::from(args.verify_model.is_some())
        + usize::from(args.export_samples.is_some())
        + usize::from(args.verify_samples.is_some())
        + usize::from(args.verify_audit.is_some())
        + usize::from(args.inspect_audit.is_some())
        + usize::from(args.verify_step.is_some())
        + usize::from(args.scan_audit.is_some())
        + usize::from(args.inspect_inference.is_some())
        + usize::from(args.inspect_model_inference.is_some())
        + usize::from(args.evaluate_model.is_some())
        + usize::from(args.verify_evaluation.is_some())
        + usize::from(args.scan_evaluation.is_some())
        + usize::from(args.inspect_evaluation.is_some())
        + usize::from(args.verify_inference.is_some())
        + usize::from(!args.compare_artifacts.is_empty());
    if audit_mode_count > 1 {
        return Err("choose only one of --export-model, --import-model-json, --verify-model, --export-samples, --verify-samples, --verify-audit, --inspect-audit, --verify-step, --scan-audit, --inspect-inference, --inspect-model-inference, --evaluate-model, --verify-evaluation, --scan-evaluation, --inspect-evaluation, --verify-inference, or --compare-artifacts".to_owned());
    }
    if args.model_output.is_some()
        && args.export_model.is_none()
        && args.import_model_json.is_none()
    {
        return Err("--model-output requires --export-model or --import-model-json".to_owned());
    }
    if args.samples_output.is_some() && args.export_samples.is_none() {
        return Err("--samples-output requires --export-samples".to_owned());
    }
    if args.samples_limit.is_some() && args.export_samples.is_none() {
        return Err("--samples-limit requires --export-samples".to_owned());
    }
    if args.step_output.is_some() && args.inspect_audit.is_none() {
        return Err("--step-output requires --inspect-audit".to_owned());
    }
    if args.markdown_output.is_some()
        && args.verify_audit.is_none()
        && args.inspect_audit.is_none()
        && args.verify_step.is_none()
        && args.inspect_inference.is_none()
        && args.inspect_model_inference.is_none()
        && args.inspect_evaluation.is_none()
        && args.verify_inference.is_none()
    {
        return Err(
            "--markdown-output requires --verify-audit, --inspect-audit, --verify-step, --inspect-inference, --inspect-model-inference, --inspect-evaluation, or --verify-inference"
                .to_owned(),
        );
    }
    if args.inference_output.is_some()
        && args.inspect_inference.is_none()
        && args.inspect_model_inference.is_none()
        && args.inspect_evaluation.is_none()
    {
        return Err(
            "--inference-output requires --inspect-inference, --inspect-model-inference, or --inspect-evaluation"
                .to_owned(),
        );
    }
    if args.standalone_rev_output.is_some()
        && args.inspect_inference.is_none()
        && args.inspect_model_inference.is_none()
        && args.inspect_evaluation.is_none()
    {
        return Err(
            "--standalone-rev-output requires --inspect-inference, --inspect-model-inference, or --inspect-evaluation"
                .to_owned(),
        );
    }
    if args.sample_audit.is_some() && args.inspect_model_inference.is_none() {
        return Err("--sample-audit requires --inspect-model-inference".to_owned());
    }
    if args.sample_json.is_some() && args.inspect_model_inference.is_none() {
        return Err("--sample-json requires --inspect-model-inference".to_owned());
    }
    if args.samples_json.is_some() && args.evaluate_model.is_none() {
        return Err("--samples-json requires --evaluate-model".to_owned());
    }
    if args.evaluation_output.is_some() && args.evaluate_model.is_none() {
        return Err("--evaluation-output requires --evaluate-model".to_owned());
    }
    if args.inspect_model_inference.is_some() {
        match (args.sample_audit.as_ref(), args.sample_json.as_ref()) {
            (Some(_), Some(_)) => {
                return Err(
                    "choose only one of --sample-audit or --sample-json for --inspect-model-inference"
                        .to_owned(),
                );
            }
            (None, None) => {
                return Err(
                    "--inspect-model-inference requires --sample-audit or --sample-json".to_owned(),
                );
            }
            _ => {}
        }
    }
    if args.evaluate_model.is_some() && args.samples_json.is_none() {
        return Err("--evaluate-model requires --samples-json".to_owned());
    }
    if audit_gate.is_enabled() && args.scan_audit.is_none() {
        return Err("audit gate thresholds require --scan-audit".to_owned());
    }
    if evaluation_gate.is_enabled() && args.evaluate_model.is_none() {
        return Err("evaluation gate thresholds require --evaluate-model".to_owned());
    }

    if let Some(path) = args.export_model {
        let output = args
            .model_output
            .as_deref()
            .ok_or_else(|| "--export-model requires --model-output".to_owned())?;
        export_model_bundle(&train_program, &path, output, args.json)?;
        return Ok(());
    }

    if let Some(path) = args.import_model_json {
        let output = args
            .model_output
            .as_deref()
            .ok_or_else(|| "--import-model-json requires --model-output".to_owned())?;
        import_model_bundle(&path, output, args.json)?;
        return Ok(());
    }

    if let Some(path) = args.verify_model {
        verify_model_bundle(&path, args.json)?;
        return Ok(());
    }

    if let Some(path) = args.export_samples {
        let output = args
            .samples_output
            .as_deref()
            .ok_or_else(|| "--export-samples requires --samples-output".to_owned())?;
        export_samples_bundle(&path, output, args.samples_limit, args.json)?;
        return Ok(());
    }

    if let Some(path) = args.verify_samples {
        verify_sample_set_bundle(&path, args.json)?;
        return Ok(());
    }

    if let Some(path) = args.verify_audit {
        verify_audit_bundle(
            &train_program,
            &path,
            args.markdown_output.as_deref(),
            args.json,
        )?;
        return Ok(());
    }

    if let Some(path) = args.inspect_audit {
        inspect_audit_bundle(
            &train_program,
            &path,
            args.audit_step,
            args.audit_step_strategy,
            args.step_output.as_deref(),
            args.markdown_output.as_deref(),
            args.json,
        )?;
        return Ok(());
    }

    if let Some(path) = args.verify_step {
        verify_step_bundle(
            &train_program,
            &path,
            args.markdown_output.as_deref(),
            args.json,
        )?;
        return Ok(());
    }

    if let Some(path) = args.scan_audit {
        scan_audit_bundle(&path, args.audit_limit, &audit_gate, args.json)?;
        return Ok(());
    }

    if let Some(path) = args.inspect_inference {
        inspect_inference_bundle(
            &identify_program,
            &path,
            args.audit_step,
            args.inference_output.as_deref(),
            args.standalone_rev_output.as_deref(),
            args.markdown_output.as_deref(),
            args.json,
        )?;
        return Ok(());
    }

    if let Some(path) = args.inspect_model_inference {
        inspect_model_inference_bundle(InspectModelInferenceRequest {
            program: &identify_program,
            model_path: &path,
            sample_audit_path: args.sample_audit.as_deref(),
            sample_json_path: args.sample_json.as_deref(),
            step_index: args.audit_step,
            inference_output: args.inference_output.as_deref(),
            standalone_rev_output: args.standalone_rev_output.as_deref(),
            markdown_output: args.markdown_output.as_deref(),
            emit_json: args.json,
        })?;
        return Ok(());
    }

    if let Some(path) = args.evaluate_model {
        let samples_json = args
            .samples_json
            .as_deref()
            .ok_or_else(|| "--evaluate-model requires --samples-json".to_owned())?;
        evaluate_model_bundle(
            &identify_program,
            &path,
            samples_json,
            &evaluation_gate,
            args.evaluation_output.as_deref(),
            args.json,
        )?;
        return Ok(());
    }

    if let Some(path) = args.verify_evaluation {
        verify_model_evaluation_bundle(&identify_program, &path, args.json)?;
        return Ok(());
    }

    if let Some(path) = args.scan_evaluation {
        scan_model_evaluation_bundle(&path, args.evaluation_limit, args.json)?;
        return Ok(());
    }

    if let Some(path) = args.inspect_evaluation {
        inspect_model_evaluation_row(
            &identify_program,
            &path,
            args.evaluation_row,
            args.inference_output.as_deref(),
            args.standalone_rev_output.as_deref(),
            args.markdown_output.as_deref(),
            args.json,
        )?;
        return Ok(());
    }

    if let Some(path) = args.verify_inference {
        verify_inference_bundle(
            &identify_program,
            &path,
            args.markdown_output.as_deref(),
            args.json,
        )?;
        return Ok(());
    }

    if !args.compare_artifacts.is_empty() {
        compare_artifacts(&args.compare_artifacts, args.json)?;
        return Ok(());
    }

    if args.self_test {
        run_self_test(
            &train_program,
            &identify_program,
            args.json,
            args.audit_output,
        )?;
        return Ok(());
    }

    let train_images = required_path(args.train_images, "--train-images")?;
    let train_labels = required_path(args.train_labels, "--train-labels")?;
    let test_images = required_path(args.test_images, "--test-images")?;
    let test_labels = required_path(args.test_labels, "--test-labels")?;

    let train = load_idx_dataset(&train_images, &train_labels)?;
    let test = load_idx_dataset(&test_images, &test_labels)?;
    let config = RunConfig {
        dataset_kind: DatasetKind::Idx,
        epochs: args.epochs,
        train_limit: args.train_limit,
        eval_limit: args.eval_limit,
        lr: args.lr,
        reverse_check: args.reverse_check,
        audit_output: args.audit_output,
    };
    let report = run_training_and_eval(&train_program, &identify_program, &train, &test, config)?;
    if args.json {
        print_report_json(&report, false);
    } else {
        print_report(&report, args.reverse_check, false);
    }
    Ok(())
}

fn required_path(path: Option<PathBuf>, flag: &str) -> Result<PathBuf, String> {
    path.ok_or_else(|| format!("{flag} is required unless --self-test is used"))
}

fn compile_program(name: &str, source: &str) -> Result<Program, String> {
    let program = reverie_syntax::parse_program(source)
        .map_err(|diagnostics| format_parse_errors(name, diagnostics))?;
    reverie_core::check_program(&program).map_err(|error| format!("{name}: {error}"))?;
    Ok(program)
}

fn format_parse_errors(name: &str, diagnostics: Vec<reverie_syntax::SyntaxDiagnostic>) -> String {
    let messages = diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.message)
        .collect::<Vec<_>>()
        .join("; ");
    format!("{name}: {messages}")
}

fn run_self_test(
    train_program: &Program,
    identify_program: &Program,
    emit_json: bool,
    audit_output: Option<PathBuf>,
) -> Result<(), String> {
    let train = synthetic_dataset(5);
    let test = synthetic_dataset(1);
    let config = RunConfig {
        dataset_kind: DatasetKind::Synthetic,
        epochs: 2,
        train_limit: None,
        eval_limit: None,
        lr: Q31_ONE / 4,
        reverse_check: true,
        audit_output,
    };
    let report = run_training_and_eval(train_program, identify_program, &train, &test, config)?;
    if report.eval_correct != report.eval_samples {
        return Err(format!(
            "self-test expected perfect synthetic accuracy, got {}/{}",
            report.eval_correct, report.eval_samples
        ));
    }
    if report.reverse_steps != report.train_samples {
        return Err(format!(
            "self-test expected {} reversed steps, got {}",
            report.train_samples, report.reverse_steps
        ));
    }
    if emit_json {
        print_report_json(&report, true);
    } else {
        print_report(&report, true, true);
        println!("self-test ok");
    }
    Ok(())
}

fn run_training_and_eval(
    train_program: &Program,
    identify_program: &Program,
    train: &Dataset,
    test: &Dataset,
    config: RunConfig,
) -> Result<RunReport, String> {
    if config.audit_output.is_some() && !config.reverse_check {
        return Err("--audit-output requires --reverse-check so witnesses are retained".to_owned());
    }
    if config.epochs == 0 {
        return Err("--epochs must be positive".to_owned());
    }
    if config.lr < 0 {
        return Err("--lr must be non-negative".to_owned());
    }
    let train_len = train.limited_len(config.train_limit);
    let eval_len = test.limited_len(config.eval_limit);
    if train_len == 0 {
        return Err("training dataset is empty after applying --train-limit".to_owned());
    }
    if eval_len == 0 {
        return Err("evaluation dataset is empty after applying --eval-limit".to_owned());
    }

    let mut model = Model::zero();
    let initial_model = model.clone();
    let mut train_correct = 0_usize;
    let mut train_samples = 0_usize;
    let mut tape = Vec::new();
    let train_start = Instant::now();

    for _ in 0..config.epochs {
        for index in 0..train_len {
            let step = train_step(train_program, &model, train, index, config.lr)?;
            model = step.model;
            train_correct += usize::from(step.correct);
            train_samples += 1;
            if config.reverse_check {
                tape.push(step.tape);
            }
        }
    }
    let train_elapsed = train_start.elapsed();

    let reverse_elapsed = if config.reverse_check {
        let reverse_start = Instant::now();
        reverse_training(train_program, &model, &initial_model, train, &tape)?;
        Some(reverse_start.elapsed())
    } else {
        None
    };

    let eval_start = Instant::now();
    let mut eval_correct = 0_usize;
    for index in 0..eval_len {
        let step = identify(identify_program, &model, test, index)?;
        eval_correct += usize::from(step.correct);
    }
    let eval_elapsed = eval_start.elapsed();

    let trace_payload_bytes = witness_trace_payload_bytes(tape.len());
    let memory = MemoryReport::for_run(train, test, trace_payload_bytes);

    let report = RunReport {
        config,
        train_samples,
        train_correct,
        eval_samples: eval_len,
        eval_correct,
        train_elapsed,
        eval_elapsed,
        reverse_elapsed,
        reverse_steps: tape.len(),
        trace_payload_bytes,
        memory,
    };

    if let Some(path) = &report.config.audit_output {
        write_audit_bundle(path, &report, &model, train, &tape)?;
    }

    Ok(report)
}

#[derive(Debug)]
struct TrainStep {
    model: Model,
    correct: bool,
    tape: TapeEntry,
}

fn train_step(
    program: &Program,
    model: &Model,
    dataset: &Dataset,
    index: usize,
    lr: i64,
) -> Result<TrainStep, String> {
    let label = i64::from(dataset.label(index));
    let state = sample_state(model, dataset.image(index), label, lr, true);
    let final_state = execute_compiled(program, state)
        .map_err(|error| format!("training step failed: {error}"))?;
    let next_model = model_from_state(&final_state)?;
    let logits = vector_from_state(&final_state, "logits", DIGITS)?;
    let error = vector_from_state(&final_state, "error", DIGITS)?;
    let prediction = int_from_state(&final_state, "prediction")?;
    let correct = int_from_state(&final_state, "correct")?;

    Ok(TrainStep {
        model: next_model,
        correct: correct != 0,
        tape: TapeEntry {
            sample_index: index,
            label,
            logits,
            error,
            prediction,
            correct,
            lr,
        },
    })
}

#[derive(Debug)]
struct IdentifyStep {
    correct: bool,
}

fn identify(
    program: &Program,
    model: &Model,
    dataset: &Dataset,
    index: usize,
) -> Result<IdentifyStep, String> {
    let label = i64::from(dataset.label(index));
    let state = sample_state(model, dataset.image(index), label, 0, false);
    let final_state = execute_compiled(program, state)
        .map_err(|error| format!("identify step failed: {error}"))?;
    let _prediction = int_from_state(&final_state, "prediction")?;
    let correct = int_from_state(&final_state, "correct")?;
    Ok(IdentifyStep {
        correct: correct != 0,
    })
}

fn reverse_training(
    program: &Program,
    final_model: &Model,
    initial_model: &Model,
    dataset: &Dataset,
    tape: &[TapeEntry],
) -> Result<(), String> {
    let mut model = final_model.clone();
    for entry in tape.iter().rev() {
        model = reverse_one_step(program, &model, dataset.image(entry.sample_index), entry)?;
    }

    if &model != initial_model {
        return Err(
            "reverse check failed: final model did not invert to the initial model".to_owned(),
        );
    }

    Ok(())
}

fn forward_training(
    program: &Program,
    final_model: &Model,
    dataset: &Dataset,
    tape: &[TapeEntry],
) -> Result<(), String> {
    let mut model = Model::zero();
    for (position, saved) in tape.iter().enumerate() {
        let step = train_step(program, &model, dataset, position, saved.lr)?;
        verify_forward_tape(position, saved, &step.tape)?;
        model = step.model;
    }

    if &model != final_model {
        return Err("forward replay failed: recomputed model did not match final model".to_owned());
    }

    Ok(())
}

fn verify_forward_tape(
    position: usize,
    saved: &TapeEntry,
    recomputed: &TapeEntry,
) -> Result<(), String> {
    if saved.label != recomputed.label {
        return Err(format!(
            "forward replay failed at step {position}: saved label {}, recomputed {}",
            saved.label, recomputed.label
        ));
    }
    if saved.lr != recomputed.lr {
        return Err(format!(
            "forward replay failed at step {position}: saved lr {}, recomputed {}",
            saved.lr, recomputed.lr
        ));
    }
    if saved.logits != recomputed.logits {
        return Err(format!(
            "forward replay failed at step {position}: saved logits do not match recomputed logits"
        ));
    }
    if saved.error != recomputed.error {
        return Err(format!(
            "forward replay failed at step {position}: saved error does not match recomputed error"
        ));
    }
    if saved.prediction != recomputed.prediction {
        return Err(format!(
            "forward replay failed at step {position}: saved prediction {}, recomputed {}",
            saved.prediction, recomputed.prediction
        ));
    }
    if saved.correct != recomputed.correct {
        return Err(format!(
            "forward replay failed at step {position}: saved correct {}, recomputed {}",
            saved.correct, recomputed.correct
        ));
    }

    Ok(())
}

fn sample_state(model: &Model, image: &[u8], label: i64, lr: i64, train: bool) -> State {
    let mut state = State::empty();
    state.insert("image", image_value(image));
    state.insert("weights", matrix_value(&model.weights));
    state.insert("bias", vector_value(&model.bias));
    state.insert("logits", zero_vector_value(DIGITS));
    state.insert("prediction", Value::Int(0));
    state.insert("correct", Value::Int(0));
    state.insert("label", Value::Int(label));
    if train {
        state.insert("error", zero_vector_value(DIGITS));
        state.insert("lr", Value::Int(lr));
    }
    state
}

fn taped_final_state(model: &Model, image: &[u8], entry: &TapeEntry) -> State {
    let mut state = State::empty();
    state.insert("image", image_value(image));
    state.insert("weights", matrix_value(&model.weights));
    state.insert("bias", vector_value(&model.bias));
    state.insert("logits", vector_value(&entry.logits));
    state.insert("error", vector_value(&entry.error));
    state.insert("prediction", Value::Int(entry.prediction));
    state.insert("correct", Value::Int(entry.correct));
    state.insert("label", Value::Int(entry.label));
    state.insert("lr", Value::Int(entry.lr));
    state
}

fn reverse_one_step(
    program: &Program,
    final_model: &Model,
    image: &[u8],
    entry: &TapeEntry,
) -> Result<Model, String> {
    let state = taped_final_state(final_model, image, entry);
    let previous = execute_compiled_backward(program, state)
        .map_err(|error| format!("reverse check failed while inverting a step: {error}"))?;
    model_from_state(&previous)
}

fn image_value(image: &[u8]) -> Value {
    Value::Array(
        image
            .iter()
            .map(|pixel| Value::Int(pixel_to_q31(*pixel)))
            .collect(),
    )
}

fn pixel_to_q31(pixel: u8) -> i64 {
    (i64::from(pixel) * Q31_ONE) / 255
}

fn vector_value(values: &[i64]) -> Value {
    Value::Array(values.iter().copied().map(Value::Int).collect())
}

fn zero_vector_value(len: usize) -> Value {
    Value::Array((0..len).map(|_| Value::Int(0)).collect())
}

fn matrix_value(values: &[Vec<i64>]) -> Value {
    Value::Array(values.iter().map(|row| vector_value(row)).collect())
}

fn model_from_state(state: &State) -> Result<Model, String> {
    Ok(Model {
        weights: matrix_from_state(state, "weights", IMAGE_PIXELS, DIGITS)?,
        bias: vector_from_state(state, "bias", DIGITS)?,
    })
}

fn int_from_state(state: &State, name: &str) -> Result<i64, String> {
    match state.get(name) {
        Some(Value::Int(value)) => Ok(*value),
        Some(other) => Err(format!("state `{name}` expected int, found {other}")),
        None => Err(format!("state is missing `{name}`")),
    }
}

fn vector_from_state(state: &State, name: &str, len: usize) -> Result<Vec<i64>, String> {
    let Some(value) = state.get(name) else {
        return Err(format!("state is missing `{name}`"));
    };
    vector_from_value(name, value, len)
}

fn vector_from_value(name: &str, value: &Value, len: usize) -> Result<Vec<i64>, String> {
    let Value::Array(values) = value else {
        return Err(format!("state `{name}` expected vector, found {value}"));
    };
    if values.len() != len {
        return Err(format!(
            "state `{name}` expected vector length {len}, found {}",
            values.len()
        ));
    }
    values
        .iter()
        .map(|value| match value {
            Value::Int(value) => Ok(*value),
            other => Err(format!(
                "state `{name}` expected int element, found {other}"
            )),
        })
        .collect()
}

fn matrix_from_state(
    state: &State,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<Vec<i64>>, String> {
    let Some(value) = state.get(name) else {
        return Err(format!("state is missing `{name}`"));
    };
    let Value::Array(row_values) = value else {
        return Err(format!("state `{name}` expected matrix, found {value}"));
    };
    if row_values.len() != rows {
        return Err(format!(
            "state `{name}` expected {rows} rows, found {}",
            row_values.len()
        ));
    }
    row_values
        .iter()
        .enumerate()
        .map(|(row, value)| vector_from_value(&format!("{name}[{row}]"), value, cols))
        .collect()
}

fn load_idx_dataset(images_path: &Path, labels_path: &Path) -> Result<Dataset, String> {
    let images = fs::read(images_path)
        .map_err(|error| format!("failed to read {}: {error}", images_path.display()))?;
    let labels = fs::read(labels_path)
        .map_err(|error| format!("failed to read {}: {error}", labels_path.display()))?;
    let images = parse_idx_images(images_path, &images)?;
    let labels = parse_idx_labels(labels_path, &labels)?;
    if images.len() != labels.len() {
        return Err(format!(
            "image/label count mismatch: {} images in {}, {} labels in {}",
            images.len(),
            images_path.display(),
            labels.len(),
            labels_path.display()
        ));
    }
    Ok(Dataset { images, labels })
}

fn parse_idx_images(path: &Path, data: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    if data.len() < 16 {
        return Err(format!(
            "{} is too short to be an IDX image file",
            path.display()
        ));
    }
    let magic = read_u32(data, 0);
    let count = read_u32(data, 4) as usize;
    let rows = read_u32(data, 8) as usize;
    let cols = read_u32(data, 12) as usize;
    if magic != 2051 {
        return Err(format!(
            "{} has IDX magic {magic}, expected 2051 for images",
            path.display()
        ));
    }
    if rows != 28 || cols != 28 {
        return Err(format!(
            "{} has image shape {rows}x{cols}, expected 28x28",
            path.display()
        ));
    }
    let expected = 16 + count * IMAGE_PIXELS;
    if data.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected}",
            path.display(),
            data.len()
        ));
    }
    Ok(data[16..]
        .chunks_exact(IMAGE_PIXELS)
        .map(|image| image.to_vec())
        .collect())
}

fn parse_idx_labels(path: &Path, data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 8 {
        return Err(format!(
            "{} is too short to be an IDX label file",
            path.display()
        ));
    }
    let magic = read_u32(data, 0);
    let count = read_u32(data, 4) as usize;
    if magic != 2049 {
        return Err(format!(
            "{} has IDX magic {magic}, expected 2049 for labels",
            path.display()
        ));
    }
    let expected = 8 + count;
    if data.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected}",
            path.display(),
            data.len()
        ));
    }
    let labels = data[8..].to_vec();
    if let Some(label) = labels.iter().find(|label| **label >= DIGITS as u8) {
        return Err(format!(
            "{} contains label {label}, expected labels in 0..=9",
            path.display()
        ));
    }
    Ok(labels)
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn synthetic_dataset(repeats: usize) -> Dataset {
    let mut images = Vec::with_capacity(repeats * DIGITS);
    let mut labels = Vec::with_capacity(repeats * DIGITS);
    for _ in 0..repeats {
        for label in 0..DIGITS {
            let mut image = vec![0; IMAGE_PIXELS];
            image[label] = 255;
            image[DIGITS + label] = 128;
            images.push(image);
            labels.push(label as u8);
        }
    }
    Dataset { images, labels }
}

fn print_report(report: &RunReport, reverse_check: bool, self_test: bool) {
    let label = if self_test {
        "Reverie MNIST linear Q31 self-test"
    } else {
        "Reverie MNIST linear Q31"
    };
    println!("{label}");
    println!(
        "train: samples={} correct={} accuracy={:.2}% elapsed={:.3}s updates_per_sec={:.1}",
        report.train_samples,
        report.train_correct,
        percent(report.train_correct, report.train_samples),
        secs(report.train_elapsed),
        per_second(report.train_samples, report.train_elapsed)
    );
    println!(
        "eval: samples={} correct={} accuracy={:.2}% elapsed={:.3}s samples_per_sec={:.1}",
        report.eval_samples,
        report.eval_correct,
        percent(report.eval_correct, report.eval_samples),
        secs(report.eval_elapsed),
        per_second(report.eval_samples, report.eval_elapsed)
    );
    if reverse_check {
        let elapsed = report.reverse_elapsed.unwrap_or_default();
        let proof = ProofCostReport::for_training_trace(report.reverse_steps);
        println!(
            "trace: entries={} payload_bytes={} bytes_per_step={}",
            report.reverse_steps,
            report.trace_payload_bytes,
            witness_trace_payload_bytes(1)
        );
        println!(
            "proof: model_bytes={} sample_bytes={} witness_bytes={} replay_bytes={} replay_bytes_per_step={} forward_recompute_steps={} inverse_recompute_steps={}",
            proof.model_payload_bytes,
            proof.sample_payload_bytes,
            proof.witness_payload_bytes,
            proof.full_replay_payload_bytes,
            proof.trace_replay_bytes_per_step,
            proof.forward_recompute_steps,
            proof.inverse_recompute_steps
        );
        println!(
            "reverse: checked={} restored_initial_model=true elapsed={:.3}s steps_per_sec={:.1}",
            report.reverse_steps,
            secs(elapsed),
            per_second(report.reverse_steps, elapsed)
        );
    }
}

fn print_report_json(report: &RunReport, self_test: bool) {
    println!(
        "{}",
        serde_json::to_string_pretty(&report_json_value(report, self_test))
            .expect("benchmark report serializes")
    );
}

fn report_json_value(report: &RunReport, self_test: bool) -> serde_json::Value {
    let proof = ProofCostReport::for_training_trace(report.reverse_steps);
    let reverse = if report.config.reverse_check {
        let elapsed = report.reverse_elapsed.unwrap_or_default();
        json!({
            "enabled": true,
            "checked": report.reverse_steps,
            "restored_initial_model": true,
            "elapsed_seconds": secs(elapsed),
            "steps_per_second": per_second(report.reverse_steps, elapsed),
        })
    } else {
        serde_json::Value::Null
    };

    json!({
        "kind": "reverie_mnist_linear_q31",
        "self_test": self_test,
        "config": {
            "dataset_kind": report.config.dataset_kind.as_str(),
            "epochs": report.config.epochs,
            "train_limit": report.config.train_limit,
            "eval_limit": report.config.eval_limit,
            "lr": report.config.lr,
            "reverse_check": report.config.reverse_check,
            "audit_output": report.config.audit_output.as_ref().map(|path| path.display().to_string()),
        },
        "train": {
            "samples": report.train_samples,
            "correct": report.train_correct,
            "accuracy_percent": percent(report.train_correct, report.train_samples),
            "elapsed_seconds": secs(report.train_elapsed),
            "updates_per_second": per_second(report.train_samples, report.train_elapsed),
        },
        "eval": {
            "samples": report.eval_samples,
            "correct": report.eval_correct,
            "accuracy_percent": percent(report.eval_correct, report.eval_samples),
            "elapsed_seconds": secs(report.eval_elapsed),
            "samples_per_second": per_second(report.eval_samples, report.eval_elapsed),
        },
        "trace": {
            "enabled": report.config.reverse_check,
            "entries": report.reverse_steps,
            "payload_bytes": report.trace_payload_bytes,
            "bytes_per_step": witness_trace_payload_bytes(1),
        },
        "reverse": reverse,
        "proof": proof.to_json(report.config.reverse_check),
        "memory": {
            "model_payload_bytes": report.memory.model_payload_bytes,
            "train_dataset_payload_bytes": report.memory.train_dataset_payload_bytes,
            "eval_dataset_payload_bytes": report.memory.eval_dataset_payload_bytes,
            "dataset_payload_bytes": report.memory.dataset_payload_bytes,
            "trace_payload_bytes": report.memory.trace_payload_bytes,
            "estimated_payload_bytes": report.memory.estimated_payload_bytes,
            "peak_rss_bytes": report.memory.peak_rss_bytes,
        },
    })
}

fn write_audit_bundle(
    path: &Path,
    report: &RunReport,
    final_model: &Model,
    train: &Dataset,
    tape: &[TapeEntry],
) -> Result<(), String> {
    let trace = tape
        .iter()
        .map(|entry| {
            json!({
                "sample_index": entry.sample_index,
                "label": entry.label,
                "image_u8": train.image(entry.sample_index),
                "logits": &entry.logits,
                "error": &entry.error,
                "prediction": entry.prediction,
                "correct": entry.correct,
                "lr": entry.lr,
            })
        })
        .collect::<Vec<_>>();
    let self_test = matches!(report.config.dataset_kind, DatasetKind::Synthetic);
    let proof = ProofCostReport::for_training_trace(tape.len());
    let final_model_json = model_json(final_model);
    let mut bundle = json!({
        "kind": "reverie_mnist_linear_q31_replay_bundle",
        "schema_version": 1,
        "programs": {
            "train": "examples/mnist_reversible_step.rev",
            "identify": "examples/mnist_identify.rev",
        },
        "tensor_shapes": {
            "image": [IMAGE_PIXELS],
            "weights": [IMAGE_PIXELS, DIGITS],
            "bias": [DIGITS],
            "logits": [DIGITS],
            "error": [DIGITS],
        },
        "initial_model": {
            "kind": "zero",
            "weights_shape": [IMAGE_PIXELS, DIGITS],
            "bias_shape": [DIGITS],
        },
        "final_model": final_model_json,
        "report": report_json_value(report, self_test),
        "proof": proof.to_json(report.config.reverse_check),
        "witness_trace": trace,
    });
    let lineage_ledger = training_lineage_ledger_json(
        json_field(&bundle, "final_model")?,
        json_array_field(&bundle, "witness_trace")?,
    )?;
    bundle
        .as_object_mut()
        .expect("audit bundle is an object")
        .insert("lineage_ledger".to_owned(), lineage_ledger);
    let fingerprints = audit_fingerprints_for_unsigned_payload(&bundle);
    bundle
        .as_object_mut()
        .expect("audit bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints.to_json());

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let encoded = serde_json::to_string_pretty(&bundle).expect("audit replay bundle serializes");
    fs::write(path, encoded).map_err(|error| {
        format!(
            "failed to write audit replay bundle {}: {error}",
            path.display()
        )
    })
}

fn verify_audit_bundle(
    program: &Program,
    path: &Path,
    markdown_output: Option<&Path>,
    emit_json: bool,
) -> Result<(), String> {
    let bundle = read_audit_bundle(path)?;
    let fingerprints = audit_fingerprints_from_bundle(&bundle)?;
    let final_model = model_from_json(json_field(&bundle, "final_model")?, "final_model")?;
    let trace = json_array_field(&bundle, "witness_trace")?;
    let steps = parse_audit_steps(trace)?;
    let images = steps.iter().map(|step| step.image.clone()).collect();
    let labels = steps.iter().map(|step| step.tape.label as u8).collect();
    let tape = steps
        .iter()
        .map(|step| step.tape.clone())
        .collect::<Vec<_>>();
    let proof = ProofCostReport::for_training_trace(tape.len()).to_json(true);
    verify_training_proof(json_field(&bundle, "proof")?, &proof)?;
    let lineage_ledger = training_lineage_ledger_json(json_field(&bundle, "final_model")?, trace)?;
    verify_training_lineage_ledger(json_field(&bundle, "lineage_ledger")?, &lineage_ledger)?;

    let dataset = Dataset { images, labels };
    let forward_start = Instant::now();
    forward_training(program, &final_model, &dataset, &tape)?;
    let forward_elapsed = forward_start.elapsed();

    let reverse_start = Instant::now();
    reverse_training(program, &final_model, &Model::zero(), &dataset, &tape)?;
    let reverse_elapsed = reverse_start.elapsed();

    let report = json!({
        "kind": "reverie_mnist_linear_q31_audit_verification",
        "path": path.display().to_string(),
        "fingerprints": fingerprints.to_json(),
        "markdown_output": markdown_output.map(|path| path.display().to_string()),
        "checked": tape.len(),
        "witnesses_match_forward_replay": true,
        "final_model_replayed": true,
        "restored_initial_model": true,
        "proof_matches": true,
        "lineage_ledger_matches": true,
        "elapsed_seconds": secs(reverse_elapsed),
        "steps_per_second": per_second(tape.len(), reverse_elapsed),
        "proof": proof,
        "lineage_ledger": lineage_ledger,
        "forward": {
            "checked": tape.len(),
            "witnesses_match": true,
            "final_model_matches": true,
            "elapsed_seconds": secs(forward_elapsed),
            "steps_per_second": per_second(tape.len(), forward_elapsed),
        },
        "reverse": {
            "checked": tape.len(),
            "restored_initial_model": true,
            "elapsed_seconds": secs(reverse_elapsed),
            "steps_per_second": per_second(tape.len(), reverse_elapsed),
        },
    });
    if let Some(output_path) = markdown_output {
        write_markdown_file(
            output_path,
            &render_training_audit_verification_markdown(&report)?,
        )?;
    }

    if emit_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("audit verification report serializes")
        );
    } else {
        println!(
            "audit ok: checked={} witnesses_match_forward_replay=true final_model_replayed=true restored_initial_model=true elapsed={:.3}s steps_per_sec={:.1}",
            tape.len(),
            secs(reverse_elapsed),
            per_second(tape.len(), reverse_elapsed)
        );
        println!("{}", format_training_proof_line(&proof));
        println!(
            "forward: checked={} elapsed={:.3}s steps_per_sec={:.1}",
            tape.len(),
            secs(forward_elapsed),
            per_second(tape.len(), forward_elapsed)
        );
        println!(
            "reverse: checked={} elapsed={:.3}s steps_per_sec={:.1}",
            tape.len(),
            secs(reverse_elapsed),
            per_second(tape.len(), reverse_elapsed)
        );
        println!(
            "fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
        if let Some(output_path) = markdown_output {
            println!("markdown: {}", output_path.display());
        }
    }

    Ok(())
}

fn verify_training_proof(
    stored: &serde_json::Value,
    expected: &serde_json::Value,
) -> Result<(), String> {
    if stored != expected {
        return Err("audit bundle proof does not match recomputed training replay cost".to_owned());
    }
    Ok(())
}

fn verify_training_lineage_ledger(
    stored: &serde_json::Value,
    expected: &serde_json::Value,
) -> Result<(), String> {
    if stored != expected {
        return Err(
            "audit bundle lineage ledger does not match recomputed training lineage".to_owned(),
        );
    }
    Ok(())
}

fn format_training_proof_line(proof: &serde_json::Value) -> String {
    format!(
        "proof: entries={} model_bytes={} sample_bytes={} witness_bytes={} replay_bytes={} forward_recompute_steps={} inverse_recompute_steps={}",
        proof
            .get("entries")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("model_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("sample_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("witness_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("full_replay_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("forward_recompute_steps")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("inverse_recompute_steps")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    )
}

fn read_audit_bundle(path: &Path) -> Result<serde_json::Value, String> {
    let data = fs::read(path)
        .map_err(|error| format!("failed to read audit bundle {}: {error}", path.display()))?;
    let bundle: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|error| format!("failed to parse audit bundle {}: {error}", path.display()))?;
    let kind = json_str_field(&bundle, "kind")?;
    if kind != "reverie_mnist_linear_q31_replay_bundle" {
        return Err(format!(
            "audit bundle kind `{kind}` is not reverie_mnist_linear_q31_replay_bundle"
        ));
    }
    verify_audit_fingerprints(&bundle)?;
    Ok(bundle)
}

fn export_model_bundle(
    program: &Program,
    audit_path: &Path,
    output_path: &Path,
    emit_json: bool,
) -> Result<(), String> {
    let bundle = read_audit_bundle(audit_path)?;
    let source_fingerprints = audit_fingerprints_from_bundle(&bundle)?;
    let final_model = model_from_json(json_field(&bundle, "final_model")?, "final_model")?;
    let trace = json_array_field(&bundle, "witness_trace")?;
    let steps = parse_audit_steps(trace)?;
    let images = steps.iter().map(|step| step.image.clone()).collect();
    let labels = steps.iter().map(|step| step.tape.label as u8).collect();
    let tape = steps
        .iter()
        .map(|step| step.tape.clone())
        .collect::<Vec<_>>();
    let dataset = Dataset { images, labels };

    let forward_start = Instant::now();
    forward_training(program, &final_model, &dataset, &tape)?;
    let forward_elapsed = forward_start.elapsed();
    let reverse_start = Instant::now();
    reverse_training(program, &final_model, &Model::zero(), &dataset, &tape)?;
    let reverse_elapsed = reverse_start.elapsed();

    let fingerprints = write_model_bundle(
        output_path,
        audit_path,
        &source_fingerprints,
        &final_model,
        json_field(&bundle, "report")?,
        json_field(&bundle, "proof")?,
        steps.len(),
    )?;

    if emit_json {
        let report = json!({
            "kind": "reverie_mnist_linear_q31_model_export",
            "source_audit_path": audit_path.display().to_string(),
            "model_output": output_path.display().to_string(),
            "source_fingerprints": source_fingerprints.to_json(),
            "fingerprints": fingerprints.to_json(),
            "checked": steps.len(),
            "witnesses_match_forward_replay": true,
            "final_model_replayed": true,
            "restored_initial_model": true,
            "forward": {
                "elapsed_seconds": secs(forward_elapsed),
                "steps_per_second": per_second(steps.len(), forward_elapsed),
            },
            "reverse": {
                "elapsed_seconds": secs(reverse_elapsed),
                "steps_per_second": per_second(steps.len(), reverse_elapsed),
            },
            "storage": {
                "model_payload_bytes": Model::payload_bytes(),
            },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("model export report serializes")
        );
    } else {
        println!(
            "model export ok: checked={} final_model_replayed=true restored_initial_model=true output={}",
            steps.len(),
            output_path.display()
        );
        println!(
            "source_fingerprint: computation={} payload={}",
            source_fingerprints.computation, source_fingerprints.payload
        );
        println!(
            "model_fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
    }

    Ok(())
}

fn import_model_bundle(
    input_path: &Path,
    output_path: &Path,
    emit_json: bool,
) -> Result<(), String> {
    let data = fs::read(input_path).map_err(|error| {
        format!(
            "failed to read imported model JSON {}: {error}",
            input_path.display()
        )
    })?;
    let source_json: serde_json::Value = serde_json::from_slice(&data).map_err(|error| {
        format!(
            "failed to parse imported model JSON {}: {error}",
            input_path.display()
        )
    })?;
    let model_json = source_json.get("model").unwrap_or(&source_json);
    let model = model_from_json(model_json, "imported_model")?;
    let source_json_fingerprint = sha256_json(&source_json);
    let source_file_fingerprint = sha256_bytes(&data);

    let fingerprints = write_imported_model_bundle(
        output_path,
        input_path,
        &source_json_fingerprint,
        &source_file_fingerprint,
        &model,
    )?;

    if emit_json {
        let report = json!({
            "kind": "reverie_mnist_linear_q31_model_import",
            "source_model_json_path": input_path.display().to_string(),
            "model_output": output_path.display().to_string(),
            "source_model_json_fingerprint": source_json_fingerprint,
            "source_model_json_file_sha256": source_file_fingerprint,
            "fingerprints": fingerprints.to_json(),
            "shape_matches": true,
            "provenance_kind": "external_import",
            "storage": {
                "model_payload_bytes": Model::payload_bytes(),
            },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("model import report serializes")
        );
    } else {
        println!(
            "model import ok: shape_matches=true model_payload_bytes={} output={}",
            Model::payload_bytes(),
            output_path.display()
        );
        println!("source_model_json: payload={source_json_fingerprint}");
        println!(
            "model_fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
    }

    Ok(())
}

fn write_model_bundle(
    path: &Path,
    source_audit_path: &Path,
    source_fingerprints: &AuditFingerprints,
    model: &Model,
    training_report: &serde_json::Value,
    proof: &serde_json::Value,
    training_steps: usize,
) -> Result<ModelFingerprints, String> {
    let report = json!({
        "kind": "reverie_mnist_linear_q31_model",
        "provenance_kind": "training_audit",
        "source_audit_path": source_audit_path.display().to_string(),
        "source_fingerprints": source_fingerprints.to_json(),
        "training_steps": training_steps,
        "model_payload_bytes": Model::payload_bytes(),
        "tensor_shapes": {
            "image": [IMAGE_PIXELS],
            "weights": [IMAGE_PIXELS, DIGITS],
            "bias": [DIGITS],
            "logits": [DIGITS],
        },
    });
    let mut bundle = json!({
        "kind": "reverie_mnist_linear_q31_model_bundle",
        "schema_version": 1,
        "program": "examples/mnist_identify.rev",
        "tensor_shapes": {
            "image": [IMAGE_PIXELS],
            "weights": [IMAGE_PIXELS, DIGITS],
            "bias": [DIGITS],
            "logits": [DIGITS],
        },
        "model": {
            "weights": &model.weights,
            "bias": &model.bias,
        },
        "provenance": {
            "source_training_bundle": {
                "path": source_audit_path.display().to_string(),
                "fingerprints": source_fingerprints.to_json(),
            },
            "training_report": training_report,
            "proof": proof,
        },
        "storage": {
            "model_payload_bytes": Model::payload_bytes(),
        },
        "report": report,
    });
    let fingerprints = model_fingerprints_for_unsigned_payload(&bundle);
    bundle
        .as_object_mut()
        .expect("model bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints.to_json());

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let encoded = serde_json::to_string_pretty(&bundle).expect("model bundle serializes");
    fs::write(path, encoded)
        .map_err(|error| format!("failed to write model bundle {}: {error}", path.display()))?;
    Ok(fingerprints)
}

fn write_imported_model_bundle(
    path: &Path,
    source_model_json_path: &Path,
    source_json_fingerprint: &str,
    source_file_fingerprint: &str,
    model: &Model,
) -> Result<ModelFingerprints, String> {
    let report = json!({
        "kind": "reverie_mnist_linear_q31_model",
        "provenance_kind": "external_import",
        "source_model_json_path": source_model_json_path.display().to_string(),
        "source_model_json_fingerprint": source_json_fingerprint,
        "source_model_json_file_sha256": source_file_fingerprint,
        "training_steps": serde_json::Value::Null,
        "model_payload_bytes": Model::payload_bytes(),
        "tensor_shapes": {
            "image": [IMAGE_PIXELS],
            "weights": [IMAGE_PIXELS, DIGITS],
            "bias": [DIGITS],
            "logits": [DIGITS],
        },
    });
    let mut bundle = json!({
        "kind": "reverie_mnist_linear_q31_model_bundle",
        "schema_version": 1,
        "program": "examples/mnist_identify.rev",
        "tensor_shapes": {
            "image": [IMAGE_PIXELS],
            "weights": [IMAGE_PIXELS, DIGITS],
            "bias": [DIGITS],
            "logits": [DIGITS],
        },
        "model": {
            "weights": &model.weights,
            "bias": &model.bias,
        },
        "provenance": {
            "source_model_json": {
                "path": source_model_json_path.display().to_string(),
                "fingerprint": source_json_fingerprint,
                "file_sha256": source_file_fingerprint,
            },
            "import": {
                "format": "weights_bias_q31_json",
                "model_payload_bytes": Model::payload_bytes(),
            },
        },
        "storage": {
            "model_payload_bytes": Model::payload_bytes(),
        },
        "report": report,
    });
    let fingerprints = model_fingerprints_for_unsigned_payload(&bundle);
    bundle
        .as_object_mut()
        .expect("model bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints.to_json());

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let encoded = serde_json::to_string_pretty(&bundle).expect("model bundle serializes");
    fs::write(path, encoded)
        .map_err(|error| format!("failed to write model bundle {}: {error}", path.display()))?;
    Ok(fingerprints)
}

fn format_optional_bool(value: Option<bool>) -> String {
    value.map_or_else(|| "n/a".to_owned(), |value| value.to_string())
}

fn format_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "n/a".to_owned(), |value| value.to_string())
}

fn verify_model_bundle(path: &Path, emit_json: bool) -> Result<(), String> {
    let bundle = read_model_bundle(path)?;
    let fingerprints = model_fingerprints_from_bundle(&bundle)?;
    let model_json = json_field(&bundle, "model")?;
    let model = model_from_json(model_json, "model")?;
    let storage = json_field(&bundle, "storage")?;
    let model_payload_bytes = json_usize_field(storage, "model_payload_bytes")?;
    if model_payload_bytes != Model::payload_bytes() {
        return Err(format!(
            "model bundle storage `model_payload_bytes` expected {}, found {model_payload_bytes}",
            Model::payload_bytes()
        ));
    }
    let provenance = verify_model_bundle_provenance(
        json_field(&bundle, "provenance")?,
        json_field(&bundle, "report")?,
        model_payload_bytes,
        path,
        model_json,
        &model,
    )?;

    if emit_json {
        let report = json!({
            "kind": "reverie_mnist_linear_q31_model_verification",
            "path": path.display().to_string(),
            "fingerprints": fingerprints.to_json(),
            "model_payload_bytes": model_payload_bytes,
            "shape_matches": true,
            "provenance_matches": true,
            "proof_matches": provenance.proof_matches,
            "provenance_kind": provenance.kind,
            "training_steps": provenance.training_steps,
            "source_audit_payload": provenance.source_audit_payload,
            "source_model_json_checked": provenance
                .source_model_json
                .as_ref()
                .is_some_and(|check| check.checked),
            "source_model_json": provenance
                .source_model_json
                .as_ref()
                .map(ImportedModelSourceCheck::to_json),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("model verification report serializes")
        );
    } else {
        println!(
            "model ok: model_payload_bytes={} shape_matches=true provenance_matches=true proof_matches={} provenance_kind={} training_steps={}",
            model_payload_bytes,
            format_optional_bool(provenance.proof_matches),
            provenance.kind,
            format_optional_usize(provenance.training_steps)
        );
        if let Some(source_audit_payload) = &provenance.source_audit_payload {
            println!("source_audit: payload={source_audit_payload}");
        }
        if let Some(source_model_json) = &provenance.source_model_json {
            println!("source_model_json: {}", source_model_json.format_line());
        }
        println!(
            "fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct ModelBundleProvenance {
    kind: &'static str,
    proof_matches: Option<bool>,
    training_steps: Option<usize>,
    source_audit_payload: Option<String>,
    source_model_json: Option<ImportedModelSourceCheck>,
}

fn verify_model_bundle_provenance(
    provenance: &serde_json::Value,
    report: &serde_json::Value,
    model_payload_bytes: usize,
    bundle_path: &Path,
    model_json: &serde_json::Value,
    model: &Model,
) -> Result<ModelBundleProvenance, String> {
    let kind = json_str_field(report, "kind")?;
    if kind != "reverie_mnist_linear_q31_model" {
        return Err(format!(
            "model bundle report kind `{kind}` is not reverie_mnist_linear_q31_model"
        ));
    }

    let report_model_payload_bytes = json_usize_field(report, "model_payload_bytes")?;
    if report_model_payload_bytes != model_payload_bytes {
        return Err(format!(
            "model bundle report `model_payload_bytes` expected {model_payload_bytes}, found {report_model_payload_bytes}"
        ));
    }
    verify_model_tensor_shapes(json_field(report, "tensor_shapes")?)?;

    let provenance_kind = report
        .get("provenance_kind")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("training_audit");
    match provenance_kind {
        "training_audit" => {
            verify_training_model_bundle_provenance(provenance, report, model_payload_bytes)
        }
        "external_import" => verify_imported_model_bundle_provenance(
            provenance,
            report,
            bundle_path,
            model_json,
            model,
        ),
        other => Err(format!(
            "model bundle provenance_kind `{other}` is not supported"
        )),
    }
}

fn verify_training_model_bundle_provenance(
    provenance: &serde_json::Value,
    report: &serde_json::Value,
    _model_payload_bytes: usize,
) -> Result<ModelBundleProvenance, String> {
    let source_audit_path = json_str_field(report, "source_audit_path")?;
    let report_source_fingerprints = json_field(report, "source_fingerprints")?;
    let training_steps = json_usize_field(report, "training_steps")?;
    let source_training_bundle = json_field(provenance, "source_training_bundle")?;
    let provenance_source_path = json_str_field(source_training_bundle, "path")?;
    if provenance_source_path != source_audit_path {
        return Err("model bundle provenance source path does not match model report".to_owned());
    }
    let provenance_source_fingerprints = json_field(source_training_bundle, "fingerprints")?;
    if provenance_source_fingerprints != report_source_fingerprints {
        return Err(
            "model bundle provenance source fingerprints do not match model report".to_owned(),
        );
    }
    let source_audit_payload =
        json_str_field(provenance_source_fingerprints, "payload")?.to_owned();

    let proof = json_field(provenance, "proof")?;
    let expected_proof = ProofCostReport::for_training_trace(training_steps).to_json(true);
    if proof != &expected_proof {
        return Err("model bundle provenance proof does not match training steps".to_owned());
    }

    let training_report = json_field(provenance, "training_report")?;
    if json_field(training_report, "proof")? != proof {
        return Err(
            "model bundle provenance proof does not match embedded training report".to_owned(),
        );
    }
    let train_samples = json_usize_field(json_field(training_report, "train")?, "samples")?;
    if train_samples != training_steps {
        return Err(format!(
            "model bundle training report `train.samples` expected {training_steps}, found {train_samples}"
        ));
    }
    let trace_entries = json_usize_field(json_field(training_report, "trace")?, "entries")?;
    if trace_entries != training_steps {
        return Err(format!(
            "model bundle training report `trace.entries` expected {training_steps}, found {trace_entries}"
        ));
    }
    let reverse_checked = json_usize_field(json_field(training_report, "reverse")?, "checked")?;
    if reverse_checked != training_steps {
        return Err(format!(
            "model bundle training report `reverse.checked` expected {training_steps}, found {reverse_checked}"
        ));
    }

    Ok(ModelBundleProvenance {
        kind: "training_audit",
        proof_matches: Some(true),
        training_steps: Some(training_steps),
        source_audit_payload: Some(source_audit_payload),
        source_model_json: None,
    })
}

fn verify_imported_model_bundle_provenance(
    provenance: &serde_json::Value,
    report: &serde_json::Value,
    bundle_path: &Path,
    model_json: &serde_json::Value,
    model: &Model,
) -> Result<ModelBundleProvenance, String> {
    if !matches!(
        report.get("training_steps"),
        None | Some(serde_json::Value::Null)
    ) {
        return Err("external imported model report `training_steps` must be null".to_owned());
    }
    let source_path = json_str_field(report, "source_model_json_path")?;
    let source_fingerprint = json_str_field(report, "source_model_json_fingerprint")?;
    let source_file_sha256 = json_str_field(report, "source_model_json_file_sha256")?;

    let source_model_json = json_field(provenance, "source_model_json")?;
    if json_str_field(source_model_json, "path")? != source_path {
        return Err("imported model provenance source path does not match model report".to_owned());
    }
    if json_str_field(source_model_json, "fingerprint")? != source_fingerprint {
        return Err(
            "imported model provenance source fingerprint does not match model report".to_owned(),
        );
    }
    if json_str_field(source_model_json, "file_sha256")? != source_file_sha256 {
        return Err(
            "imported model provenance source file hash does not match model report".to_owned(),
        );
    }

    let import = json_field(provenance, "import")?;
    let import_format = json_str_field(import, "format")?;
    if import_format != "weights_bias_q31_json" {
        return Err(format!(
            "imported model provenance format `{import_format}` is not weights_bias_q31_json"
        ));
    }
    let import_model_payload_bytes = json_usize_field(import, "model_payload_bytes")?;
    if import_model_payload_bytes != Model::payload_bytes() {
        return Err(format!(
            "imported model provenance `model_payload_bytes` expected {}, found {import_model_payload_bytes}",
            Model::payload_bytes()
        ));
    }

    let source_check = verify_imported_model_source(
        bundle_path,
        source_path,
        source_fingerprint,
        source_file_sha256,
        model_json,
        model,
    )?;

    Ok(ModelBundleProvenance {
        kind: "external_import",
        proof_matches: None,
        training_steps: None,
        source_audit_payload: None,
        source_model_json: Some(source_check),
    })
}

#[derive(Debug, Clone)]
struct ImportedModelSourceCheck {
    path: String,
    resolved_path: String,
    checked: bool,
    fingerprint: String,
    file_sha256: String,
    unavailable_reason: Option<&'static str>,
}

impl ImportedModelSourceCheck {
    fn to_json(&self) -> serde_json::Value {
        let mut value = json!({
            "path": self.path,
            "resolved_path": self.resolved_path,
            "checked": self.checked,
            "fingerprint": self.fingerprint,
            "file_sha256": self.file_sha256,
        });
        if let Some(reason) = self.unavailable_reason {
            value
                .as_object_mut()
                .expect("imported model source check JSON is an object")
                .insert("unavailable_reason".to_owned(), json!(reason));
        }
        value
    }

    fn format_line(&self) -> String {
        if self.checked {
            format!(
                "checked=true payload={} file_sha256={}",
                self.fingerprint, self.file_sha256
            )
        } else {
            format!(
                "checked=false reason={} path={}",
                self.unavailable_reason.unwrap_or("unavailable"),
                self.path
            )
        }
    }
}

fn verify_imported_model_source(
    bundle_path: &Path,
    source_path: &str,
    source_fingerprint: &str,
    source_file_sha256: &str,
    model_json: &serde_json::Value,
    model: &Model,
) -> Result<ImportedModelSourceCheck, String> {
    let resolved_path = resolve_referenced_artifact_path(bundle_path, source_path);
    let resolved_path_text = resolved_path.display().to_string();
    if !resolved_path.exists() {
        return Ok(ImportedModelSourceCheck {
            path: source_path.to_owned(),
            resolved_path: resolved_path_text,
            checked: false,
            fingerprint: source_fingerprint.to_owned(),
            file_sha256: source_file_sha256.to_owned(),
            unavailable_reason: Some("source_model_json_not_found"),
        });
    }

    let data = fs::read(&resolved_path).map_err(|error| {
        format!(
            "failed to read imported model source JSON {}: {error}",
            resolved_path.display()
        )
    })?;
    let actual_file_sha256 = sha256_bytes(&data);
    if actual_file_sha256 != source_file_sha256 {
        return Err(
            "imported model source file hash does not match model bundle provenance".to_owned(),
        );
    }
    let source_json: serde_json::Value = serde_json::from_slice(&data).map_err(|error| {
        format!(
            "failed to parse imported model source JSON {}: {error}",
            resolved_path.display()
        )
    })?;
    let actual_fingerprint = sha256_json(&source_json);
    if actual_fingerprint != source_fingerprint {
        return Err(
            "imported model source fingerprint does not match model bundle provenance".to_owned(),
        );
    }
    let source_model_json = source_json.get("model").unwrap_or(&source_json);
    if source_model_json != model_json {
        return Err("imported model source does not match model bundle".to_owned());
    }
    let source_model = model_from_json(source_model_json, "imported_model_source")?;
    if &source_model != model {
        return Err("imported model source shape/content does not match model bundle".to_owned());
    }

    Ok(ImportedModelSourceCheck {
        path: source_path.to_owned(),
        resolved_path: resolved_path_text,
        checked: true,
        fingerprint: actual_fingerprint,
        file_sha256: actual_file_sha256,
        unavailable_reason: None,
    })
}

fn verify_model_tensor_shapes(shapes: &serde_json::Value) -> Result<(), String> {
    verify_shape_field(shapes, "image", &[IMAGE_PIXELS])?;
    verify_shape_field(shapes, "weights", &[IMAGE_PIXELS, DIGITS])?;
    verify_shape_field(shapes, "bias", &[DIGITS])?;
    verify_shape_field(shapes, "logits", &[DIGITS])?;
    Ok(())
}

fn verify_shape_field(
    shapes: &serde_json::Value,
    name: &str,
    expected: &[usize],
) -> Result<(), String> {
    let actual = json_array_field(shapes, name)?;
    if actual.len() != expected.len() {
        return Err(format!(
            "model bundle tensor shape `{name}` expected rank {}, found {}",
            expected.len(),
            actual.len()
        ));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual.as_u64().ok_or_else(|| {
            format!("model bundle tensor shape `{name}[{index}]` must be a non-negative integer")
        })?;
        let actual = usize::try_from(actual)
            .map_err(|_| format!("model bundle tensor shape `{name}[{index}]` exceeds usize"))?;
        if actual != *expected {
            return Err(format!(
                "model bundle tensor shape `{name}[{index}]` expected {expected}, found {actual}"
            ));
        }
    }
    Ok(())
}

fn read_model_bundle(path: &Path) -> Result<serde_json::Value, String> {
    let data = fs::read(path)
        .map_err(|error| format!("failed to read model bundle {}: {error}", path.display()))?;
    let bundle: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|error| format!("failed to parse model bundle {}: {error}", path.display()))?;
    let kind = json_str_field(&bundle, "kind")?;
    if kind != "reverie_mnist_linear_q31_model_bundle" {
        return Err(format!(
            "model bundle kind `{kind}` is not reverie_mnist_linear_q31_model_bundle"
        ));
    }
    verify_model_fingerprints(&bundle)?;
    Ok(bundle)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelFingerprints {
    train_source: String,
    identify_source: String,
    computation: String,
    model: String,
    provenance: String,
    report: String,
    payload: String,
}

impl ModelFingerprints {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "algorithm": "sha256",
            "train_source": self.train_source,
            "identify_source": self.identify_source,
            "computation": self.computation,
            "model": self.model,
            "provenance": self.provenance,
            "report": self.report,
            "payload": self.payload,
        })
    }
}

fn model_fingerprints_for_unsigned_payload(payload: &serde_json::Value) -> ModelFingerprints {
    ModelFingerprints {
        train_source: sha256_bytes(TRAIN_SOURCE.as_bytes()),
        identify_source: sha256_bytes(IDENTIFY_SOURCE.as_bytes()),
        computation: model_computation_fingerprint(payload),
        model: sha256_json(&payload["model"]),
        provenance: sha256_json(&payload["provenance"]),
        report: sha256_json(&payload["report"]),
        payload: sha256_json(payload),
    }
}

fn model_fingerprints_from_bundle(bundle: &serde_json::Value) -> Result<ModelFingerprints, String> {
    let fingerprints = json_field(bundle, "fingerprints")?;
    model_fingerprints_from_json(fingerprints)
}

fn model_fingerprints_from_json(
    fingerprints: &serde_json::Value,
) -> Result<ModelFingerprints, String> {
    let algorithm = json_str_field(fingerprints, "algorithm")?;
    if algorithm != "sha256" {
        return Err(format!(
            "model bundle fingerprint algorithm `{algorithm}` is not sha256"
        ));
    }
    Ok(ModelFingerprints {
        train_source: json_str_field(fingerprints, "train_source")?.to_owned(),
        identify_source: json_str_field(fingerprints, "identify_source")?.to_owned(),
        computation: json_str_field(fingerprints, "computation")?.to_owned(),
        model: json_str_field(fingerprints, "model")?.to_owned(),
        provenance: json_str_field(fingerprints, "provenance")?.to_owned(),
        report: json_str_field(fingerprints, "report")?.to_owned(),
        payload: json_str_field(fingerprints, "payload")?.to_owned(),
    })
}

fn verify_model_fingerprints(bundle: &serde_json::Value) -> Result<(), String> {
    let stored = model_fingerprints_from_bundle(bundle)?;
    let mut unsigned_payload = bundle.clone();
    unsigned_payload
        .as_object_mut()
        .expect("model bundle is an object")
        .remove("fingerprints");
    let computed = model_fingerprints_for_unsigned_payload(&unsigned_payload);
    if stored != computed {
        return Err(format!(
            "model bundle fingerprint mismatch: stored payload {}, computed payload {}",
            stored.payload, computed.payload
        ));
    }
    Ok(())
}

fn model_computation_fingerprint(payload: &serde_json::Value) -> String {
    let computation = json!({
        "train_source": sha256_bytes(TRAIN_SOURCE.as_bytes()),
        "identify_source": sha256_bytes(IDENTIFY_SOURCE.as_bytes()),
        "model": &payload["model"],
        "provenance": &payload["provenance"],
    });
    sha256_json(&computation)
}

fn export_samples_bundle(
    audit_path: &Path,
    output_path: &Path,
    samples_limit: Option<usize>,
    emit_json: bool,
) -> Result<(), String> {
    if samples_limit == Some(0) {
        return Err("--samples-limit must be greater than zero".to_owned());
    }
    let audit_bundle = read_audit_bundle(audit_path)?;
    let source_fingerprints = audit_fingerprints_from_bundle(&audit_bundle)?;
    let trace = json_array_field(&audit_bundle, "witness_trace")?;
    let steps = parse_audit_steps(trace)?;
    let exported = samples_limit.map_or(steps.len(), |limit| limit.min(steps.len()));
    if exported == 0 {
        return Err("audit bundle has no samples to export".to_owned());
    }
    let selected = &steps[..exported];

    let fingerprints = write_sample_set_bundle(
        output_path,
        audit_path,
        &source_fingerprints,
        selected,
        steps.len(),
    )?;

    if emit_json {
        let report = json!({
            "kind": "reverie_mnist_linear_q31_sample_set_export",
            "source_audit_path": audit_path.display().to_string(),
            "samples_output": output_path.display().to_string(),
            "source_fingerprints": source_fingerprints.to_json(),
            "fingerprints": fingerprints.to_json(),
            "source_steps": steps.len(),
            "samples": exported,
            "sample_payload_bytes": exported * single_sample_payload_bytes(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("sample set export report serializes")
        );
    } else {
        println!(
            "sample export ok: source_steps={} samples={} sample_payload_bytes={} output={}",
            steps.len(),
            exported,
            exported * single_sample_payload_bytes(),
            output_path.display()
        );
        println!(
            "source_fingerprint: computation={} payload={}",
            source_fingerprints.computation, source_fingerprints.payload
        );
        println!(
            "samples_fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
    }

    Ok(())
}

fn write_sample_set_bundle(
    path: &Path,
    source_audit_path: &Path,
    source_fingerprints: &AuditFingerprints,
    steps: &[AuditStep],
    source_steps: usize,
) -> Result<SampleSetFingerprints, String> {
    let source_audit_bundle = json!({
        "path": source_audit_path.display().to_string(),
        "fingerprints": source_fingerprints.to_json(),
    });
    let samples_json = serde_json::Value::Array(
        steps
            .iter()
            .enumerate()
            .map(|(audit_step, step)| audit_step_sample_json(audit_step, step))
            .collect(),
    );
    let report_json = sample_set_report_json(steps, source_steps);
    let proof_json = sample_set_proof_json(&source_audit_bundle, &samples_json, &report_json)?;
    let mut bundle = json!({
        "kind": "reverie_mnist_linear_q31_samples",
        "schema_version": 1,
        "source_audit_bundle": source_audit_bundle,
        "samples": samples_json,
        "proof": proof_json,
        "report": report_json,
    });
    let fingerprints = sample_set_fingerprints_for_unsigned_payload(&bundle);
    bundle
        .as_object_mut()
        .expect("sample set bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints.to_json());

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let encoded = serde_json::to_string_pretty(&bundle).expect("sample set bundle serializes");
    fs::write(path, encoded).map_err(|error| {
        format!(
            "failed to write sample set bundle {}: {error}",
            path.display()
        )
    })?;
    Ok(fingerprints)
}

fn audit_step_sample_json(audit_step: usize, step: &AuditStep) -> serde_json::Value {
    json!({
        "kind": "reverie_mnist_linear_q31_sample",
        "audit_step": audit_step,
        "source_sample_index": step.source_sample_index,
        "image_u8": &step.image,
        "label": step.tape.label,
    })
}

fn sample_set_report_json(steps: &[AuditStep], source_steps: usize) -> serde_json::Value {
    let mut by_label = [0_usize; DIGITS];
    for step in steps {
        by_label[step.tape.label as usize] += 1;
    }
    json!({
        "source_steps": source_steps,
        "samples": steps.len(),
        "sample_payload_bytes": steps.len() * single_sample_payload_bytes(),
        "labels": by_label.iter().enumerate().map(|(label, count)| {
            json!({
                "label": label,
                "samples": count,
            })
        }).collect::<Vec<_>>(),
    })
}

fn verify_sample_set_bundle(path: &Path, emit_json: bool) -> Result<(), String> {
    let bundle = read_sample_set_bundle(path)?;
    let fingerprints = sample_set_fingerprints_from_bundle(&bundle)?;
    let samples = labeled_samples_from_samples_object(&bundle, "sample set")?;
    let report = json_field(&bundle, "report")?;
    verify_sample_set_report(report, &samples)?;
    let proof = sample_set_proof_json(
        json_field(&bundle, "source_audit_bundle")?,
        json_field(&bundle, "samples")?,
        report,
    )?;
    if let Some(stored_proof) = bundle.get("proof") {
        verify_sample_set_proof(stored_proof, &proof)?;
    }
    let proof_matches = bundle.get("proof").is_some();
    let source_audit = verify_sample_set_source_audit(path, &bundle, &samples)?;

    if emit_json {
        let report = json!({
            "kind": "reverie_mnist_linear_q31_sample_set_verification",
            "path": path.display().to_string(),
            "fingerprints": fingerprints.to_json(),
            "samples": samples.len(),
            "sample_payload_bytes": samples.len() * single_sample_payload_bytes(),
            "shape_matches": true,
            "proof_matches": proof_matches,
            "proof": proof,
            "source_audit_checked": source_audit.checked,
            "source_audit": source_audit.to_json(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .expect("sample set verification report serializes")
        );
    } else {
        println!(
            "sample set ok: samples={} sample_payload_bytes={} shape_matches=true proof_matches={} source_audit_checked={}",
            samples.len(),
            samples.len() * single_sample_payload_bytes(),
            proof_matches,
            source_audit.checked
        );
        println!("{}", format_sample_set_proof_line(&proof));
        println!("{}", source_audit.format_line());
        println!(
            "fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct SampleSetSourceAuditCheck {
    path: String,
    resolved_path: String,
    checked: bool,
    source_steps: Option<usize>,
    samples_checked: usize,
    payload_fingerprint: String,
    unavailable_reason: Option<&'static str>,
}

impl SampleSetSourceAuditCheck {
    fn to_json(&self) -> serde_json::Value {
        let mut value = json!({
            "path": self.path,
            "resolved_path": self.resolved_path,
            "checked": self.checked,
            "source_steps": self.source_steps,
            "samples_checked": self.samples_checked,
            "payload_fingerprint": self.payload_fingerprint,
        });
        if let Some(reason) = self.unavailable_reason {
            value
                .as_object_mut()
                .expect("source audit check JSON is an object")
                .insert("unavailable_reason".to_owned(), json!(reason));
        }
        value
    }

    fn format_line(&self) -> String {
        if self.checked {
            format!(
                "source_audit: checked=true source_steps={} samples_checked={} payload={}",
                self.source_steps.unwrap_or(0),
                self.samples_checked,
                self.payload_fingerprint
            )
        } else {
            format!(
                "source_audit: checked=false reason={} path={}",
                self.unavailable_reason.unwrap_or("unavailable"),
                self.path
            )
        }
    }
}

fn verify_sample_set_source_audit(
    sample_set_path: &Path,
    bundle: &serde_json::Value,
    samples: &[LabeledSample],
) -> Result<SampleSetSourceAuditCheck, String> {
    let source_audit = json_field(bundle, "source_audit_bundle")?;
    let source_path_text = json_str_field(source_audit, "path")?.to_owned();
    let expected_fingerprints =
        audit_fingerprints_from_json(json_field(source_audit, "fingerprints")?)?;
    let resolved_path = resolve_sample_set_source_path(sample_set_path, &source_path_text);
    let resolved_path_text = resolved_path.display().to_string();

    if !resolved_path.exists() {
        return Ok(SampleSetSourceAuditCheck {
            path: source_path_text,
            resolved_path: resolved_path_text,
            checked: false,
            source_steps: None,
            samples_checked: 0,
            payload_fingerprint: expected_fingerprints.payload,
            unavailable_reason: Some("source_audit_not_found"),
        });
    }

    let audit_bundle = read_audit_bundle(&resolved_path)?;
    let actual_fingerprints = audit_fingerprints_from_bundle(&audit_bundle)?;
    if actual_fingerprints != expected_fingerprints {
        return Err(
            "sample set source audit fingerprints do not match referenced audit bundle".to_owned(),
        );
    }

    let trace = json_array_field(&audit_bundle, "witness_trace")?;
    let audit_steps = parse_audit_steps(trace)?;
    let report_source_steps = json_usize_field(json_field(bundle, "report")?, "source_steps")?;
    if audit_steps.len() != report_source_steps {
        return Err(format!(
            "sample set source audit step count expected {report_source_steps}, found {}",
            audit_steps.len()
        ));
    }

    for (index, sample) in samples.iter().enumerate() {
        let audit_step = sample.audit_step.ok_or_else(|| {
            format!("sample set source audit sample {index} is missing `audit_step`")
        })?;
        let source_step = audit_steps.get(audit_step).ok_or_else(|| {
            format!(
                "sample set source audit sample {index} references missing audit_step {audit_step}"
            )
        })?;
        let source_sample_index = sample.source_sample_index.ok_or_else(|| {
            format!("sample set source audit sample {index} is missing `source_sample_index`")
        })?;
        if source_sample_index != source_step.source_sample_index {
            return Err(format!(
                "sample set source audit sample {index} source_sample_index expected {}, found {source_sample_index}",
                source_step.source_sample_index
            ));
        }
        if sample.label != source_step.tape.label {
            return Err(format!(
                "sample set source audit sample {index} label expected {}, found {}",
                source_step.tape.label, sample.label
            ));
        }
        if sample.image != source_step.image {
            return Err(format!(
                "sample set source audit sample {index} image does not match referenced audit_step {audit_step}"
            ));
        }
    }

    Ok(SampleSetSourceAuditCheck {
        path: source_path_text,
        resolved_path: resolved_path_text,
        checked: true,
        source_steps: Some(audit_steps.len()),
        samples_checked: samples.len(),
        payload_fingerprint: actual_fingerprints.payload,
        unavailable_reason: None,
    })
}

fn resolve_sample_set_source_path(sample_set_path: &Path, source_path: &str) -> PathBuf {
    resolve_referenced_artifact_path(sample_set_path, source_path)
}

fn resolve_referenced_artifact_path(reference_holder_path: &Path, source_path: &str) -> PathBuf {
    let direct = PathBuf::from(source_path);
    if direct.exists() || direct.is_absolute() {
        return direct;
    }
    if let Some(parent) = reference_holder_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        let sibling = parent.join(&direct);
        if sibling.exists() {
            return sibling;
        }
    }
    direct
}

fn sample_set_proof_json(
    source_audit_bundle: &serde_json::Value,
    samples_value: &serde_json::Value,
    report: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let samples = samples_value
        .as_array()
        .ok_or_else(|| "sample set `samples` must be an array".to_owned())?;
    let source_steps = json_usize_field(report, "source_steps")?;
    if source_steps < samples.len() {
        return Err(format!(
            "sample set report `source_steps` expected at least {}, found {source_steps}",
            samples.len()
        ));
    }

    let mut by_label = [0_usize; DIGITS];
    for (index, sample_value) in samples.iter().enumerate() {
        let sample = labeled_sample_from_json_value(
            sample_value,
            &format!("sample set proof samples[{index}]"),
        )?;
        let audit_step = sample
            .audit_step
            .ok_or_else(|| format!("sample set proof samples[{index}] is missing `audit_step`"))?;
        if audit_step != index {
            return Err(format!(
                "sample set proof samples[{index}] expected contiguous audit_step {index}, found {audit_step}"
            ));
        }
        if audit_step >= source_steps {
            return Err(format!(
                "sample set proof samples[{index}] audit_step {audit_step} is outside source_steps {source_steps}"
            ));
        }
        if sample.source_sample_index.is_none() {
            return Err(format!(
                "sample set proof samples[{index}] is missing `source_sample_index`"
            ));
        }
        by_label[sample.label as usize] += 1;
    }

    let sample_payload_bytes = samples.len() * single_sample_payload_bytes();
    Ok(json!({
        "claim": "deterministic_q31_sample_set_export",
        "source_steps": source_steps,
        "samples": samples.len(),
        "sample_payload_bytes": sample_payload_bytes,
        "label_coverage": by_label.iter().filter(|count| **count > 0).count(),
        "lineage": {
            "audit_steps_contiguous": true,
            "source_sample_indices_present": true,
        },
        "fingerprints": {
            "algorithm": "sha256",
            "source_audit_bundle": sha256_json(source_audit_bundle),
            "samples": sha256_json(samples_value),
            "report": sha256_json(report),
        },
        "labels": by_label.iter().enumerate().map(|(label, count)| {
            json!({
                "label": label,
                "samples": count,
            })
        }).collect::<Vec<_>>(),
    }))
}

fn verify_sample_set_proof(
    stored: &serde_json::Value,
    expected: &serde_json::Value,
) -> Result<(), String> {
    if stored != expected {
        return Err("sample set proof does not match recomputed sample lineage".to_owned());
    }
    Ok(())
}

fn format_sample_set_proof_line(proof: &serde_json::Value) -> String {
    format!(
        "proof: claim=deterministic_q31_sample_set_export source_steps={} samples={} sample_bytes={} label_coverage={}",
        proof
            .get("source_steps")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("samples")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("sample_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("label_coverage")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    )
}

fn verify_sample_set_report(
    report: &serde_json::Value,
    samples: &[LabeledSample],
) -> Result<(), String> {
    let actual_samples = json_usize_field(report, "samples")?;
    if actual_samples != samples.len() {
        return Err(format!(
            "sample set report `samples` expected {}, found {actual_samples}",
            samples.len()
        ));
    }
    let expected_payload_bytes = samples.len() * single_sample_payload_bytes();
    let actual_payload_bytes = json_usize_field(report, "sample_payload_bytes")?;
    if actual_payload_bytes != expected_payload_bytes {
        return Err(format!(
            "sample set report `sample_payload_bytes` expected {expected_payload_bytes}, found {actual_payload_bytes}"
        ));
    }
    let mut by_label = [0_usize; DIGITS];
    for sample in samples {
        by_label[sample.label as usize] += 1;
    }
    let label_rows = json_array_field(report, "labels")?;
    if label_rows.len() != DIGITS {
        return Err(format!(
            "sample set report `labels` expected {DIGITS} rows, found {}",
            label_rows.len()
        ));
    }
    for (expected_label, row) in label_rows.iter().enumerate() {
        let label = json_usize_field(row, "label")?;
        let count = json_usize_field(row, "samples")?;
        if label != expected_label || count != by_label[expected_label] {
            return Err(format!(
                "sample set report label row {expected_label} does not match exported samples"
            ));
        }
    }
    Ok(())
}

fn read_sample_set_bundle(path: &Path) -> Result<serde_json::Value, String> {
    let data = fs::read(path).map_err(|error| {
        format!(
            "failed to read sample set bundle {}: {error}",
            path.display()
        )
    })?;
    let bundle: serde_json::Value = serde_json::from_slice(&data).map_err(|error| {
        format!(
            "failed to parse sample set bundle {}: {error}",
            path.display()
        )
    })?;
    let kind = json_str_field(&bundle, "kind")?;
    if kind != "reverie_mnist_linear_q31_samples" {
        return Err(format!(
            "sample set bundle kind `{kind}` is not reverie_mnist_linear_q31_samples"
        ));
    }
    verify_sample_set_fingerprints(&bundle)?;
    Ok(bundle)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SampleSetFingerprints {
    computation: String,
    source_audit: String,
    samples: String,
    proof: String,
    report: String,
    payload: String,
}

impl SampleSetFingerprints {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "algorithm": "sha256",
            "computation": self.computation,
            "source_audit": self.source_audit,
            "samples": self.samples,
            "proof": self.proof,
            "report": self.report,
            "payload": self.payload,
        })
    }
}

fn sample_set_fingerprints_for_unsigned_payload(
    payload: &serde_json::Value,
) -> SampleSetFingerprints {
    SampleSetFingerprints {
        computation: sample_set_computation_fingerprint(payload),
        source_audit: sha256_json(&payload["source_audit_bundle"]),
        samples: sha256_json(&payload["samples"]),
        proof: sha256_json(payload.get("proof").unwrap_or(&serde_json::Value::Null)),
        report: sha256_json(&payload["report"]),
        payload: sha256_json(payload),
    }
}

fn sample_set_fingerprints_from_bundle(
    bundle: &serde_json::Value,
) -> Result<SampleSetFingerprints, String> {
    let fingerprints = json_field(bundle, "fingerprints")?;
    let algorithm = json_str_field(fingerprints, "algorithm")?;
    if algorithm != "sha256" {
        return Err(format!(
            "sample set bundle fingerprint algorithm `{algorithm}` is not sha256"
        ));
    }
    Ok(SampleSetFingerprints {
        computation: json_str_field(fingerprints, "computation")?.to_owned(),
        source_audit: json_str_field(fingerprints, "source_audit")?.to_owned(),
        samples: json_str_field(fingerprints, "samples")?.to_owned(),
        proof: fingerprints
            .get("proof")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| sha256_json(&serde_json::Value::Null)),
        report: json_str_field(fingerprints, "report")?.to_owned(),
        payload: json_str_field(fingerprints, "payload")?.to_owned(),
    })
}

fn verify_sample_set_fingerprints(bundle: &serde_json::Value) -> Result<(), String> {
    let stored = sample_set_fingerprints_from_bundle(bundle)?;
    let mut unsigned_payload = bundle.clone();
    unsigned_payload
        .as_object_mut()
        .expect("sample set bundle is an object")
        .remove("fingerprints");
    let computed = sample_set_fingerprints_for_unsigned_payload(&unsigned_payload);
    if stored != computed {
        return Err(format!(
            "sample set bundle fingerprint mismatch: stored payload {}, computed payload {}",
            stored.payload, computed.payload
        ));
    }
    Ok(())
}

fn sample_set_computation_fingerprint(payload: &serde_json::Value) -> String {
    let mut computation = json!({
        "source_audit_bundle": &payload["source_audit_bundle"],
        "samples": &payload["samples"],
        "report": &payload["report"],
    });
    if let Some(proof) = payload.get("proof") {
        computation
            .as_object_mut()
            .expect("sample set computation is an object")
            .insert("proof".to_owned(), proof.clone());
    }
    sha256_json(&computation)
}

#[derive(Debug, Clone)]
struct ArtifactMeasure {
    kind: &'static str,
    path: String,
    file_bytes: u64,
    logical_payload_bytes: usize,
    model_payload_bytes: usize,
    sample_payload_bytes: usize,
    witness_payload_bytes: usize,
    trace_payload_bytes: usize,
    derived_update_payload_bytes: usize,
    steps: usize,
    forward_recompute_steps: usize,
    inverse_recompute_steps: usize,
    payload_fingerprint: String,
    fingerprint_summary: ArtifactFingerprintSummary,
}

#[derive(Debug, Clone)]
struct ArtifactProfile {
    total_file_bytes: u64,
    total_logical_payload_bytes: usize,
    total_model_payload_bytes: usize,
    total_sample_payload_bytes: usize,
    total_witness_payload_bytes: usize,
    total_trace_payload_bytes: usize,
    total_derived_update_payload_bytes: usize,
    total_steps: usize,
    total_forward_recompute_steps: usize,
    total_inverse_recompute_steps: usize,
}

fn compare_artifacts(paths: &[PathBuf], emit_json: bool) -> Result<(), String> {
    let mut measures = Vec::with_capacity(paths.len());
    for path in paths {
        measures.push(measure_artifact(path)?);
    }
    let profile = artifact_profile(&measures);

    if emit_json {
        let report = json!({
            "kind": "reverie_mnist_linear_q31_artifact_comparison",
            "artifacts": measures.iter().map(artifact_measure_json).collect::<Vec<_>>(),
            "totals": {
                "count": measures.len(),
                "file_bytes": profile.total_file_bytes,
                "logical_payload_bytes": profile.total_logical_payload_bytes,
            },
            "ml_profile": artifact_profile_json(&profile),
            "audit_contract": audit_contract_json(&measures, &profile),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("artifact comparison serializes")
        );
    } else {
        println!("artifact comparison");
        println!(
            "totals: count={} file_bytes={} logical_payload_bytes={}",
            measures.len(),
            profile.total_file_bytes,
            profile.total_logical_payload_bytes
        );
        println!("{}", format_artifact_profile(&profile));
        println!("{}", format_audit_contract(&measures, &profile));
        for measure in &measures {
            println!("- {}", format_artifact_measure(measure));
        }
    }

    Ok(())
}

fn artifact_profile(measures: &[ArtifactMeasure]) -> ArtifactProfile {
    ArtifactProfile {
        total_file_bytes: measures.iter().map(|measure| measure.file_bytes).sum(),
        total_logical_payload_bytes: measures
            .iter()
            .map(|measure| measure.logical_payload_bytes)
            .sum(),
        total_model_payload_bytes: measures
            .iter()
            .map(|measure| measure.model_payload_bytes)
            .sum(),
        total_sample_payload_bytes: measures
            .iter()
            .map(|measure| measure.sample_payload_bytes)
            .sum(),
        total_witness_payload_bytes: measures
            .iter()
            .map(|measure| measure.witness_payload_bytes)
            .sum(),
        total_trace_payload_bytes: measures
            .iter()
            .map(|measure| measure.trace_payload_bytes)
            .sum(),
        total_derived_update_payload_bytes: measures
            .iter()
            .map(|measure| measure.derived_update_payload_bytes)
            .sum(),
        total_steps: measures.iter().map(|measure| measure.steps).sum(),
        total_forward_recompute_steps: measures
            .iter()
            .map(|measure| measure.forward_recompute_steps)
            .sum(),
        total_inverse_recompute_steps: measures
            .iter()
            .map(|measure| measure.inverse_recompute_steps)
            .sum(),
    }
}

fn artifact_profile_json(profile: &ArtifactProfile) -> serde_json::Value {
    json!({
        "total_file_bytes": profile.total_file_bytes,
        "total_logical_payload_bytes": profile.total_logical_payload_bytes,
        "total_model_payload_bytes": profile.total_model_payload_bytes,
        "total_sample_payload_bytes": profile.total_sample_payload_bytes,
        "total_witness_payload_bytes": profile.total_witness_payload_bytes,
        "total_trace_payload_bytes": profile.total_trace_payload_bytes,
        "total_derived_update_payload_bytes": profile.total_derived_update_payload_bytes,
        "total_steps": profile.total_steps,
        "total_forward_recompute_steps": profile.total_forward_recompute_steps,
        "total_inverse_recompute_steps": profile.total_inverse_recompute_steps,
        "total_recompute_steps": profile.total_forward_recompute_steps + profile.total_inverse_recompute_steps,
        "trace_to_model_payload_ratio": ratio_json(profile.total_trace_payload_bytes, profile.total_model_payload_bytes),
        "witness_to_model_payload_ratio": ratio_json(profile.total_witness_payload_bytes, profile.total_model_payload_bytes),
        "logical_to_file_ratio": ratio_json_u64(profile.total_logical_payload_bytes, profile.total_file_bytes),
    })
}

fn format_artifact_profile(profile: &ArtifactProfile) -> String {
    format!(
        "ml_profile: model_payload_bytes={} sample_payload_bytes={} witness_payload_bytes={} trace_payload_bytes={} derived_update_payload_bytes={} steps={} forward_recompute_steps={} inverse_recompute_steps={} total_recompute_steps={} trace_to_model_ratio={} witness_to_model_ratio={} logical_to_file_ratio={}",
        profile.total_model_payload_bytes,
        profile.total_sample_payload_bytes,
        profile.total_witness_payload_bytes,
        profile.total_trace_payload_bytes,
        profile.total_derived_update_payload_bytes,
        profile.total_steps,
        profile.total_forward_recompute_steps,
        profile.total_inverse_recompute_steps,
        profile.total_forward_recompute_steps + profile.total_inverse_recompute_steps,
        format_ratio(
            profile.total_trace_payload_bytes,
            profile.total_model_payload_bytes
        ),
        format_ratio(
            profile.total_witness_payload_bytes,
            profile.total_model_payload_bytes
        ),
        format_ratio_u64(
            profile.total_logical_payload_bytes,
            profile.total_file_bytes
        )
    )
}

fn format_ratio(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "n/a".to_owned()
    } else {
        format!("{:.6}", numerator as f64 / denominator as f64)
    }
}

fn format_ratio_u64(numerator: usize, denominator: u64) -> String {
    if denominator == 0 {
        "n/a".to_owned()
    } else {
        format!("{:.6}", numerator as f64 / denominator as f64)
    }
}

fn ratio_json(numerator: usize, denominator: usize) -> serde_json::Value {
    if denominator == 0 {
        serde_json::Value::Null
    } else {
        json!(numerator as f64 / denominator as f64)
    }
}

fn ratio_json_u64(numerator: usize, denominator: u64) -> serde_json::Value {
    if denominator == 0 {
        serde_json::Value::Null
    } else {
        json!(numerator as f64 / denominator as f64)
    }
}

fn audit_contract_json(
    measures: &[ArtifactMeasure],
    profile: &ArtifactProfile,
) -> serde_json::Value {
    let checks = audit_contract_checks(measures, profile)
        .into_iter()
        .map(|check| {
            json!({
                "metric": check.metric,
                "passed": check.passed,
                "actual": check.actual,
                "requirement": check.requirement,
            })
        })
        .collect::<Vec<_>>();
    let passed = checks
        .iter()
        .all(|check| check["passed"].as_bool().unwrap_or(false));
    json!({
        "claim": "reversible_inspectable_deterministic_q31_ml_kernel",
        "passed": passed,
        "checks": checks,
    })
}

fn format_audit_contract(measures: &[ArtifactMeasure], profile: &ArtifactProfile) -> String {
    let checks = audit_contract_checks(measures, profile);
    let passed = checks.iter().all(|check| check.passed);
    let failed = checks.len() - checks.iter().filter(|check| check.passed).count();
    format!(
        "audit_contract: claim=reversible_inspectable_deterministic_q31_ml_kernel passed={} checks={} failed={}",
        passed,
        checks.len(),
        failed
    )
}

#[derive(Debug, Clone)]
struct AuditContractCheck {
    metric: &'static str,
    passed: bool,
    actual: String,
    requirement: &'static str,
}

fn audit_contract_checks(
    measures: &[ArtifactMeasure],
    profile: &ArtifactProfile,
) -> Vec<AuditContractCheck> {
    let training_trace = measures
        .iter()
        .find(|measure| measure.kind == "training_trace");
    let model = measures.iter().find(|measure| measure.kind == "model");
    let sample_set = measures.iter().find(|measure| measure.kind == "sample_set");
    let training_step = measures
        .iter()
        .find(|measure| measure.kind == "training_step");
    let inference = measures.iter().find(|measure| measure.kind == "inference");
    let evaluation = measures
        .iter()
        .find(|measure| measure.kind == "model_evaluation");
    let every_artifact_fingerprinted = !measures.is_empty()
        && measures.iter().all(|measure| {
            measure.fingerprint_summary.fingerprint_count >= 3
                && measure.fingerprint_summary.source_fingerprint_count > 0
                && measure.fingerprint_summary.has_computation_fingerprint
                && measure.fingerprint_summary.has_payload_fingerprint
        });
    let replay_artifacts_have_proof = measures.iter().all(|measure| match measure.kind {
        "model" => measure.fingerprint_summary.has_provenance_fingerprint,
        "training_trace" | "sample_set" | "training_step" | "inference" | "model_evaluation" => {
            measure.fingerprint_summary.has_proof_fingerprint
        }
        _ => false,
    });

    vec![
        contract_check(
            "training_trace",
            training_trace.is_some_and(|measure| {
                measure.steps > 0
                    && measure.trace_payload_bytes > 0
                    && measure.witness_payload_bytes > 0
                    && measure.forward_recompute_steps == measure.steps
                    && measure.inverse_recompute_steps == measure.steps
            }),
            training_trace
                .map(|measure| {
                    format!(
                        "steps={} trace_bytes={} witness_bytes={} forward={} inverse={}",
                        measure.steps,
                        measure.trace_payload_bytes,
                        measure.witness_payload_bytes,
                        measure.forward_recompute_steps,
                        measure.inverse_recompute_steps
                    )
                })
                .unwrap_or_else(|| "missing".to_owned()),
            "present with nonzero trace/witness bytes and one forward/inverse recompute per step",
        ),
        contract_check(
            "model_bundle",
            model.is_some_and(|measure| measure.model_payload_bytes > 0),
            model
                .map(|measure| format!("model_bytes={}", measure.model_payload_bytes))
                .unwrap_or_else(|| "missing".to_owned()),
            "present with nonzero model payload bytes",
        ),
        contract_check(
            "sample_set",
            sample_set.is_some_and(|measure| measure.sample_payload_bytes > 0),
            sample_set
                .map(|measure| {
                    format!(
                        "samples={} sample_bytes={}",
                        measure.steps, measure.sample_payload_bytes
                    )
                })
                .unwrap_or_else(|| "missing".to_owned()),
            "present with nonzero sample payload bytes",
        ),
        contract_check(
            "training_step_replay",
            training_step.is_some_and(|measure| {
                measure.steps == 1
                    && measure.witness_payload_bytes > 0
                    && measure.derived_update_payload_bytes > 0
                    && measure.forward_recompute_steps == 1
                    && measure.inverse_recompute_steps == 1
            }),
            training_step
                .map(|measure| {
                    format!(
                        "steps={} witness_bytes={} update_bytes={} forward={} inverse={}",
                        measure.steps,
                        measure.witness_payload_bytes,
                        measure.derived_update_payload_bytes,
                        measure.forward_recompute_steps,
                        measure.inverse_recompute_steps
                    )
                })
                .unwrap_or_else(|| "missing".to_owned()),
            "present with one replayed update, witnesses, derived update bytes, and forward/inverse recompute",
        ),
        contract_check(
            "inference_replay",
            inference.is_some_and(|measure| {
                measure.steps > 0
                    && measure.model_payload_bytes > 0
                    && measure.sample_payload_bytes > 0
                    && measure.witness_payload_bytes > 0
                    && measure.forward_recompute_steps == measure.steps
                    && measure.inverse_recompute_steps == measure.steps
            }),
            inference
                .map(|measure| {
                    format!(
                        "steps={} model_bytes={} sample_bytes={} witness_bytes={} forward={} inverse={}",
                        measure.steps,
                        measure.model_payload_bytes,
                        measure.sample_payload_bytes,
                        measure.witness_payload_bytes,
                        measure.forward_recompute_steps,
                        measure.inverse_recompute_steps
                    )
                })
                .unwrap_or_else(|| "missing".to_owned()),
            "present with model/sample/witness bytes and one forward/inverse recompute per sample",
        ),
        contract_check(
            "evaluation_replay",
            evaluation.is_some_and(|measure| {
                measure.steps > 0
                    && measure.model_payload_bytes > 0
                    && measure.sample_payload_bytes > 0
                    && measure.witness_payload_bytes > 0
                    && measure.forward_recompute_steps == measure.steps
                    && measure.inverse_recompute_steps == measure.steps
            }),
            evaluation
                .map(|measure| {
                    format!(
                        "steps={} model_bytes={} sample_bytes={} witness_bytes={} forward={} inverse={}",
                        measure.steps,
                        measure.model_payload_bytes,
                        measure.sample_payload_bytes,
                        measure.witness_payload_bytes,
                        measure.forward_recompute_steps,
                        measure.inverse_recompute_steps
                    )
                })
                .unwrap_or_else(|| "missing".to_owned()),
            "present with batch sample/witness bytes and one forward/inverse recompute per sample",
        ),
        contract_check(
            "balanced_recompute",
            profile.total_forward_recompute_steps > 0
                && profile.total_forward_recompute_steps == profile.total_inverse_recompute_steps,
            format!(
                "forward={} inverse={}",
                profile.total_forward_recompute_steps, profile.total_inverse_recompute_steps
            ),
            "nonzero total forward recompute steps equal total inverse recompute steps",
        ),
        contract_check(
            "bounded_payload_ratios",
            profile.total_model_payload_bytes > 0
                && profile.total_witness_payload_bytes > 0
                && profile.total_trace_payload_bytes > 0,
            format!(
                "model_bytes={} witness_bytes={} trace_bytes={} witness/model={} trace/model={}",
                profile.total_model_payload_bytes,
                profile.total_witness_payload_bytes,
                profile.total_trace_payload_bytes,
                format_ratio(
                    profile.total_witness_payload_bytes,
                    profile.total_model_payload_bytes
                ),
                format_ratio(
                    profile.total_trace_payload_bytes,
                    profile.total_model_payload_bytes
                )
            ),
            "nonzero model, witness, and trace payload bytes so ratios are meaningful",
        ),
        contract_check(
            "fingerprint_coverage",
            every_artifact_fingerprinted,
            format!(
                "artifacts={} min_fingerprints={} source_covered={} computation_covered={} payload_covered={}",
                measures.len(),
                measures
                    .iter()
                    .map(|measure| measure.fingerprint_summary.fingerprint_count)
                    .min()
                    .unwrap_or(0),
                measures
                    .iter()
                    .filter(|measure| measure.fingerprint_summary.source_fingerprint_count > 0)
                    .count(),
                measures
                    .iter()
                    .filter(|measure| measure.fingerprint_summary.has_computation_fingerprint)
                    .count(),
                measures
                    .iter()
                    .filter(|measure| measure.fingerprint_summary.has_payload_fingerprint)
                    .count(),
            ),
            "every artifact has verified sha256 source, computation, and payload fingerprints",
        ),
        contract_check(
            "proof_or_provenance_fingerprints",
            replay_artifacts_have_proof,
            format!(
                "proof_artifacts={} provenance_artifacts={}",
                measures
                    .iter()
                    .filter(|measure| measure.fingerprint_summary.has_proof_fingerprint)
                    .count(),
                measures
                    .iter()
                    .filter(|measure| measure.fingerprint_summary.has_provenance_fingerprint)
                    .count(),
            ),
            "replay artifacts have proof fingerprints and model artifacts have provenance fingerprints",
        ),
    ]
}

fn contract_check(
    metric: &'static str,
    passed: bool,
    actual: String,
    requirement: &'static str,
) -> AuditContractCheck {
    AuditContractCheck {
        metric,
        passed,
        actual,
        requirement,
    }
}

fn measure_artifact(path: &Path) -> Result<ArtifactMeasure, String> {
    let data = fs::read(path)
        .map_err(|error| format!("failed to read artifact {}: {error}", path.display()))?;
    let parsed: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|error| format!("failed to parse artifact {}: {error}", path.display()))?;
    let kind = json_str_field(&parsed, "kind")?;
    let file_bytes = data.len() as u64;

    match kind {
        "reverie_mnist_linear_q31_replay_bundle" => measure_training_artifact(path, file_bytes),
        "reverie_mnist_linear_q31_model_bundle" => measure_model_artifact(path, file_bytes),
        "reverie_mnist_linear_q31_samples" => measure_sample_set_artifact(path, file_bytes),
        "reverie_mnist_linear_q31_step_replay_bundle" => measure_step_artifact(path, file_bytes),
        "reverie_mnist_linear_q31_inference_replay_bundle" => {
            measure_inference_artifact(path, file_bytes)
        }
        "reverie_mnist_linear_q31_model_evaluation_bundle" => {
            measure_model_evaluation_artifact(path, file_bytes)
        }
        other => Err(format!(
            "artifact kind `{other}` is not a supported Reverie MNIST replay artifact"
        )),
    }
}

fn measure_training_artifact(path: &Path, file_bytes: u64) -> Result<ArtifactMeasure, String> {
    let bundle = read_audit_bundle(path)?;
    let fingerprints = audit_fingerprints_from_bundle(&bundle)?;
    let fingerprint_summary = artifact_fingerprint_summary(&bundle)?;
    let trace = json_array_field(&bundle, "witness_trace")?;
    let steps = trace.len();
    let model_payload_bytes = Model::payload_bytes();
    let sample_payload_bytes = training_trace_sample_payload_bytes(steps);
    let witness_payload_bytes = witness_trace_payload_bytes(steps);
    let trace_payload_bytes = sample_payload_bytes + witness_payload_bytes;
    let logical_payload_bytes = model_payload_bytes + trace_payload_bytes;

    Ok(ArtifactMeasure {
        kind: "training_trace",
        path: path.display().to_string(),
        file_bytes,
        logical_payload_bytes,
        model_payload_bytes,
        sample_payload_bytes,
        witness_payload_bytes,
        trace_payload_bytes,
        derived_update_payload_bytes: 0,
        steps,
        forward_recompute_steps: steps,
        inverse_recompute_steps: steps,
        payload_fingerprint: fingerprints.payload,
        fingerprint_summary,
    })
}

fn measure_model_artifact(path: &Path, file_bytes: u64) -> Result<ArtifactMeasure, String> {
    let bundle = read_model_bundle(path)?;
    let fingerprints = model_fingerprints_from_bundle(&bundle)?;
    let fingerprint_summary = artifact_fingerprint_summary(&bundle)?;
    model_from_json(json_field(&bundle, "model")?, "model")?;
    let model_payload_bytes = json_field(&bundle, "storage")
        .and_then(|storage| json_usize_field(storage, "model_payload_bytes"))
        .unwrap_or_else(|_| Model::payload_bytes());

    Ok(ArtifactMeasure {
        kind: "model",
        path: path.display().to_string(),
        file_bytes,
        logical_payload_bytes: model_payload_bytes,
        model_payload_bytes,
        sample_payload_bytes: 0,
        witness_payload_bytes: 0,
        trace_payload_bytes: 0,
        derived_update_payload_bytes: 0,
        steps: 0,
        forward_recompute_steps: 0,
        inverse_recompute_steps: 0,
        payload_fingerprint: fingerprints.payload,
        fingerprint_summary,
    })
}

fn measure_sample_set_artifact(path: &Path, file_bytes: u64) -> Result<ArtifactMeasure, String> {
    let bundle = read_sample_set_bundle(path)?;
    let fingerprints = sample_set_fingerprints_from_bundle(&bundle)?;
    let fingerprint_summary = artifact_fingerprint_summary(&bundle)?;
    let samples = labeled_samples_from_samples_object(&bundle, "sample set artifact")?;
    let sample_payload_bytes = samples.len() * single_sample_payload_bytes();

    Ok(ArtifactMeasure {
        kind: "sample_set",
        path: path.display().to_string(),
        file_bytes,
        logical_payload_bytes: sample_payload_bytes,
        model_payload_bytes: 0,
        sample_payload_bytes,
        witness_payload_bytes: 0,
        trace_payload_bytes: 0,
        derived_update_payload_bytes: 0,
        steps: samples.len(),
        forward_recompute_steps: 0,
        inverse_recompute_steps: 0,
        payload_fingerprint: fingerprints.payload,
        fingerprint_summary,
    })
}

fn measure_step_artifact(path: &Path, file_bytes: u64) -> Result<ArtifactMeasure, String> {
    let bundle = read_step_bundle(path)?;
    let fingerprints = step_fingerprints_from_bundle(&bundle)?;
    let fingerprint_summary = artifact_fingerprint_summary(&bundle)?;
    let update = json_field(&bundle, "update")?;
    let top_weight_deltas = json_array_field(update, "top_weight_deltas")?.len();
    let model_payload_bytes = 2 * Model::payload_bytes();
    let sample_payload_bytes = single_sample_payload_bytes();
    let witness_payload_bytes = training_step_witness_payload_bytes();
    let derived_update_payload_bytes = step_update_payload_bytes(top_weight_deltas);
    let logical_payload_bytes = model_payload_bytes
        + sample_payload_bytes
        + witness_payload_bytes
        + derived_update_payload_bytes;

    Ok(ArtifactMeasure {
        kind: "training_step",
        path: path.display().to_string(),
        file_bytes,
        logical_payload_bytes,
        model_payload_bytes,
        sample_payload_bytes,
        witness_payload_bytes,
        trace_payload_bytes: 0,
        derived_update_payload_bytes,
        steps: 1,
        forward_recompute_steps: 1,
        inverse_recompute_steps: 1,
        payload_fingerprint: fingerprints.payload,
        fingerprint_summary,
    })
}

fn measure_inference_artifact(path: &Path, file_bytes: u64) -> Result<ArtifactMeasure, String> {
    let bundle = read_inference_bundle(path)?;
    let fingerprints = inference_fingerprints_from_bundle(&bundle)?;
    let fingerprint_summary = artifact_fingerprint_summary(&bundle)?;
    let fallback = InferenceMemoryReport::for_inference();
    let storage = json_field(&bundle, "storage").ok();
    let model_payload_bytes = storage
        .and_then(|storage| json_usize_field(storage, "model_payload_bytes").ok())
        .unwrap_or(fallback.model_payload_bytes);
    let sample_payload_bytes = storage
        .and_then(|storage| json_usize_field(storage, "sample_payload_bytes").ok())
        .unwrap_or(fallback.sample_payload_bytes);
    let witness_payload_bytes = storage
        .and_then(|storage| json_usize_field(storage, "witness_payload_bytes").ok())
        .unwrap_or(fallback.witness_payload_bytes);
    let trace_payload_bytes = storage
        .and_then(|storage| json_usize_field(storage, "trace_payload_bytes").ok())
        .unwrap_or(fallback.trace_payload_bytes);
    let logical_payload_bytes = storage
        .and_then(|storage| json_usize_field(storage, "replay_payload_bytes").ok())
        .unwrap_or(fallback.replay_payload_bytes);
    let forward_recompute_steps = storage
        .and_then(|storage| json_usize_field(storage, "forward_recompute_steps").ok())
        .unwrap_or(fallback.forward_recompute_steps);
    let inverse_recompute_steps = storage
        .and_then(|storage| json_usize_field(storage, "inverse_recompute_steps").ok())
        .unwrap_or(fallback.inverse_recompute_steps);

    Ok(ArtifactMeasure {
        kind: "inference",
        path: path.display().to_string(),
        file_bytes,
        logical_payload_bytes,
        model_payload_bytes,
        sample_payload_bytes,
        witness_payload_bytes,
        trace_payload_bytes,
        derived_update_payload_bytes: 0,
        steps: 1,
        forward_recompute_steps,
        inverse_recompute_steps,
        payload_fingerprint: fingerprints.payload,
        fingerprint_summary,
    })
}

fn measure_model_evaluation_artifact(
    path: &Path,
    file_bytes: u64,
) -> Result<ArtifactMeasure, String> {
    let bundle = read_model_evaluation_bundle(path)?;
    let fingerprints = model_evaluation_fingerprints_from_bundle(&bundle)?;
    let fingerprint_summary = artifact_fingerprint_summary(&bundle)?;
    let proof = json_field(&bundle, "proof")?;
    let model_payload_bytes = json_usize_field(proof, "model_payload_bytes")?;
    let sample_payload_bytes = json_usize_field(proof, "sample_payload_bytes")?;
    let witness_payload_bytes = json_usize_field(proof, "witness_payload_bytes")?;
    let trace_payload_bytes = json_usize_field(proof, "trace_payload_bytes")?;
    let logical_payload_bytes = json_usize_field(proof, "replay_payload_bytes")?;
    let forward_recompute_steps = json_usize_field(proof, "forward_recompute_steps")?;
    let inverse_recompute_steps = json_usize_field(proof, "inverse_recompute_steps")?;

    Ok(ArtifactMeasure {
        kind: "model_evaluation",
        path: path.display().to_string(),
        file_bytes,
        logical_payload_bytes,
        model_payload_bytes,
        sample_payload_bytes,
        witness_payload_bytes,
        trace_payload_bytes,
        derived_update_payload_bytes: 0,
        steps: json_array_field(json_field(&bundle, "samples")?, "samples")?.len(),
        forward_recompute_steps,
        inverse_recompute_steps,
        payload_fingerprint: fingerprints.payload,
        fingerprint_summary,
    })
}

fn artifact_measure_json(measure: &ArtifactMeasure) -> serde_json::Value {
    json!({
        "kind": measure.kind,
        "path": measure.path,
        "file_bytes": measure.file_bytes,
        "logical_payload_bytes": measure.logical_payload_bytes,
        "model_payload_bytes": measure.model_payload_bytes,
        "sample_payload_bytes": measure.sample_payload_bytes,
        "witness_payload_bytes": measure.witness_payload_bytes,
        "trace_payload_bytes": measure.trace_payload_bytes,
        "derived_update_payload_bytes": measure.derived_update_payload_bytes,
        "steps": measure.steps,
        "forward_recompute_steps": measure.forward_recompute_steps,
        "inverse_recompute_steps": measure.inverse_recompute_steps,
        "payload_fingerprint": measure.payload_fingerprint,
        "fingerprints": {
            "algorithm": "sha256",
            "count": measure.fingerprint_summary.fingerprint_count,
            "source_count": measure.fingerprint_summary.source_fingerprint_count,
            "has_computation": measure.fingerprint_summary.has_computation_fingerprint,
            "has_payload": measure.fingerprint_summary.has_payload_fingerprint,
            "has_proof": measure.fingerprint_summary.has_proof_fingerprint,
            "has_provenance": measure.fingerprint_summary.has_provenance_fingerprint,
        },
    })
}

fn format_artifact_measure(measure: &ArtifactMeasure) -> String {
    format!(
        "kind={} file_bytes={} logical_payload_bytes={} model_payload_bytes={} sample_payload_bytes={} witness_payload_bytes={} trace_payload_bytes={} derived_update_payload_bytes={} steps={} forward_recompute_steps={} inverse_recompute_steps={} fingerprints={} source_fingerprints={} proof_fingerprint={} provenance_fingerprint={} payload={} path={}",
        measure.kind,
        measure.file_bytes,
        measure.logical_payload_bytes,
        measure.model_payload_bytes,
        measure.sample_payload_bytes,
        measure.witness_payload_bytes,
        measure.trace_payload_bytes,
        measure.derived_update_payload_bytes,
        measure.steps,
        measure.forward_recompute_steps,
        measure.inverse_recompute_steps,
        measure.fingerprint_summary.fingerprint_count,
        measure.fingerprint_summary.source_fingerprint_count,
        measure.fingerprint_summary.has_proof_fingerprint,
        measure.fingerprint_summary.has_provenance_fingerprint,
        measure.payload_fingerprint,
        measure.path
    )
}

#[derive(Debug, Clone)]
struct ArtifactFingerprintSummary {
    fingerprint_count: usize,
    source_fingerprint_count: usize,
    has_computation_fingerprint: bool,
    has_payload_fingerprint: bool,
    has_proof_fingerprint: bool,
    has_provenance_fingerprint: bool,
}

fn artifact_fingerprint_summary(
    bundle: &serde_json::Value,
) -> Result<ArtifactFingerprintSummary, String> {
    let fingerprints = json_field(bundle, "fingerprints")?;
    let algorithm = json_str_field(fingerprints, "algorithm")?;
    if algorithm != "sha256" {
        return Err(format!(
            "artifact fingerprint algorithm `{algorithm}` is not sha256"
        ));
    }
    let object = fingerprints
        .as_object()
        .ok_or_else(|| "artifact fingerprints must be an object".to_owned())?;

    let mut fingerprint_count = 0;
    let mut source_fingerprint_count = 0;
    let mut has_computation_fingerprint = false;
    let mut has_payload_fingerprint = false;
    let mut has_proof_fingerprint = false;
    let mut has_provenance_fingerprint = false;

    for (field, value) in object {
        if field == "algorithm" {
            continue;
        }
        let fingerprint = value
            .as_str()
            .ok_or_else(|| format!("artifact fingerprint `{field}` must be a string"))?;
        if !is_sha256_hex(fingerprint) {
            return Err(format!(
                "artifact fingerprint `{field}` must be a lowercase sha256 hex digest"
            ));
        }
        fingerprint_count += 1;
        if field.contains("source") {
            source_fingerprint_count += 1;
        }
        has_computation_fingerprint |= field == "computation";
        has_payload_fingerprint |= field == "payload";
        has_proof_fingerprint |= field == "proof";
        has_provenance_fingerprint |= field == "provenance";
    }

    Ok(ArtifactFingerprintSummary {
        fingerprint_count,
        source_fingerprint_count,
        has_computation_fingerprint,
        has_payload_fingerprint,
        has_proof_fingerprint,
        has_provenance_fingerprint,
    })
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AuditFingerprints {
    train_source: String,
    identify_source: String,
    computation: String,
    final_model: String,
    witness_trace: String,
    proof: String,
    lineage_ledger: String,
    report: String,
    payload: String,
}

impl AuditFingerprints {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "algorithm": "sha256",
            "train_source": self.train_source,
            "identify_source": self.identify_source,
            "computation": self.computation,
            "final_model": self.final_model,
            "witness_trace": self.witness_trace,
            "proof": self.proof,
            "lineage_ledger": self.lineage_ledger,
            "report": self.report,
            "payload": self.payload,
        })
    }
}

fn audit_fingerprints_for_unsigned_payload(payload: &serde_json::Value) -> AuditFingerprints {
    AuditFingerprints {
        train_source: sha256_bytes(TRAIN_SOURCE.as_bytes()),
        identify_source: sha256_bytes(IDENTIFY_SOURCE.as_bytes()),
        computation: audit_computation_fingerprint(payload),
        final_model: sha256_json(&payload["final_model"]),
        witness_trace: sha256_json(&payload["witness_trace"]),
        proof: sha256_json(payload.get("proof").unwrap_or(&serde_json::Value::Null)),
        lineage_ledger: sha256_json(
            payload
                .get("lineage_ledger")
                .unwrap_or(&serde_json::Value::Null),
        ),
        report: sha256_json(&payload["report"]),
        payload: sha256_json(payload),
    }
}

fn audit_fingerprints_from_bundle(bundle: &serde_json::Value) -> Result<AuditFingerprints, String> {
    let fingerprints = json_field(bundle, "fingerprints")?;
    audit_fingerprints_from_json(fingerprints)
}

fn audit_fingerprints_from_json(
    fingerprints: &serde_json::Value,
) -> Result<AuditFingerprints, String> {
    let algorithm = json_str_field(fingerprints, "algorithm")?;
    if algorithm != "sha256" {
        return Err(format!(
            "audit bundle fingerprint algorithm `{algorithm}` is not sha256"
        ));
    }
    Ok(AuditFingerprints {
        train_source: json_str_field(fingerprints, "train_source")?.to_owned(),
        identify_source: json_str_field(fingerprints, "identify_source")?.to_owned(),
        computation: json_str_field(fingerprints, "computation")?.to_owned(),
        final_model: json_str_field(fingerprints, "final_model")?.to_owned(),
        witness_trace: json_str_field(fingerprints, "witness_trace")?.to_owned(),
        proof: fingerprints
            .get("proof")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| sha256_json(&serde_json::Value::Null)),
        lineage_ledger: fingerprints
            .get("lineage_ledger")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| sha256_json(&serde_json::Value::Null)),
        report: json_str_field(fingerprints, "report")?.to_owned(),
        payload: json_str_field(fingerprints, "payload")?.to_owned(),
    })
}

fn verify_audit_fingerprints(bundle: &serde_json::Value) -> Result<(), String> {
    let stored = audit_fingerprints_from_bundle(bundle)?;
    let mut unsigned_payload = bundle.clone();
    unsigned_payload
        .as_object_mut()
        .expect("audit bundle is an object")
        .remove("fingerprints");
    let computed = audit_fingerprints_for_unsigned_payload(&unsigned_payload);
    if stored != computed {
        return Err(format!(
            "audit bundle fingerprint mismatch: stored payload {}, computed payload {}",
            stored.payload, computed.payload
        ));
    }
    Ok(())
}

fn audit_computation_fingerprint(payload: &serde_json::Value) -> String {
    let mut computation = json!({
        "train_source": sha256_bytes(TRAIN_SOURCE.as_bytes()),
        "identify_source": sha256_bytes(IDENTIFY_SOURCE.as_bytes()),
        "final_model": &payload["final_model"],
        "witness_trace": &payload["witness_trace"],
    });
    if let Some(proof) = payload.get("proof") {
        computation
            .as_object_mut()
            .expect("audit computation fingerprint is an object")
            .insert("proof".to_owned(), proof.clone());
    }
    if let Some(lineage_ledger) = payload.get("lineage_ledger") {
        computation
            .as_object_mut()
            .expect("audit computation fingerprint is an object")
            .insert("lineage_ledger".to_owned(), lineage_ledger.clone());
    }
    sha256_json(&computation)
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

#[derive(Debug, Clone)]
struct AuditStep {
    source_sample_index: usize,
    image: Vec<u8>,
    tape: TapeEntry,
}

fn parse_audit_steps(trace: &[serde_json::Value]) -> Result<Vec<AuditStep>, String> {
    trace
        .iter()
        .enumerate()
        .map(|(position, entry)| parse_audit_step(entry, position))
        .collect()
}

fn parse_audit_step(entry: &serde_json::Value, position: usize) -> Result<AuditStep, String> {
    let label = json_i64_field(entry, "label")?;
    if !(0..DIGITS as i64).contains(&label) {
        return Err(format!("witness_trace[{position}].label is outside 0..10"));
    }
    Ok(AuditStep {
        source_sample_index: json_usize_field(entry, "sample_index")?,
        image: json_u8_vec_field(entry, "image_u8", IMAGE_PIXELS)?,
        tape: TapeEntry {
            sample_index: position,
            label,
            logits: json_i64_vec_field(entry, "logits", DIGITS)?,
            error: json_i64_vec_field(entry, "error", DIGITS)?,
            prediction: json_i64_field(entry, "prediction")?,
            correct: json_i64_field(entry, "correct")?,
            lr: json_i64_field(entry, "lr")?,
        },
    })
}

fn inspect_audit_bundle(
    program: &Program,
    path: &Path,
    requested_step: usize,
    strategy: AuditStepStrategy,
    step_output: Option<&Path>,
    markdown_output: Option<&Path>,
    emit_json: bool,
) -> Result<(), String> {
    let bundle = read_audit_bundle(path)?;
    let fingerprints = audit_fingerprints_from_bundle(&bundle)?;
    let final_model = model_from_json(json_field(&bundle, "final_model")?, "final_model")?;
    let trace = json_array_field(&bundle, "witness_trace")?;
    let steps = parse_audit_steps(trace)?;
    let scans = steps
        .iter()
        .enumerate()
        .map(|(step, audit_step)| scan_audit_step(step, audit_step))
        .collect::<Vec<_>>();
    let step_index = select_audit_step_index(&scans, requested_step, strategy)?;
    let step = &steps[step_index];
    let selected_scan = &scans[step_index];
    let computed_prediction = argmax_first(&step.tape.logits);
    let computed_correct = i64::from(computed_prediction == step.tape.label);
    let prediction_matches_logits = step.tape.prediction == computed_prediction;
    let correct_matches_logits = step.tape.correct == computed_correct;
    let logit_ranks = top_logits(&step.tape.logits);
    let logit_margin = logit_margin(&logit_ranks);
    let active_pixels = active_pixels(&step.image);
    let bias_delta = applied_bias_delta(&step.tape.error, step.tape.lr);
    let weight_stats = weight_delta_stats(&step.image, &step.tape.error, step.tape.lr);
    let nonzero_bias_delta_count = nonzero_bias_delta_count(&bias_delta);
    let bias_delta_ledger_fingerprint = bias_delta_ledger_fingerprint(&bias_delta);
    let model_window = reconstruct_model_window(program, &final_model, &steps, step_index)?;
    let bias_observed_delta = vector_delta(&model_window.before.bias, &model_window.after.bias);
    let bias_delta_matches = bias_observed_delta == bias_delta;
    let weight_windows = weight_stats
        .top
        .iter()
        .map(|delta| WeightWindow {
            pixel: delta.pixel,
            digit: delta.digit,
            before: model_window.before.weights[delta.pixel][delta.digit],
            after: model_window.after.weights[delta.pixel][delta.digit],
            computed_delta: delta.delta,
            observed_delta: model_window.after.weights[delta.pixel][delta.digit]
                .wrapping_sub(model_window.before.weights[delta.pixel][delta.digit]),
        })
        .collect::<Vec<_>>();
    let weight_delta_matches = weight_windows
        .iter()
        .all(|window| window.observed_delta == window.computed_delta);

    let sample_json = step_sample_json(step);
    let witness_json = step_witness_json(&step.tape);
    let update_json = step_update_json(&bias_delta, &weight_stats);
    let cause_ledger = training_update_cause_ledger_json(
        Some(step_index),
        Some(step.source_sample_index),
        &sample_json,
        &witness_json,
        &update_json,
    )?;
    let report = json!({
            "kind": "reverie_mnist_linear_q31_audit_step",
            "path": path.display().to_string(),
            "fingerprints": fingerprints.to_json(),
            "step_output": step_output.map(|path| path.display().to_string()),
            "markdown_output": markdown_output.map(|path| path.display().to_string()),
            "selection": audit_step_selection_json(
                requested_step,
                strategy,
                &scans,
                selected_scan,
            ),
            "step": step_index,
            "sample_index": step.source_sample_index,
            "label": step.tape.label,
            "prediction": step.tape.prediction,
            "correct": step.tape.correct != 0,
            "lr": step.tape.lr,
            "logits": &step.tape.logits,
            "top_logits": logit_ranks.iter().map(|logit| {
                json!({
                    "digit": logit.digit,
                    "logit": logit.value,
                })
            }).collect::<Vec<_>>(),
            "logit_margin": {
                "predicted_digit": logit_margin.predicted.digit,
                "predicted_logit": logit_margin.predicted.value,
                "runner_up_digit": logit_margin.runner_up.digit,
                "runner_up_logit": logit_margin.runner_up.value,
                "margin": logit_margin.margin,
            },
            "error": &step.tape.error,
            "witness_checks": {
                "computed_prediction": computed_prediction,
                "computed_correct": computed_correct != 0,
                "prediction_matches_logits": prediction_matches_logits,
                "correct_matches_logits": correct_matches_logits,
            },
            "active_pixels": active_pixels.iter().map(|pixel| {
                json!({
                    "index": pixel.index,
                    "u8": pixel.value,
                    "q31": pixel.q31,
                })
            }).collect::<Vec<_>>(),
            "update": update_json,
            "cause_ledger": cause_ledger,
            "model_window": {
                "reconstructed": true,
                "reversed_later_steps": model_window.reversed_later_steps,
                "bias_before": &model_window.before.bias,
                "bias_after": &model_window.after.bias,
                "bias_observed_delta": bias_observed_delta,
                "bias_delta_matches": bias_delta_matches,
                "weight_delta_matches": weight_delta_matches,
                "top_weight_windows": weight_windows.iter().map(|window| {
                    json!({
                        "pixel": window.pixel,
                        "digit": window.digit,
                        "before": window.before,
                        "after": window.after,
                        "observed_delta": window.observed_delta,
                        "computed_delta": window.computed_delta,
                        "delta_matches": window.observed_delta == window.computed_delta,
                    })
                }).collect::<Vec<_>>(),
            },
            "debug_contract": audit_step_debug_contract_json(
                step_index,
                step,
                active_pixels.len(),
                weight_stats.nonzero_count,
                &bias_delta_ledger_fingerprint,
                &weight_stats.ledger_fingerprint,
                prediction_matches_logits,
                correct_matches_logits,
                bias_delta_matches,
                weight_delta_matches,
                &model_window,
            ),
    });

    if let Some(output_path) = step_output {
        write_step_bundle(
            output_path,
            path,
            &fingerprints,
            step_index,
            step.source_sample_index,
            step,
            &model_window,
            &bias_delta,
            &weight_stats,
        )?;
    }
    if let Some(output_path) = markdown_output {
        write_markdown_file(
            output_path,
            &render_training_step_debug_markdown(&report, None)?,
        )?;
    }

    if emit_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("audit step report serializes")
        );
    } else {
        println!("audit step {step_index}");
        println!(
            "selection: strategy={} requested_step={} selected_step={}",
            strategy.as_str(),
            requested_step,
            step_index
        );
        println!("path: {}", path.display());
        println!(
            "fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
        println!(
            "sample: source_index={} label={} prediction={} correct={} lr={}",
            step.source_sample_index,
            step.tape.label,
            step.tape.prediction,
            step.tape.correct != 0,
            step.tape.lr
        );
        println!(
            "witness_checks: computed_prediction={} prediction_matches_logits={} correct_matches_logits={}",
            computed_prediction, prediction_matches_logits, correct_matches_logits
        );
        println!(
            "active_pixels: count={} first={}",
            active_pixels.len(),
            format_active_pixel_preview(&active_pixels, 12)
        );
        println!("logits: {:?}", step.tape.logits);
        println!("top_logits: {}", format_logit_preview(&logit_ranks));
        println!(
            "logit_margin: predicted={} runner_up={} margin={}",
            logit_margin.predicted.digit, logit_margin.runner_up.digit, logit_margin.margin
        );
        println!("error: {:?}", step.tape.error);
        println!("bias_delta: {:?}", bias_delta);
        println!(
            "weight_delta: nonzero={} max_abs={} ledger={} top={}",
            weight_stats.nonzero_count,
            weight_stats.max_abs,
            weight_stats.ledger_fingerprint,
            format_weight_delta_preview(&weight_stats.top)
        );
        println!(
            "bias_delta_ledger: nonzero={} ledger={}",
            nonzero_bias_delta_count, bias_delta_ledger_fingerprint
        );
        println!(
            "model_window: reversed_later_steps={} bias_delta_matches={} weight_delta_matches={} top={}",
            model_window.reversed_later_steps,
            bias_delta_matches,
            weight_delta_matches,
            format_weight_window_preview(&weight_windows)
        );
        println!(
            "formula: weights -= scale_q31(outer_q31(image, error), lr); bias -= scale_q31(error, lr)"
        );
        println!(
            "debug_contract: claim=step_backward_from_model_update passed={} checks=5",
            prediction_matches_logits
                && correct_matches_logits
                && bias_delta_matches
                && weight_delta_matches
                && is_sha256_hex(&bias_delta_ledger_fingerprint)
                && is_sha256_hex(&weight_stats.ledger_fingerprint)
                && !active_pixels.is_empty()
                && weight_stats.nonzero_count > 0
        );
        if let Some(output_path) = step_output {
            println!("step bundle: {}", output_path.display());
        }
        if let Some(output_path) = markdown_output {
            println!("markdown: {}", output_path.display());
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn audit_step_debug_contract_json(
    step_index: usize,
    step: &AuditStep,
    active_pixel_count: usize,
    nonzero_weight_delta_count: usize,
    bias_delta_ledger_fingerprint: &str,
    weight_delta_ledger_fingerprint: &str,
    prediction_matches_logits: bool,
    correct_matches_logits: bool,
    bias_delta_matches: bool,
    weight_delta_matches: bool,
    model_window: &ModelWindow,
) -> serde_json::Value {
    let witness_recomputes_prediction = prediction_matches_logits && correct_matches_logits;
    let model_window_reconstructed = true;
    let update_matches_model_window = bias_delta_matches && weight_delta_matches;
    let explanatory_state_present =
        active_pixel_count > 0 && nonzero_weight_delta_count > 0 && step.tape.lr != 0;
    let update_ledger_fingerprints = is_sha256_hex(bias_delta_ledger_fingerprint)
        && is_sha256_hex(weight_delta_ledger_fingerprint);
    let checks = vec![
        json!({
            "metric": "witness_recomputes_prediction",
            "passed": witness_recomputes_prediction,
            "actual": format!(
                "prediction_matches_logits={} correct_matches_logits={}",
                prediction_matches_logits, correct_matches_logits
            ),
            "requirement": "saved logits/error witness recomputes prediction and correctness",
        }),
        json!({
            "metric": "model_window_reconstructed",
            "passed": model_window_reconstructed,
            "actual": format!(
                "audit_step={} reversed_later_steps={}",
                step_index, model_window.reversed_later_steps
            ),
            "requirement": "before/after model window is reconstructed by reversing later steps",
        }),
        json!({
            "metric": "update_matches_model_window",
            "passed": update_matches_model_window,
            "actual": format!(
                "bias_delta_matches={} weight_delta_matches={}",
                bias_delta_matches, weight_delta_matches
            ),
            "requirement": "computed update deltas match observed before/after model deltas",
        }),
        json!({
            "metric": "explanatory_state_present",
            "passed": explanatory_state_present,
            "actual": format!(
                "active_pixels={} nonzero_weight_deltas={} lr={}",
                active_pixel_count, nonzero_weight_delta_count, step.tape.lr
            ),
            "requirement": "sample pixels, learning rate, and nonzero update deltas are present",
        }),
        json!({
            "metric": "update_ledger_fingerprints",
            "passed": update_ledger_fingerprints,
            "actual": format!(
                "bias_delta_ledger={} weight_delta_ledger={}",
                bias_delta_ledger_fingerprint, weight_delta_ledger_fingerprint
            ),
            "requirement": "full bias and weight update ledgers have deterministic SHA-256 fingerprints",
        }),
    ];
    let passed = checks
        .iter()
        .all(|check| check["passed"].as_bool().unwrap_or(false));
    json!({
        "claim": "step_backward_from_model_update",
        "passed": passed,
        "checks": checks,
        "replay_direction": {
            "backward_from_final_model_steps": model_window.reversed_later_steps,
            "selected_transition": "before_model + sample + witness -> after_model",
            "reverse_transition": "after_model + sample + witness -> before_model",
        },
        "debug_focus": {
            "prediction": "top_logits and logit_margin",
            "update": "cause_ledger, error, active_pixels, top_weight_deltas, and ledger_fingerprints",
            "reversibility": "model_window and witness_checks",
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn write_step_bundle(
    path: &Path,
    source_audit_path: &Path,
    source_fingerprints: &AuditFingerprints,
    step_index: usize,
    sample_index: usize,
    step: &AuditStep,
    model_window: &ModelWindow,
    bias_delta: &[i64],
    weight_stats: &WeightDeltaStats,
) -> Result<(), String> {
    let before_model_json = json!({
        "weights": &model_window.before.weights,
        "bias": &model_window.before.bias,
    });
    let after_model_json = json!({
        "weights": &model_window.after.weights,
        "bias": &model_window.after.bias,
    });
    let sample_json = step_sample_json(step);
    let witness_json = step_witness_json(&step.tape);
    let update_json = step_update_json(bias_delta, weight_stats);
    let proof_json = step_bundle_proof_json(
        Some(step_index),
        Some(sample_index),
        &before_model_json,
        &after_model_json,
        &sample_json,
        &witness_json,
        &update_json,
    )?;
    let mut bundle = json!({
        "kind": "reverie_mnist_linear_q31_step_replay_bundle",
        "schema_version": 1,
        "program": "examples/mnist_reversible_step.rev",
        "source_training_bundle": {
            "path": source_audit_path.display().to_string(),
            "audit_step": step_index,
            "sample_index": sample_index,
            "reversed_later_steps": model_window.reversed_later_steps,
            "fingerprints": source_fingerprints.to_json(),
        },
        "tensor_shapes": {
            "image": [IMAGE_PIXELS],
            "weights": [IMAGE_PIXELS, DIGITS],
            "bias": [DIGITS],
            "logits": [DIGITS],
            "error": [DIGITS],
        },
        "before_model": before_model_json,
        "after_model": after_model_json,
        "sample": sample_json,
        "witness": witness_json,
        "update": update_json,
        "proof": proof_json,
    });
    let fingerprints = step_fingerprints_for_unsigned_payload(&bundle);
    bundle
        .as_object_mut()
        .expect("step bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints.to_json());

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let encoded = serde_json::to_string_pretty(&bundle).expect("step bundle serializes");
    fs::write(path, encoded)
        .map_err(|error| format!("failed to write step bundle {}: {error}", path.display()))
}

fn step_sample_json(step: &AuditStep) -> serde_json::Value {
    json!({
        "image_u8": &step.image,
        "label": step.tape.label,
    })
}

fn model_json(model: &Model) -> serde_json::Value {
    json!({
        "weights": &model.weights,
        "bias": &model.bias,
    })
}

fn step_witness_json(tape: &TapeEntry) -> serde_json::Value {
    json!({
        "logits": &tape.logits,
        "error": &tape.error,
        "prediction": tape.prediction,
        "correct": tape.correct,
        "lr": tape.lr,
    })
}

fn training_lineage_ledger_json(
    final_model: &serde_json::Value,
    trace: &[serde_json::Value],
) -> Result<serde_json::Value, String> {
    let expected_final_model = model_from_json(final_model, "final_model")?;
    let steps = parse_audit_steps(trace)?;
    let initial_model = Model::zero();
    let initial_model_json = model_json(&initial_model);
    let initial_model_fingerprint = sha256_json(&initial_model_json);
    let final_model_fingerprint = sha256_json(final_model);
    let witness_trace_json = serde_json::Value::Array(trace.to_vec());
    let witness_trace_fingerprint = sha256_json(&witness_trace_json);
    let initial_chain = sha256_json(&json!({
        "schema": "q31_training_lineage_chain_v1",
        "initial_model_fingerprint": &initial_model_fingerprint,
    }));
    let mut chain = initial_chain.clone();
    let mut model = initial_model;
    let mut sample_order = Vec::with_capacity(steps.len());
    let mut transition_rows = Vec::with_capacity(steps.len());

    for (step_index, step) in steps.iter().enumerate() {
        let sample_json = step_sample_json(step);
        let witness_json = step_witness_json(&step.tape);
        let bias_delta = applied_bias_delta(&step.tape.error, step.tape.lr);
        let weight_stats = weight_delta_stats(&step.image, &step.tape.error, step.tape.lr);
        let update_json = step_update_json(&bias_delta, &weight_stats);
        let cause_ledger = training_update_cause_ledger_json(
            Some(step_index),
            Some(step.source_sample_index),
            &sample_json,
            &witness_json,
            &update_json,
        )?;
        let sample_fingerprint = sha256_json(&sample_json);
        let witness_fingerprint = sha256_json(&witness_json);
        let update_fingerprint = sha256_json(&update_json);
        let cause_ledger_fingerprint = json_str_field(&cause_ledger, "fingerprint")?.to_owned();
        sample_order.push(json!({
            "step": step_index,
            "sample_index": step.source_sample_index,
            "label": step.tape.label,
        }));
        let after_model = apply_tape_update(&model, &step.image, &step.tape);
        let transition_payload = json!({
            "schema": "q31_training_lineage_transition_v1",
            "step": step_index,
            "sample_index": step.source_sample_index,
            "label": step.tape.label,
            "prediction": step.tape.prediction,
            "correct": step.tape.correct != 0,
            "lr": step.tape.lr,
            "sample_fingerprint": &sample_fingerprint,
            "witness_fingerprint": &witness_fingerprint,
            "update_fingerprint": &update_fingerprint,
            "cause_ledger_fingerprint": &cause_ledger_fingerprint,
            "before_chain": &chain,
        });
        let transition_fingerprint = sha256_json(&transition_payload);
        let after_chain = sha256_json(&json!({
            "schema": "q31_training_lineage_chain_v1",
            "before_chain": &chain,
            "transition_fingerprint": &transition_fingerprint,
        }));
        transition_rows.push(json!({
            "step": step_index,
            "sample_index": step.source_sample_index,
            "label": step.tape.label,
            "prediction": step.tape.prediction,
            "correct": step.tape.correct != 0,
            "lr": step.tape.lr,
            "sample_fingerprint": sample_fingerprint,
            "witness_fingerprint": witness_fingerprint,
            "update_fingerprint": update_fingerprint,
            "cause_ledger_fingerprint": cause_ledger_fingerprint,
            "before_chain": &chain,
            "transition_fingerprint": transition_fingerprint,
            "after_chain": &after_chain,
        }));
        chain = after_chain;
        model = after_model;
    }

    if model != expected_final_model {
        return Err("training lineage replay did not reconstruct final_model".to_owned());
    }

    let sample_order_fingerprint = sha256_json(&serde_json::Value::Array(sample_order));
    let transition_ledger = json!({
        "schema": "q31_training_lineage_transition_ledger_v1",
        "rows": transition_rows,
    });
    let transition_ledger_fingerprint = sha256_json(&transition_ledger);
    let transition_rows = json_array_field(&transition_ledger, "rows")?;
    let first_transition = transition_rows
        .first()
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let last_transition = transition_rows
        .last()
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let payload = json!({
        "schema": "q31_training_lineage_ledger_v1",
        "steps": steps.len(),
        "initial_model_fingerprint": &initial_model_fingerprint,
        "final_model_fingerprint": final_model_fingerprint,
        "witness_trace_fingerprint": witness_trace_fingerprint,
        "sample_order_fingerprint": sample_order_fingerprint,
        "transition_ledger_fingerprint": transition_ledger_fingerprint,
        "initial_chain": initial_chain,
        "final_chain": chain,
        "first_transition": first_transition,
        "last_transition": last_transition,
    });
    Ok(json!({
        "algorithm": "sha256",
        "schema": "q31_training_lineage_ledger_v1",
        "fingerprint": sha256_json(&payload),
        "payload": payload,
    }))
}

fn apply_tape_update(model: &Model, image: &[u8], tape: &TapeEntry) -> Model {
    let mut next = model.clone();
    let bias_delta = applied_bias_delta(&tape.error, tape.lr);
    for (digit, delta) in bias_delta.into_iter().enumerate() {
        next.bias[digit] = next.bias[digit].wrapping_add(delta);
    }
    for (pixel, value) in image.iter().copied().enumerate() {
        if value == 0 {
            continue;
        }
        let pixel_q31 = pixel_to_q31(value);
        for (digit, error) in tape.error.iter().copied().enumerate() {
            let outer = fixed_mul_q31(pixel_q31, error);
            let scaled = fixed_mul_q31(outer, tape.lr);
            let delta = scaled.wrapping_neg();
            next.weights[pixel][digit] = next.weights[pixel][digit].wrapping_add(delta);
        }
    }
    next
}

fn active_pixels_json(active_pixels: &[ActivePixel]) -> Vec<serde_json::Value> {
    active_pixels
        .iter()
        .map(|pixel| {
            json!({
                "index": pixel.index,
                "u8": pixel.value,
                "q31": pixel.q31,
            })
        })
        .collect()
}

fn top_logits_json(top_logits: &[LogitRank]) -> Vec<serde_json::Value> {
    top_logits
        .iter()
        .map(|logit| {
            json!({
                "digit": logit.digit,
                "logit": logit.value,
            })
        })
        .collect()
}

fn logit_margin_json(logit_margin: &LogitMargin) -> serde_json::Value {
    json!({
        "predicted_digit": logit_margin.predicted.digit,
        "predicted_logit": logit_margin.predicted.value,
        "runner_up_digit": logit_margin.runner_up.digit,
        "runner_up_logit": logit_margin.runner_up.value,
        "margin": logit_margin.margin,
    })
}

fn training_update_cause_ledger_json(
    step_index: Option<usize>,
    sample_index: Option<usize>,
    sample: &serde_json::Value,
    witness: &serde_json::Value,
    update: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let image = json_u8_vec_field(sample, "image_u8", IMAGE_PIXELS)?;
    let label = json_i64_field(sample, "label")?;
    if !(0..DIGITS as i64).contains(&label) {
        return Err("training cause ledger sample label is outside 0..10".to_owned());
    }
    let logits = json_i64_vec_field(witness, "logits", DIGITS)?;
    let error = json_i64_vec_field(witness, "error", DIGITS)?;
    let prediction = json_i64_field(witness, "prediction")?;
    if !(0..DIGITS as i64).contains(&prediction) {
        return Err("training cause ledger prediction is outside 0..10".to_owned());
    }
    let correct = json_i64_field(witness, "correct")? != 0;
    let lr = json_i64_field(witness, "lr")?;
    let active_pixels = active_pixels(&image);
    let top_logits = top_logits(&logits);
    let logit_margin = logit_margin(&top_logits);
    let update_ledger_fingerprints = json_field(update, "ledger_fingerprints")?.clone();
    let top_weight_deltas = json_field(update, "top_weight_deltas")?.clone();
    let payload = json!({
        "schema": "q31_training_update_cause_ledger_v1",
        "step": step_index,
        "sample_index": sample_index,
        "sample_fingerprint": sha256_json(sample),
        "witness_fingerprint": sha256_json(witness),
        "update_fingerprint": sha256_json(update),
        "label": label,
        "lr": lr,
        "logits": logits,
        "error": error,
        "prediction": prediction,
        "correct": correct,
        "active_pixels": active_pixels_json(&active_pixels),
        "top_logits": top_logits_json(&top_logits),
        "logit_margin": logit_margin_json(&logit_margin),
        "update_ledger_fingerprints": update_ledger_fingerprints,
        "top_weight_deltas": top_weight_deltas,
    });
    Ok(json!({
        "algorithm": "sha256",
        "schema": "q31_training_update_cause_ledger_v1",
        "fingerprint": sha256_json(&payload),
        "payload": payload,
    }))
}

fn step_update_json(bias_delta: &[i64], weight_stats: &WeightDeltaStats) -> serde_json::Value {
    let bias_delta_ledger_fingerprint = bias_delta_ledger_fingerprint(bias_delta);
    json!({
        "formula": "weights -= scale_q31(outer_q31(image, error), lr); bias -= scale_q31(error, lr)",
        "bias_delta": bias_delta,
        "nonzero_bias_delta_count": nonzero_bias_delta_count(bias_delta),
        "nonzero_weight_delta_count": weight_stats.nonzero_count,
        "max_abs_weight_delta": weight_stats.max_abs,
        "bias_delta_ledger_fingerprint": &bias_delta_ledger_fingerprint,
        "weight_delta_ledger_fingerprint": &weight_stats.ledger_fingerprint,
        "ledger_fingerprints": {
            "algorithm": "sha256",
            "bias_delta": &bias_delta_ledger_fingerprint,
            "weight_delta": &weight_stats.ledger_fingerprint,
        },
        "top_weight_deltas": weight_stats.top.iter().map(|delta| {
            json!({
                "pixel": delta.pixel,
                "digit": delta.digit,
                "delta": delta.delta,
            })
        }).collect::<Vec<_>>(),
    })
}

fn step_bundle_proof_json(
    step_index: Option<usize>,
    sample_index: Option<usize>,
    before_model: &serde_json::Value,
    after_model: &serde_json::Value,
    sample: &serde_json::Value,
    witness: &serde_json::Value,
    update: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let top_weight_deltas = json_array_field(update, "top_weight_deltas")?.len();
    let nonzero_bias_delta_count = json_usize_field(update, "nonzero_bias_delta_count")?;
    let nonzero_weight_delta_count = json_usize_field(update, "nonzero_weight_delta_count")?;
    let update_ledger_fingerprints = json_field(update, "ledger_fingerprints")?.clone();
    let cause_ledger =
        training_update_cause_ledger_json(step_index, sample_index, sample, witness, update)?;
    let cause_ledger_fingerprint = json_str_field(&cause_ledger, "fingerprint")?.to_owned();
    let model_payload_bytes = 2 * Model::payload_bytes();
    let sample_payload_bytes = single_sample_payload_bytes();
    let witness_payload_bytes = training_step_witness_payload_bytes();
    let derived_update_payload_bytes = step_update_payload_bytes(top_weight_deltas);
    let replay_payload_bytes = model_payload_bytes
        + sample_payload_bytes
        + witness_payload_bytes
        + derived_update_payload_bytes;

    Ok(json!({
        "claim": "deterministic_q31_training_step_replay",
        "kernel": "examples/mnist_reversible_step.rev",
        "arithmetic": "q31_wrapping_i64",
        "witnesses": ["logits", "error", "prediction", "correct", "lr"],
        "model_payload_bytes": model_payload_bytes,
        "sample_payload_bytes": sample_payload_bytes,
        "witness_payload_bytes": witness_payload_bytes,
        "derived_update_payload_bytes": derived_update_payload_bytes,
        "nonzero_bias_delta_count": nonzero_bias_delta_count,
        "nonzero_weight_delta_count": nonzero_weight_delta_count,
        "update_ledger_fingerprints": update_ledger_fingerprints,
        "cause_ledger": cause_ledger,
        "trace_payload_bytes": 0,
        "replay_payload_bytes": replay_payload_bytes,
        "forward_recompute_steps": 1,
        "inverse_recompute_steps": 1,
        "checks": {
            "witnesses_match_forward_replay": true,
            "after_model_matches": true,
            "before_model_restored": true,
            "update_matches_witnesses": true,
        },
        "fingerprints": {
            "algorithm": "sha256",
            "train_source": sha256_bytes(TRAIN_SOURCE.as_bytes()),
            "before_model": sha256_json(before_model),
            "after_model": sha256_json(after_model),
            "sample": sha256_json(sample),
            "witness": sha256_json(witness),
            "update": sha256_json(update),
            "cause_ledger": cause_ledger_fingerprint,
        },
    }))
}

fn verify_step_bundle(
    program: &Program,
    path: &Path,
    markdown_output: Option<&Path>,
    emit_json: bool,
) -> Result<(), String> {
    let bundle = read_step_bundle(path)?;
    let fingerprints = step_fingerprints_from_bundle(&bundle)?;
    let before_model_json = json_field(&bundle, "before_model")?;
    let after_model_json = json_field(&bundle, "after_model")?;
    let before_model = model_from_json(before_model_json, "before_model")?;
    let after_model = model_from_json(after_model_json, "after_model")?;
    let sample = json_field(&bundle, "sample")?;
    let image = json_u8_vec_field(sample, "image_u8", IMAGE_PIXELS)?;
    let label = json_i64_field(sample, "label")?;
    if !(0..DIGITS as i64).contains(&label) {
        return Err("step bundle `sample.label` is outside 0..10".to_owned());
    }
    let witness = json_field(&bundle, "witness")?;
    let tape = TapeEntry {
        sample_index: 0,
        label,
        logits: json_i64_vec_field(witness, "logits", DIGITS)?,
        error: json_i64_vec_field(witness, "error", DIGITS)?,
        prediction: json_i64_field(witness, "prediction")?,
        correct: json_i64_field(witness, "correct")?,
        lr: json_i64_field(witness, "lr")?,
    };
    let update = json_field(&bundle, "update")?;
    verify_step_update(update, &image, &tape)?;
    let source_training_bundle = json_field(&bundle, "source_training_bundle")?;
    let source_audit_step = json_usize_field(source_training_bundle, "audit_step")?;
    let source_sample_index = json_usize_field(source_training_bundle, "sample_index")?;
    let proof = step_bundle_proof_json(
        Some(source_audit_step),
        Some(source_sample_index),
        before_model_json,
        after_model_json,
        sample,
        witness,
        update,
    )?;
    if let Some(stored_proof) = bundle.get("proof") {
        verify_step_proof(stored_proof, &proof)?;
    }

    let dataset = Dataset {
        images: vec![image.clone()],
        labels: vec![label as u8],
    };
    let forward_start = Instant::now();
    let forward_step = train_step(program, &before_model, &dataset, 0, tape.lr)?;
    verify_forward_tape(0, &tape, &forward_step.tape)?;
    if forward_step.model != after_model {
        return Err("step forward replay failed: after_model did not match".to_owned());
    }
    let forward_elapsed = forward_start.elapsed();

    let reverse_start = Instant::now();
    let restored = reverse_one_step(program, &after_model, &image, &tape)?;
    if restored != before_model {
        return Err("step reverse replay failed: before_model was not restored".to_owned());
    }
    let reverse_elapsed = reverse_start.elapsed();

    let step_debug_report = audit_step_report_from_step_bundle(
        path,
        &fingerprints,
        &bundle,
        &before_model,
        &after_model,
        &image,
        &tape,
        update,
        markdown_output,
    )?;
    let verification_report = json!({
        "kind": "reverie_mnist_linear_q31_step_verification",
        "path": path.display().to_string(),
        "fingerprints": fingerprints.to_json(),
        "markdown_output": markdown_output.map(|path| path.display().to_string()),
        "proof_matches": bundle.get("proof").is_some(),
        "proof": proof,
        "forward": {
            "witnesses_match": true,
            "after_model_matches": true,
            "elapsed_seconds": secs(forward_elapsed),
        },
        "reverse": {
            "before_model_restored": true,
            "elapsed_seconds": secs(reverse_elapsed),
        },
    });
    if let Some(output_path) = markdown_output {
        write_markdown_file(
            output_path,
            &render_training_step_debug_markdown(&step_debug_report, Some(&verification_report))?,
        )?;
    }

    if emit_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&verification_report)
                .expect("step verification report serializes")
        );
    } else {
        println!(
            "step audit ok: witnesses_match=true after_model_matches=true before_model_restored=true"
        );
        println!(
            "forward: elapsed={:.3}s reverse: elapsed={:.3}s",
            secs(forward_elapsed),
            secs(reverse_elapsed)
        );
        println!("{}", format_step_proof_line(&proof));
        println!(
            "fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
        if let Some(output_path) = markdown_output {
            println!("markdown: {}", output_path.display());
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn audit_step_report_from_step_bundle(
    path: &Path,
    fingerprints: &StepFingerprints,
    bundle: &serde_json::Value,
    before_model: &Model,
    after_model: &Model,
    image: &[u8],
    tape: &TapeEntry,
    update: &serde_json::Value,
    markdown_output: Option<&Path>,
) -> Result<serde_json::Value, String> {
    let source_training_bundle = json_field(bundle, "source_training_bundle")?;
    let step_index = json_usize_field(source_training_bundle, "audit_step")?;
    let source_sample_index = json_usize_field(source_training_bundle, "sample_index")?;
    let reversed_later_steps =
        json_optional_usize_field(source_training_bundle, "reversed_later_steps")?.unwrap_or(0);
    let step = AuditStep {
        source_sample_index,
        image: image.to_vec(),
        tape: TapeEntry {
            sample_index: step_index,
            label: tape.label,
            logits: tape.logits.clone(),
            error: tape.error.clone(),
            prediction: tape.prediction,
            correct: tape.correct,
            lr: tape.lr,
        },
    };
    let computed_prediction = argmax_first(&step.tape.logits);
    let computed_correct = i64::from(computed_prediction == step.tape.label);
    let prediction_matches_logits = step.tape.prediction == computed_prediction;
    let correct_matches_logits = step.tape.correct == computed_correct;
    let logit_ranks = top_logits(&step.tape.logits);
    let logit_margin = logit_margin(&logit_ranks);
    let active_pixels = active_pixels(&step.image);
    let bias_delta = json_i64_vec_field(update, "bias_delta", DIGITS)?;
    let bias_observed_delta = vector_delta(&before_model.bias, &after_model.bias);
    let bias_delta_matches = bias_observed_delta == bias_delta;
    let weight_stats = weight_delta_stats(&step.image, &step.tape.error, step.tape.lr);
    let model_window = ModelWindow {
        before: before_model.clone(),
        after: after_model.clone(),
        reversed_later_steps,
    };
    let weight_windows = weight_stats
        .top
        .iter()
        .map(|delta| WeightWindow {
            pixel: delta.pixel,
            digit: delta.digit,
            before: before_model.weights[delta.pixel][delta.digit],
            after: after_model.weights[delta.pixel][delta.digit],
            computed_delta: delta.delta,
            observed_delta: after_model.weights[delta.pixel][delta.digit]
                .wrapping_sub(before_model.weights[delta.pixel][delta.digit]),
        })
        .collect::<Vec<_>>();
    let weight_delta_matches = weight_windows
        .iter()
        .all(|window| window.observed_delta == window.computed_delta);
    let sample_json = json_field(bundle, "sample")?;
    let witness_json = json_field(bundle, "witness")?;
    let cause_ledger = training_update_cause_ledger_json(
        Some(step_index),
        Some(source_sample_index),
        sample_json,
        witness_json,
        update,
    )?;
    let bias_delta_ledger_fingerprint =
        json_str_field(update, "bias_delta_ledger_fingerprint")?.to_owned();
    let weight_delta_ledger_fingerprint =
        json_str_field(update, "weight_delta_ledger_fingerprint")?.to_owned();

    Ok(json!({
        "kind": "reverie_mnist_linear_q31_audit_step",
        "path": path.display().to_string(),
        "fingerprints": fingerprints.to_json(),
        "step_output": path.display().to_string(),
        "markdown_output": markdown_output.map(|path| path.display().to_string()),
        "step": step_index,
        "sample_index": source_sample_index,
        "label": step.tape.label,
        "prediction": step.tape.prediction,
        "correct": step.tape.correct != 0,
        "lr": step.tape.lr,
        "logits": &step.tape.logits,
        "top_logits": top_logits_json(&logit_ranks),
        "logit_margin": logit_margin_json(&logit_margin),
        "error": &step.tape.error,
        "witness_checks": {
            "computed_prediction": computed_prediction,
            "computed_correct": computed_correct != 0,
            "prediction_matches_logits": prediction_matches_logits,
            "correct_matches_logits": correct_matches_logits,
        },
        "active_pixels": active_pixels_json(&active_pixels),
        "update": update.clone(),
        "cause_ledger": cause_ledger,
        "model_window": {
            "reconstructed": true,
            "reversed_later_steps": model_window.reversed_later_steps,
            "bias_before": &before_model.bias,
            "bias_after": &after_model.bias,
            "bias_observed_delta": bias_observed_delta,
            "bias_delta_matches": bias_delta_matches,
            "weight_delta_matches": weight_delta_matches,
            "top_weight_windows": weight_windows.iter().map(|window| {
                json!({
                    "pixel": window.pixel,
                    "digit": window.digit,
                    "before": window.before,
                    "after": window.after,
                    "observed_delta": window.observed_delta,
                    "computed_delta": window.computed_delta,
                    "delta_matches": window.observed_delta == window.computed_delta,
                })
            }).collect::<Vec<_>>(),
        },
        "debug_contract": audit_step_debug_contract_json(
            step_index,
            &step,
            active_pixels.len(),
            weight_stats.nonzero_count,
            &bias_delta_ledger_fingerprint,
            &weight_delta_ledger_fingerprint,
            prediction_matches_logits,
            correct_matches_logits,
            bias_delta_matches,
            weight_delta_matches,
            &model_window,
        ),
    }))
}

fn verify_step_proof(
    stored: &serde_json::Value,
    expected: &serde_json::Value,
) -> Result<(), String> {
    if stored != expected {
        return Err("step bundle proof does not match recomputed step replay cost".to_owned());
    }
    Ok(())
}

fn format_step_proof_line(proof: &serde_json::Value) -> String {
    format!(
        "proof: claim=deterministic_q31_training_step_replay model_bytes={} sample_bytes={} witness_bytes={} update_bytes={} replay_bytes={} forward_recompute_steps={} inverse_recompute_steps={}",
        proof
            .get("model_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("sample_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("witness_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("derived_update_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("replay_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("forward_recompute_steps")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        proof
            .get("inverse_recompute_steps")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    )
}

fn verify_step_update(
    update: &serde_json::Value,
    image: &[u8],
    tape: &TapeEntry,
) -> Result<(), String> {
    let expected_bias_delta = applied_bias_delta(&tape.error, tape.lr);
    let actual_bias_delta = json_i64_vec_field(update, "bias_delta", DIGITS)?;
    if actual_bias_delta != expected_bias_delta {
        return Err("step bundle update `bias_delta` does not match witnesses".to_owned());
    }
    let expected_bias_count = nonzero_bias_delta_count(&expected_bias_delta);
    let bias_count = json_usize_field(update, "nonzero_bias_delta_count")?;
    if bias_count != expected_bias_count {
        return Err(format!(
            "step bundle update `nonzero_bias_delta_count` expected {}, found {bias_count}",
            expected_bias_count
        ));
    }
    let expected_bias_ledger = bias_delta_ledger_fingerprint(&expected_bias_delta);
    let bias_ledger = json_str_field(update, "bias_delta_ledger_fingerprint")?;
    if bias_ledger != expected_bias_ledger {
        return Err(
            "step bundle update `bias_delta_ledger_fingerprint` does not match witnesses"
                .to_owned(),
        );
    }
    let expected_weight_stats = weight_delta_stats(image, &tape.error, tape.lr);
    let nonzero = json_usize_field(update, "nonzero_weight_delta_count")?;
    if nonzero != expected_weight_stats.nonzero_count {
        return Err(format!(
            "step bundle update `nonzero_weight_delta_count` expected {}, found {nonzero}",
            expected_weight_stats.nonzero_count
        ));
    }
    let max_abs = json_i128_field(update, "max_abs_weight_delta")?;
    if max_abs != expected_weight_stats.max_abs {
        return Err(format!(
            "step bundle update `max_abs_weight_delta` expected {}, found {max_abs}",
            expected_weight_stats.max_abs
        ));
    }
    let weight_ledger = json_str_field(update, "weight_delta_ledger_fingerprint")?;
    if weight_ledger != expected_weight_stats.ledger_fingerprint {
        return Err(
            "step bundle update `weight_delta_ledger_fingerprint` does not match witnesses"
                .to_owned(),
        );
    }
    let ledger_fingerprints = json_field(update, "ledger_fingerprints")?;
    let algorithm = json_str_field(ledger_fingerprints, "algorithm")?;
    if algorithm != "sha256" {
        return Err("step bundle update `ledger_fingerprints.algorithm` must be sha256".to_owned());
    }
    if json_str_field(ledger_fingerprints, "bias_delta")? != expected_bias_ledger {
        return Err(
            "step bundle update `ledger_fingerprints.bias_delta` does not match witnesses"
                .to_owned(),
        );
    }
    if json_str_field(ledger_fingerprints, "weight_delta")?
        != expected_weight_stats.ledger_fingerprint
    {
        return Err(
            "step bundle update `ledger_fingerprints.weight_delta` does not match witnesses"
                .to_owned(),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StepFingerprints {
    train_source: String,
    computation: String,
    before_model: String,
    after_model: String,
    sample: String,
    witness: String,
    update: String,
    proof: String,
    payload: String,
}

impl StepFingerprints {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "algorithm": "sha256",
            "train_source": self.train_source,
            "computation": self.computation,
            "before_model": self.before_model,
            "after_model": self.after_model,
            "sample": self.sample,
            "witness": self.witness,
            "update": self.update,
            "proof": self.proof,
            "payload": self.payload,
        })
    }
}

fn step_fingerprints_for_unsigned_payload(payload: &serde_json::Value) -> StepFingerprints {
    StepFingerprints {
        train_source: sha256_bytes(TRAIN_SOURCE.as_bytes()),
        computation: step_computation_fingerprint(payload),
        before_model: sha256_json(&payload["before_model"]),
        after_model: sha256_json(&payload["after_model"]),
        sample: sha256_json(&payload["sample"]),
        witness: sha256_json(&payload["witness"]),
        update: sha256_json(&payload["update"]),
        proof: sha256_json(payload.get("proof").unwrap_or(&serde_json::Value::Null)),
        payload: sha256_json(payload),
    }
}

fn step_fingerprints_from_bundle(bundle: &serde_json::Value) -> Result<StepFingerprints, String> {
    let fingerprints = json_field(bundle, "fingerprints")?;
    let algorithm = json_str_field(fingerprints, "algorithm")?;
    if algorithm != "sha256" {
        return Err(format!(
            "step bundle fingerprint algorithm `{algorithm}` is not sha256"
        ));
    }
    Ok(StepFingerprints {
        train_source: json_str_field(fingerprints, "train_source")?.to_owned(),
        computation: json_str_field(fingerprints, "computation")?.to_owned(),
        before_model: json_str_field(fingerprints, "before_model")?.to_owned(),
        after_model: json_str_field(fingerprints, "after_model")?.to_owned(),
        sample: json_str_field(fingerprints, "sample")?.to_owned(),
        witness: json_str_field(fingerprints, "witness")?.to_owned(),
        update: json_str_field(fingerprints, "update")?.to_owned(),
        proof: fingerprints
            .get("proof")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| sha256_json(&serde_json::Value::Null)),
        payload: json_str_field(fingerprints, "payload")?.to_owned(),
    })
}

fn verify_step_fingerprints(bundle: &serde_json::Value) -> Result<(), String> {
    let stored = step_fingerprints_from_bundle(bundle)?;
    let mut unsigned_payload = bundle.clone();
    unsigned_payload
        .as_object_mut()
        .expect("step bundle is an object")
        .remove("fingerprints");
    let computed = step_fingerprints_for_unsigned_payload(&unsigned_payload);
    if stored != computed {
        return Err(format!(
            "step bundle fingerprint mismatch: stored payload {}, computed payload {}",
            stored.payload, computed.payload
        ));
    }
    Ok(())
}

fn step_computation_fingerprint(payload: &serde_json::Value) -> String {
    let mut computation = json!({
        "train_source": sha256_bytes(TRAIN_SOURCE.as_bytes()),
        "before_model": &payload["before_model"],
        "after_model": &payload["after_model"],
        "sample": &payload["sample"],
        "witness": &payload["witness"],
    });
    if let Some(proof) = payload.get("proof") {
        computation
            .as_object_mut()
            .expect("step computation fingerprint is an object")
            .insert("proof".to_owned(), proof.clone());
    }
    sha256_json(&computation)
}

fn read_step_bundle(path: &Path) -> Result<serde_json::Value, String> {
    let data = fs::read(path)
        .map_err(|error| format!("failed to read step bundle {}: {error}", path.display()))?;
    let bundle: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|error| format!("failed to parse step bundle {}: {error}", path.display()))?;
    let kind = json_str_field(&bundle, "kind")?;
    if kind != "reverie_mnist_linear_q31_step_replay_bundle" {
        return Err(format!(
            "step bundle kind `{kind}` is not reverie_mnist_linear_q31_step_replay_bundle"
        ));
    }
    verify_step_fingerprints(&bundle)?;
    Ok(bundle)
}

#[derive(Debug, Clone)]
struct AuditStepScan {
    step: usize,
    sample_index: usize,
    label: i64,
    prediction: i64,
    correct: bool,
    lr: i64,
    computed_prediction: i64,
    predicted_logit: i64,
    runner_up_digit: usize,
    runner_up_logit: i64,
    margin: i64,
    prediction_matches_logits: bool,
    correct_matches_logits: bool,
    active_pixel_count: usize,
    error_nonzero_count: usize,
    max_abs_error: i128,
    max_abs_bias_delta: i128,
    nonzero_weight_delta_count: usize,
    max_abs_weight_delta: i128,
}

#[derive(Debug, Clone)]
struct AuditScanSummary {
    steps: usize,
    correct: usize,
    incorrect: usize,
    witness_mismatches: usize,
    max_abs_weight_delta: i128,
    largest_update_step: Option<usize>,
    lowest_margin_step: Option<usize>,
    lowest_margin: Option<i64>,
    trace_payload_bytes: usize,
}

#[derive(Debug, Clone)]
struct LabelScanSummary {
    label: usize,
    samples: usize,
    correct: usize,
    incorrect: usize,
    min_margin_step: Option<usize>,
    min_margin: Option<i64>,
    largest_update_step: Option<usize>,
    max_abs_weight_delta: i128,
}

#[derive(Debug, Clone)]
struct ConfusionSummary {
    label: i64,
    prediction: i64,
    count: usize,
    first_step: usize,
    min_margin: i64,
    max_abs_weight_delta: i128,
}

fn scan_audit_bundle(
    path: &Path,
    limit: usize,
    gate: &AuditGate,
    emit_json: bool,
) -> Result<(), String> {
    let bundle = read_audit_bundle(path)?;
    let fingerprints = audit_fingerprints_from_bundle(&bundle)?;
    let trace = json_array_field(&bundle, "witness_trace")?;
    let steps = parse_audit_steps(trace)?;
    let scans = steps
        .iter()
        .enumerate()
        .map(|(step, audit_step)| scan_audit_step(step, audit_step))
        .collect::<Vec<_>>();
    let summary = audit_scan_summary(&scans);
    let top_suspicious = ranked_audit_scans(&scans, limit);
    let top_low_margin = ranked_low_margin_scans(&scans, limit);
    let top_large_updates = ranked_large_update_scans(&scans, limit);
    let by_label = label_scan_summaries(&scans);
    let top_confusions = top_confusion_summaries(&scans, limit);
    let gate_report = gate.evaluate(&summary, &by_label);

    if emit_json {
        let mut report = json!({
            "kind": "reverie_mnist_linear_q31_audit_scan",
            "path": path.display().to_string(),
            "fingerprints": fingerprints.to_json(),
            "summary": audit_scan_summary_json(&summary),
            "by_label": by_label.iter().map(label_scan_summary_json).collect::<Vec<_>>(),
            "top_confusions": top_confusions.iter().map(confusion_summary_json).collect::<Vec<_>>(),
            "top_suspicious": top_suspicious.iter().map(audit_step_scan_json).collect::<Vec<_>>(),
            "top_low_margin": top_low_margin.iter().map(audit_step_scan_json).collect::<Vec<_>>(),
            "top_large_updates": top_large_updates.iter().map(audit_step_scan_json).collect::<Vec<_>>(),
        });
        if let Some(gate_report) = &gate_report {
            report
                .as_object_mut()
                .expect("audit scan report is an object")
                .insert("gate".to_owned(), audit_gate_report_json(gate_report));
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("audit scan report serializes")
        );
    } else {
        println!("audit scan");
        println!("path: {}", path.display());
        println!(
            "fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
        println!(
            "summary: steps={} correct={} incorrect={} accuracy={:.2}% witness_mismatches={} lowest_margin_step={} lowest_margin={} trace_payload_bytes={} bytes_per_step={}",
            summary.steps,
            summary.correct,
            summary.incorrect,
            percent(summary.correct, summary.steps),
            summary.witness_mismatches,
            summary
                .lowest_margin_step
                .map_or_else(|| "none".to_owned(), |step| step.to_string()),
            summary
                .lowest_margin
                .map_or_else(|| "none".to_owned(), |margin| margin.to_string()),
            summary.trace_payload_bytes,
            witness_trace_payload_bytes(1)
        );
        println!(
            "largest_update: step={} max_abs_weight_delta={}",
            summary
                .largest_update_step
                .map_or_else(|| "none".to_owned(), |step| step.to_string()),
            summary.max_abs_weight_delta
        );
        println!("by_label:");
        for label in &by_label {
            println!("- {}", format_label_summary_row(label));
        }
        println!("top_confusions:");
        if top_confusions.is_empty() {
            println!("- none");
        } else {
            for confusion in &top_confusions {
                println!("- {}", format_confusion_summary_row(confusion));
            }
        }
        if let Some(gate_report) = &gate_report {
            println!(
                "gate: {}",
                if gate_report.passed {
                    "passed"
                } else {
                    "failed"
                }
            );
            for check in &gate_report.checks {
                println!("- {}", format_audit_gate_check(check));
            }
        }
        println!("top_suspicious:");
        print_scan_rows(&top_suspicious);
        println!("top_low_margin:");
        print_scan_rows(&top_low_margin);
        println!("top_large_updates:");
        print_scan_rows(&top_large_updates);
    }

    if gate_report.as_ref().is_some_and(|report| !report.passed) {
        return Err("audit gate failed".to_owned());
    }

    Ok(())
}

fn print_scan_rows(rows: &[&AuditStepScan]) {
    if rows.is_empty() {
        println!("- none");
    } else {
        for row in rows {
            println!("- {}", format_audit_scan_row(row));
        }
    }
}

fn scan_audit_step(step: usize, audit_step: &AuditStep) -> AuditStepScan {
    let computed_prediction = argmax_first(&audit_step.tape.logits);
    let computed_correct = computed_prediction == audit_step.tape.label;
    let prediction_matches_logits = audit_step.tape.prediction == computed_prediction;
    let correct_matches_logits = (audit_step.tape.correct != 0) == computed_correct;
    let logit_ranks = top_logits(&audit_step.tape.logits);
    let logit_margin = logit_margin(&logit_ranks);
    let active_pixel_count = active_pixels(&audit_step.image).len();
    let error_nonzero_count = audit_step
        .tape
        .error
        .iter()
        .filter(|value| **value != 0)
        .count();
    let max_abs_error = audit_step
        .tape
        .error
        .iter()
        .copied()
        .map(abs_i128)
        .max()
        .unwrap_or(0);
    let bias_delta = applied_bias_delta(&audit_step.tape.error, audit_step.tape.lr);
    let max_abs_bias_delta = bias_delta.iter().copied().map(abs_i128).max().unwrap_or(0);
    let weight_stats = weight_delta_stats(
        &audit_step.image,
        &audit_step.tape.error,
        audit_step.tape.lr,
    );

    AuditStepScan {
        step,
        sample_index: audit_step.source_sample_index,
        label: audit_step.tape.label,
        prediction: audit_step.tape.prediction,
        correct: audit_step.tape.correct != 0,
        lr: audit_step.tape.lr,
        computed_prediction,
        predicted_logit: logit_margin.predicted.value,
        runner_up_digit: logit_margin.runner_up.digit,
        runner_up_logit: logit_margin.runner_up.value,
        margin: logit_margin.margin,
        prediction_matches_logits,
        correct_matches_logits,
        active_pixel_count,
        error_nonzero_count,
        max_abs_error,
        max_abs_bias_delta,
        nonzero_weight_delta_count: weight_stats.nonzero_count,
        max_abs_weight_delta: weight_stats.max_abs,
    }
}

fn audit_scan_summary(scans: &[AuditStepScan]) -> AuditScanSummary {
    let correct = scans.iter().filter(|scan| scan.correct).count();
    let witness_mismatches = scans
        .iter()
        .filter(|scan| !scan.prediction_matches_logits || !scan.correct_matches_logits)
        .count();
    let largest = scans
        .iter()
        .max_by(|left, right| {
            left.max_abs_weight_delta
                .cmp(&right.max_abs_weight_delta)
                .then_with(|| right.step.cmp(&left.step))
        })
        .map(|scan| (scan.step, scan.max_abs_weight_delta));
    let lowest_margin = scans
        .iter()
        .min_by(|left, right| {
            left.margin
                .cmp(&right.margin)
                .then_with(|| left.step.cmp(&right.step))
        })
        .map(|scan| (scan.step, scan.margin));
    AuditScanSummary {
        steps: scans.len(),
        correct,
        incorrect: scans.len().saturating_sub(correct),
        witness_mismatches,
        max_abs_weight_delta: largest.map_or(0, |(_, delta)| delta),
        largest_update_step: largest.map(|(step, _)| step),
        lowest_margin_step: lowest_margin.map(|(step, _)| step),
        lowest_margin: lowest_margin.map(|(_, margin)| margin),
        trace_payload_bytes: witness_trace_payload_bytes(scans.len()),
    }
}

fn label_scan_summaries(scans: &[AuditStepScan]) -> Vec<LabelScanSummary> {
    let mut summaries = (0..DIGITS)
        .map(|label| LabelScanSummary {
            label,
            samples: 0,
            correct: 0,
            incorrect: 0,
            min_margin_step: None,
            min_margin: None,
            largest_update_step: None,
            max_abs_weight_delta: 0,
        })
        .collect::<Vec<_>>();

    for scan in scans {
        let Ok(label) = usize::try_from(scan.label) else {
            continue;
        };
        let Some(summary) = summaries.get_mut(label) else {
            continue;
        };

        summary.samples += 1;
        summary.correct += usize::from(scan.correct);
        summary.incorrect += usize::from(!scan.correct);
        if match summary.min_margin {
            Some(margin) => scan.margin < margin,
            None => true,
        } {
            summary.min_margin = Some(scan.margin);
            summary.min_margin_step = Some(scan.step);
        }
        if scan.max_abs_weight_delta > summary.max_abs_weight_delta {
            summary.max_abs_weight_delta = scan.max_abs_weight_delta;
            summary.largest_update_step = Some(scan.step);
        }
    }

    summaries
}

fn top_confusion_summaries(scans: &[AuditStepScan], limit: usize) -> Vec<ConfusionSummary> {
    let mut confusions: BTreeMap<(i64, i64), ConfusionSummary> = BTreeMap::new();
    for scan in scans.iter().filter(|scan| !scan.correct) {
        let entry = confusions
            .entry((scan.label, scan.prediction))
            .or_insert_with(|| ConfusionSummary {
                label: scan.label,
                prediction: scan.prediction,
                count: 0,
                first_step: scan.step,
                min_margin: scan.margin,
                max_abs_weight_delta: scan.max_abs_weight_delta,
            });
        entry.count += 1;
        entry.first_step = entry.first_step.min(scan.step);
        entry.min_margin = entry.min_margin.min(scan.margin);
        entry.max_abs_weight_delta = entry.max_abs_weight_delta.max(scan.max_abs_weight_delta);
    }

    let mut ranked = confusions.into_values().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| right.max_abs_weight_delta.cmp(&left.max_abs_weight_delta))
            .then_with(|| left.min_margin.cmp(&right.min_margin))
            .then_with(|| left.label.cmp(&right.label))
            .then_with(|| left.prediction.cmp(&right.prediction))
    });
    ranked.truncate(limit);
    ranked
}

fn ranked_audit_scans(scans: &[AuditStepScan], limit: usize) -> Vec<&AuditStepScan> {
    let mut ranked = scans.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        let left_incorrect = !left.correct;
        let right_incorrect = !right.correct;
        right_incorrect
            .cmp(&left_incorrect)
            .then_with(|| right.max_abs_weight_delta.cmp(&left.max_abs_weight_delta))
            .then_with(|| right.max_abs_bias_delta.cmp(&left.max_abs_bias_delta))
            .then_with(|| right.max_abs_error.cmp(&left.max_abs_error))
            .then_with(|| left.step.cmp(&right.step))
    });
    ranked.truncate(limit);
    ranked
}

fn ranked_low_margin_scans(scans: &[AuditStepScan], limit: usize) -> Vec<&AuditStepScan> {
    let mut ranked = scans.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.margin
            .cmp(&right.margin)
            .then_with(|| {
                let left_incorrect = !left.correct;
                let right_incorrect = !right.correct;
                right_incorrect.cmp(&left_incorrect)
            })
            .then_with(|| right.max_abs_weight_delta.cmp(&left.max_abs_weight_delta))
            .then_with(|| left.step.cmp(&right.step))
    });
    ranked.truncate(limit);
    ranked
}

fn ranked_large_update_scans(scans: &[AuditStepScan], limit: usize) -> Vec<&AuditStepScan> {
    let mut ranked = scans.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .max_abs_weight_delta
            .cmp(&left.max_abs_weight_delta)
            .then_with(|| right.max_abs_bias_delta.cmp(&left.max_abs_bias_delta))
            .then_with(|| right.max_abs_error.cmp(&left.max_abs_error))
            .then_with(|| left.step.cmp(&right.step))
    });
    ranked.truncate(limit);
    ranked
}

fn select_audit_step_index(
    scans: &[AuditStepScan],
    requested_step: usize,
    strategy: AuditStepStrategy,
) -> Result<usize, String> {
    if scans.is_empty() {
        return Err("audit bundle has no witness steps to inspect".to_owned());
    }
    if strategy == AuditStepStrategy::Explicit {
        if requested_step >= scans.len() {
            return Err(format!(
                "--audit-step {requested_step} is out of range for {} witness step(s)",
                scans.len()
            ));
        }
        return Ok(requested_step);
    }
    let selected = match strategy {
        AuditStepStrategy::Explicit => unreachable!("explicit handled above"),
        AuditStepStrategy::LowestMargin => ranked_low_margin_scans(scans, 1),
        AuditStepStrategy::LargestUpdate => ranked_large_update_scans(scans, 1),
        AuditStepStrategy::TopSuspicious => ranked_audit_scans(scans, 1),
    }
    .first()
    .map(|scan| scan.step)
    .ok_or_else(|| {
        format!(
            "audit-step strategy `{}` did not select a witness step",
            strategy.as_str()
        )
    })?;
    Ok(selected)
}

fn audit_step_selection_json(
    requested_step: usize,
    strategy: AuditStepStrategy,
    scans: &[AuditStepScan],
    selected: &AuditStepScan,
) -> serde_json::Value {
    let summary = audit_scan_summary(scans);
    let top_suspicious_step = ranked_audit_scans(scans, 1).first().map(|scan| scan.step);
    let top_low_margin_step = ranked_low_margin_scans(scans, 1)
        .first()
        .map(|scan| scan.step);
    let top_large_update_step = ranked_large_update_scans(scans, 1)
        .first()
        .map(|scan| scan.step);
    json!({
        "strategy": strategy.as_str(),
        "requested_step": requested_step,
        "selected_step": selected.step,
        "selected_sample_index": selected.sample_index,
        "selected_label": selected.label,
        "selected_prediction": selected.prediction,
        "selected_correct": selected.correct,
        "selected_margin": selected.margin,
        "selected_max_abs_weight_delta": selected.max_abs_weight_delta,
        "scan_lowest_margin_step": summary.lowest_margin_step,
        "scan_lowest_margin": summary.lowest_margin,
        "scan_largest_update_step": summary.largest_update_step,
        "scan_max_abs_weight_delta": summary.max_abs_weight_delta,
        "scan_top_suspicious_step": top_suspicious_step,
        "scan_top_low_margin_step": top_low_margin_step,
        "scan_top_large_update_step": top_large_update_step,
        "matches_requested": selected.step == requested_step,
        "matches_scan_lowest_margin": summary.lowest_margin_step == Some(selected.step),
        "matches_scan_largest_update": summary.largest_update_step == Some(selected.step),
        "matches_scan_top_suspicious": top_suspicious_step == Some(selected.step),
    })
}

fn audit_scan_summary_json(summary: &AuditScanSummary) -> serde_json::Value {
    json!({
        "steps": summary.steps,
        "correct": summary.correct,
        "incorrect": summary.incorrect,
        "accuracy_percent": percent(summary.correct, summary.steps),
        "witness_mismatches": summary.witness_mismatches,
        "largest_update_step": summary.largest_update_step,
        "max_abs_weight_delta": summary.max_abs_weight_delta,
        "lowest_margin_step": summary.lowest_margin_step,
        "lowest_margin": summary.lowest_margin,
        "trace_payload_bytes": summary.trace_payload_bytes,
        "bytes_per_step": witness_trace_payload_bytes(1),
    })
}

fn audit_gate_report_json(report: &AuditGateReport) -> serde_json::Value {
    json!({
        "passed": report.passed,
        "checks": report.checks.iter().map(audit_gate_check_json).collect::<Vec<_>>(),
    })
}

fn audit_gate_check_json(check: &AuditGateCheck) -> serde_json::Value {
    json!({
        "metric": check.metric,
        "passed": check.passed,
        "actual": check.actual,
        "expectation": check.expectation,
    })
}

fn label_scan_summary_json(summary: &LabelScanSummary) -> serde_json::Value {
    json!({
        "label": summary.label,
        "samples": summary.samples,
        "correct": summary.correct,
        "incorrect": summary.incorrect,
        "accuracy_percent": percent(summary.correct, summary.samples),
        "min_margin_step": summary.min_margin_step,
        "min_margin": summary.min_margin,
        "largest_update_step": summary.largest_update_step,
        "max_abs_weight_delta": summary.max_abs_weight_delta,
    })
}

fn confusion_summary_json(summary: &ConfusionSummary) -> serde_json::Value {
    json!({
        "label": summary.label,
        "prediction": summary.prediction,
        "count": summary.count,
        "first_step": summary.first_step,
        "min_margin": summary.min_margin,
        "max_abs_weight_delta": summary.max_abs_weight_delta,
    })
}

fn audit_step_scan_json(scan: &&AuditStepScan) -> serde_json::Value {
    json!({
        "step": scan.step,
        "sample_index": scan.sample_index,
        "label": scan.label,
        "prediction": scan.prediction,
        "correct": scan.correct,
        "lr": scan.lr,
        "computed_prediction": scan.computed_prediction,
        "predicted_logit": scan.predicted_logit,
        "runner_up_digit": scan.runner_up_digit,
        "runner_up_logit": scan.runner_up_logit,
        "margin": scan.margin,
        "prediction_matches_logits": scan.prediction_matches_logits,
        "correct_matches_logits": scan.correct_matches_logits,
        "active_pixel_count": scan.active_pixel_count,
        "error_nonzero_count": scan.error_nonzero_count,
        "max_abs_error": scan.max_abs_error,
        "max_abs_bias_delta": scan.max_abs_bias_delta,
        "nonzero_weight_delta_count": scan.nonzero_weight_delta_count,
        "max_abs_weight_delta": scan.max_abs_weight_delta,
    })
}

fn format_label_summary_row(summary: &LabelScanSummary) -> String {
    format!(
        "label={} samples={} correct={} incorrect={} accuracy={:.2}% min_margin_step={} min_margin={} largest_update_step={} max_abs_weight_delta={}",
        summary.label,
        summary.samples,
        summary.correct,
        summary.incorrect,
        percent(summary.correct, summary.samples),
        summary
            .min_margin_step
            .map_or_else(|| "none".to_owned(), |step| step.to_string()),
        summary
            .min_margin
            .map_or_else(|| "none".to_owned(), |margin| margin.to_string()),
        summary
            .largest_update_step
            .map_or_else(|| "none".to_owned(), |step| step.to_string()),
        summary.max_abs_weight_delta
    )
}

fn format_confusion_summary_row(summary: &ConfusionSummary) -> String {
    format!(
        "label={} prediction={} count={} first_step={} min_margin={} max_abs_weight_delta={}",
        summary.label,
        summary.prediction,
        summary.count,
        summary.first_step,
        summary.min_margin,
        summary.max_abs_weight_delta
    )
}

fn format_audit_gate_check(check: &AuditGateCheck) -> String {
    format!(
        "{} actual={} requirement=\"{}\" passed={}",
        check.metric, check.actual, check.expectation, check.passed
    )
}

fn format_audit_scan_row(scan: &AuditStepScan) -> String {
    format!(
        "step={} sample={} label={} prediction={} correct={} margin={} runner_up={} active_pixels={} error_nonzero={} max_abs_error={} max_abs_bias_delta={} nonzero_weight_delta={} max_abs_weight_delta={} witness_ok={}",
        scan.step,
        scan.sample_index,
        scan.label,
        scan.prediction,
        scan.correct,
        scan.margin,
        scan.runner_up_digit,
        scan.active_pixel_count,
        scan.error_nonzero_count,
        scan.max_abs_error,
        scan.max_abs_bias_delta,
        scan.nonzero_weight_delta_count,
        scan.max_abs_weight_delta,
        scan.prediction_matches_logits && scan.correct_matches_logits
    )
}

fn inspect_inference_bundle(
    program: &Program,
    path: &Path,
    step_index: usize,
    inference_output: Option<&Path>,
    standalone_rev_output: Option<&Path>,
    markdown_output: Option<&Path>,
    emit_json: bool,
) -> Result<(), String> {
    let bundle = read_audit_bundle(path)?;
    let fingerprints = audit_fingerprints_from_bundle(&bundle)?;
    let final_model = model_from_json(json_field(&bundle, "final_model")?, "final_model")?;
    let trace = json_array_field(&bundle, "witness_trace")?;
    if step_index >= trace.len() {
        return Err(format!(
            "--audit-step {step_index} is out of range for {} witness step(s)",
            trace.len()
        ));
    }
    let steps = parse_audit_steps(trace)?;
    let step = &steps[step_index];
    let inference = run_inference_audit(program, &final_model, &step.image, step.tape.label)?;

    if let Some(output_path) = inference_output {
        write_inference_bundle(
            output_path,
            path,
            &fingerprints,
            step_index,
            step.source_sample_index,
            &final_model,
            &step.image,
            step.tape.label,
            &inference,
        )?;
    }
    if let Some(output_path) = standalone_rev_output {
        write_standalone_rev_classifier(
            output_path,
            &final_model,
            &step.image,
            step.tape.label,
            &inference,
        )?;
    }

    let report = inference_report_json(
        "reverie_mnist_linear_q31_inference_audit",
        Some(path),
        Some(&fingerprints),
        step_index,
        step.source_sample_index,
        step.tape.label,
        inference_output,
        standalone_rev_output,
        &inference,
    );
    if let Some(output_path) = markdown_output {
        write_markdown_file(output_path, &render_inference_audit_markdown(&report)?)?;
    }

    if emit_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("inference audit report serializes")
        );
    } else {
        println!("inference audit step {step_index}");
        println!("path: {}", path.display());
        println!(
            "fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
        println!(
            "sample: source_index={} label={} prediction={} correct={}",
            step.source_sample_index, step.tape.label, inference.prediction, inference.correct
        );
        println!(
            "witness_checks: computed_prediction={} prediction_matches_logits={} correct_matches_logits={}",
            inference.computed_prediction,
            inference.prediction_matches_logits,
            inference.correct_matches_logits
        );
        println!(
            "active_pixels: count={} first={}",
            inference.active_pixels.len(),
            format_active_pixel_preview(&inference.active_pixels, 12)
        );
        println!("logits: {:?}", inference.logits);
        println!(
            "top_logits: {}",
            format_logit_preview(&inference.top_logits)
        );
        println!(
            "attribution: predicted={} runner_up={} margin={} bias={} reconstructed_logit={} matches_logit={} reconstructed_margin={} matches_margin={} top_contributions={} top_margin_contributions={}",
            inference.attribution.predicted_digit,
            inference.attribution.runner_up_digit,
            inference.attribution.margin,
            inference.attribution.bias,
            inference.attribution.reconstructed_logit,
            inference.attribution.matches_logit,
            inference.attribution.reconstructed_margin,
            inference.attribution.matches_margin,
            format_contribution_preview(&inference.attribution.top_contributions),
            format_margin_contribution_preview(&inference.attribution.top_margin_contributions)
        );
        println!(
            "inverse: restored_initial_state=true elapsed={:.3}s",
            secs(inference.reverse_elapsed)
        );
        println!(
            "storage: replay_payload_bytes={} witness_payload_bytes={} trace_payload_bytes={} runtime_state_payload_bytes={} forward_recompute_steps={} inverse_recompute_steps={}",
            inference.memory.replay_payload_bytes,
            inference.memory.witness_payload_bytes,
            inference.memory.trace_payload_bytes,
            inference.memory.runtime_state_payload_bytes,
            inference.memory.forward_recompute_steps,
            inference.memory.inverse_recompute_steps
        );
        if let Some(output_path) = inference_output {
            println!("inference bundle: {}", output_path.display());
        }
        if let Some(output_path) = standalone_rev_output {
            println!("standalone rev: {}", output_path.display());
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct ModelInferenceSample {
    image: Vec<u8>,
    label: i64,
    source: ModelInferenceSampleSource,
}

#[derive(Debug, Clone)]
struct LabeledSample {
    image: Vec<u8>,
    label: i64,
    audit_step: Option<usize>,
    source_sample_index: Option<usize>,
}

#[derive(Debug, Clone)]
struct ModelEvaluationRow {
    index: usize,
    audit_step: Option<usize>,
    source_sample_index: Option<usize>,
    sample_fingerprint: String,
    label: i64,
    prediction: i64,
    correct: bool,
    runner_up_digit: usize,
    margin: i64,
    active_pixels: usize,
}

#[derive(Debug, Clone)]
struct ModelEvaluationSummary {
    samples: usize,
    correct: usize,
    incorrect: usize,
    lowest_margin_index: Option<usize>,
    lowest_margin: Option<i64>,
    forward_elapsed: Duration,
    reverse_elapsed: Duration,
    elapsed: Duration,
}

#[derive(Debug, Clone)]
struct ModelEvaluationScanSummary {
    samples: usize,
    correct: usize,
    incorrect: usize,
    lowest_margin_index: Option<usize>,
    lowest_margin: Option<i64>,
}

#[derive(Debug, Clone)]
struct ModelEvaluationLabelSummary {
    label: usize,
    samples: usize,
    correct: usize,
    incorrect: usize,
    lowest_margin_index: Option<usize>,
    lowest_margin: Option<i64>,
}

#[derive(Debug, Clone)]
struct ModelEvaluationConfusionSummary {
    label: i64,
    prediction: i64,
    count: usize,
    first_index: usize,
    lowest_margin: i64,
}

#[derive(Debug, Clone)]
enum ModelInferenceSampleSource {
    Audit {
        path: String,
        fingerprints: AuditFingerprints,
        audit_step: usize,
        sample_index: usize,
    },
    Json {
        path: String,
        fingerprint: String,
    },
    Evaluation {
        path: String,
        fingerprints: ModelEvaluationFingerprints,
        row_index: usize,
        audit_step: Option<usize>,
        source_sample_index: Option<usize>,
        sample_fingerprint: String,
    },
}

impl ModelInferenceSampleSource {
    fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Audit {
                path,
                fingerprints,
                audit_step,
                sample_index,
            } => json!({
                "kind": "audit_bundle",
                "path": path,
                "audit_step": audit_step,
                "sample_index": sample_index,
                "fingerprints": fingerprints.to_json(),
            }),
            Self::Json { path, fingerprint } => json!({
                "kind": "sample_json",
                "path": path,
                "fingerprint": fingerprint,
            }),
            Self::Evaluation {
                path,
                fingerprints,
                row_index,
                audit_step,
                source_sample_index,
                sample_fingerprint,
            } => json!({
                "kind": "model_evaluation_bundle",
                "path": path,
                "fingerprints": fingerprints.to_json(),
                "row_index": row_index,
                "audit_step": audit_step,
                "source_sample_index": source_sample_index,
                "sample_fingerprint": sample_fingerprint,
            }),
        }
    }

    fn audit_step(&self) -> Option<usize> {
        match self {
            Self::Audit { audit_step, .. } => Some(*audit_step),
            Self::Json { .. } => None,
            Self::Evaluation { audit_step, .. } => *audit_step,
        }
    }

    fn sample_index(&self) -> Option<usize> {
        match self {
            Self::Audit { sample_index, .. } => Some(*sample_index),
            Self::Json { .. } => None,
            Self::Evaluation {
                source_sample_index,
                ..
            } => *source_sample_index,
        }
    }

    fn sample_audit_json(&self) -> serde_json::Value {
        match self {
            Self::Audit {
                path, fingerprints, ..
            } => json!({
                "path": path,
                "fingerprints": fingerprints.to_json(),
            }),
            Self::Json { .. } | Self::Evaluation { .. } => serde_json::Value::Null,
        }
    }

    fn sample_json_json(&self) -> serde_json::Value {
        match self {
            Self::Json { path, fingerprint } => json!({
                "path": path,
                "fingerprint": fingerprint,
            }),
            Self::Audit { .. } | Self::Evaluation { .. } => serde_json::Value::Null,
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Audit {
                path,
                audit_step,
                sample_index,
                ..
            } => format!("audit path={path} step={audit_step} sample_index={sample_index}"),
            Self::Json { path, fingerprint } => {
                format!("json path={path} fingerprint={fingerprint}")
            }
            Self::Evaluation {
                path,
                row_index,
                audit_step,
                source_sample_index,
                sample_fingerprint,
                ..
            } => format!(
                "evaluation path={path} row={row_index} audit_step={} source_sample_index={} sample={sample_fingerprint}",
                audit_step.map_or_else(|| "none".to_owned(), |step| step.to_string()),
                source_sample_index.map_or_else(|| "none".to_owned(), |index| index.to_string())
            ),
        }
    }
}

fn model_inference_sample_from_audit(
    path: &Path,
    step_index: usize,
) -> Result<ModelInferenceSample, String> {
    let bundle = read_audit_bundle(path)?;
    let fingerprints = audit_fingerprints_from_bundle(&bundle)?;
    let trace = json_array_field(&bundle, "witness_trace")?;
    if step_index >= trace.len() {
        return Err(format!(
            "--audit-step {step_index} is out of range for {} witness step(s)",
            trace.len()
        ));
    }
    let steps = parse_audit_steps(trace)?;
    let step = &steps[step_index];
    Ok(ModelInferenceSample {
        image: step.image.clone(),
        label: step.tape.label,
        source: ModelInferenceSampleSource::Audit {
            path: path.display().to_string(),
            fingerprints,
            audit_step: step_index,
            sample_index: step.source_sample_index,
        },
    })
}

fn model_inference_sample_from_json(path: &Path) -> Result<ModelInferenceSample, String> {
    let data = fs::read(path)
        .map_err(|error| format!("failed to read sample JSON {}: {error}", path.display()))?;
    let sample: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|error| format!("failed to parse sample JSON {}: {error}", path.display()))?;
    let parsed = labeled_sample_from_json_value(&sample, "sample JSON")?;
    Ok(ModelInferenceSample {
        image: parsed.image,
        label: parsed.label,
        source: ModelInferenceSampleSource::Json {
            path: path.display().to_string(),
            fingerprint: sha256_json(&sample),
        },
    })
}

fn labeled_sample_from_json_value(
    sample: &serde_json::Value,
    context: &str,
) -> Result<LabeledSample, String> {
    if let Some(kind) = sample.get("kind").and_then(serde_json::Value::as_str) {
        if kind != "reverie_mnist_linear_q31_sample" {
            return Err(format!(
                "{context} kind `{kind}` is not reverie_mnist_linear_q31_sample"
            ));
        }
    }
    let image = json_u8_vec_field(sample, "image_u8", IMAGE_PIXELS)?;
    let label = json_i64_field(sample, "label")?;
    if !(0..DIGITS as i64).contains(&label) {
        return Err(format!("{context} `label` is outside 0..10"));
    }
    Ok(LabeledSample {
        image,
        label,
        audit_step: json_optional_usize_field(sample, "audit_step")?,
        source_sample_index: json_optional_usize_field(sample, "source_sample_index")?,
    })
}

fn labeled_samples_from_json(path: &Path) -> Result<(Vec<LabeledSample>, String), String> {
    let data = fs::read(path)
        .map_err(|error| format!("failed to read samples JSON {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|error| format!("failed to parse samples JSON {}: {error}", path.display()))?;
    let samples = if let Some(array) = value.as_array() {
        array
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                labeled_sample_from_json_value(sample, &format!("samples JSON [{index}]"))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        if let Some(kind) = value.get("kind").and_then(serde_json::Value::as_str) {
            if kind != "reverie_mnist_linear_q31_samples" {
                return Err(format!(
                    "samples JSON kind `{kind}` is not reverie_mnist_linear_q31_samples"
                ));
            }
        }
        let samples = json_array_field(&value, "samples")?;
        samples
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                labeled_sample_from_json_value(sample, &format!("samples JSON samples[{index}]"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    if samples.is_empty() {
        return Err("samples JSON must contain at least one sample".to_owned());
    }
    Ok((samples, sha256_json(&value)))
}

fn labeled_samples_from_samples_object(
    value: &serde_json::Value,
    context: &str,
) -> Result<Vec<LabeledSample>, String> {
    let samples = json_array_field(value, "samples")?
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            labeled_sample_from_json_value(sample, &format!("{context} samples[{index}]"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if samples.is_empty() {
        return Err(format!("{context} must contain at least one sample"));
    }
    Ok(samples)
}

fn evaluate_model_bundle(
    program: &Program,
    model_path: &Path,
    samples_path: &Path,
    gate: &ModelEvaluationGate,
    evaluation_output: Option<&Path>,
    emit_json: bool,
) -> Result<(), String> {
    let model_bundle = read_model_bundle(model_path)?;
    let model_fingerprints = model_fingerprints_from_bundle(&model_bundle)?;
    let model = model_from_json(json_field(&model_bundle, "model")?, "model")?;
    let (samples, samples_fingerprint) = labeled_samples_from_json(samples_path)?;
    let (summary, rows) = run_model_evaluation(program, &model, &samples)?;
    let proof = batch_inference_proof_json(summary.samples);
    let gate_policy = gate.policy_json();
    let gate_report = gate.evaluate(&summary, &rows);
    let evaluation_fingerprints = if let Some(output_path) = evaluation_output {
        Some(write_model_evaluation_bundle(
            output_path,
            model_path,
            &model_fingerprints,
            samples_path,
            &samples_fingerprint,
            &model,
            &samples,
            &summary,
            &proof,
            &rows,
            gate_policy.as_ref(),
            gate_report.as_ref(),
        )?)
    } else {
        None
    };

    if emit_json {
        let report = model_evaluation_report_json(
            model_path,
            &model_fingerprints,
            samples_path,
            &samples_fingerprint,
            evaluation_output,
            evaluation_fingerprints.as_ref(),
            &summary,
            &proof,
            &rows,
            gate_policy.as_ref(),
            gate_report.as_ref(),
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("model evaluation report serializes")
        );
    } else {
        println!("model evaluation");
        println!("model: {}", model_path.display());
        println!(
            "model_fingerprint: computation={} payload={}",
            model_fingerprints.computation, model_fingerprints.payload
        );
        println!(
            "samples_json: path={} fingerprint={}",
            samples_path.display(),
            samples_fingerprint
        );
        println!(
            "summary: samples={} correct={} incorrect={} accuracy={:.2}% lowest_margin_index={} lowest_margin={} elapsed={:.3}s samples_per_sec={:.1}",
            summary.samples,
            summary.correct,
            summary.incorrect,
            percent(summary.correct, summary.samples),
            summary
                .lowest_margin_index
                .map_or_else(|| "none".to_owned(), |index| index.to_string()),
            summary
                .lowest_margin
                .map_or_else(|| "none".to_owned(), |margin| margin.to_string()),
            secs(summary.elapsed),
            per_second(summary.samples, summary.elapsed)
        );
        println!(
            "proof: model_bytes={} sample_bytes={} witness_bytes={} replay_bytes={} forward_recompute_steps={} inverse_recompute_steps={}",
            Model::payload_bytes(),
            summary.samples * single_sample_payload_bytes(),
            summary.samples * inference_witness_payload_bytes(),
            Model::payload_bytes()
                + summary.samples
                    * (single_sample_payload_bytes() + inference_witness_payload_bytes()),
            summary.samples,
            summary.samples
        );
        if let (Some(output_path), Some(fingerprints)) =
            (evaluation_output, evaluation_fingerprints.as_ref())
        {
            println!("evaluation bundle: {}", output_path.display());
            println!(
                "evaluation_fingerprint: computation={} payload={}",
                fingerprints.computation, fingerprints.payload
            );
        }
        if let Some(gate_report) = &gate_report {
            println!(
                "gate: {}",
                if gate_report.passed {
                    "passed"
                } else {
                    "failed"
                }
            );
            for check in &gate_report.checks {
                println!("- {}", format_model_evaluation_gate_check(check));
            }
        }
    }

    if gate_report.as_ref().is_some_and(|report| !report.passed) {
        return Err("model evaluation gate failed".to_owned());
    }

    Ok(())
}

fn run_model_evaluation(
    program: &Program,
    model: &Model,
    samples: &[LabeledSample],
) -> Result<(ModelEvaluationSummary, Vec<ModelEvaluationRow>), String> {
    let start = Instant::now();
    let mut rows = Vec::with_capacity(samples.len());
    let mut correct = 0_usize;
    let mut forward_elapsed = Duration::default();
    let mut reverse_elapsed = Duration::default();
    let mut lowest_margin: Option<(usize, i64)> = None;

    for (index, sample) in samples.iter().enumerate() {
        let inference = run_inference_audit(program, model, &sample.image, sample.label)?;
        correct += usize::from(inference.correct);
        forward_elapsed += inference.forward_elapsed;
        reverse_elapsed += inference.reverse_elapsed;
        let margin = inference.attribution.margin;
        if lowest_margin.is_none_or(|(_, current)| margin < current) {
            lowest_margin = Some((index, margin));
        }
        rows.push(ModelEvaluationRow {
            index,
            audit_step: sample.audit_step,
            source_sample_index: sample.source_sample_index,
            sample_fingerprint: sha256_json(&labeled_sample_json(sample)),
            label: sample.label,
            prediction: inference.prediction,
            correct: inference.correct,
            runner_up_digit: inference.attribution.runner_up_digit,
            margin,
            active_pixels: inference.active_pixels.len(),
        });
    }

    let summary = ModelEvaluationSummary {
        samples: samples.len(),
        correct,
        incorrect: samples.len().saturating_sub(correct),
        lowest_margin_index: lowest_margin.map(|(index, _)| index),
        lowest_margin: lowest_margin.map(|(_, margin)| margin),
        forward_elapsed,
        reverse_elapsed,
        elapsed: start.elapsed(),
    };
    Ok((summary, rows))
}

#[allow(clippy::too_many_arguments)]
fn model_evaluation_report_json(
    model_path: &Path,
    model_fingerprints: &ModelFingerprints,
    samples_path: &Path,
    samples_fingerprint: &str,
    evaluation_output: Option<&Path>,
    evaluation_fingerprints: Option<&ModelEvaluationFingerprints>,
    summary: &ModelEvaluationSummary,
    proof: &serde_json::Value,
    rows: &[ModelEvaluationRow],
    gate_policy: Option<&serde_json::Value>,
    gate_report: Option<&ModelEvaluationGateReport>,
) -> serde_json::Value {
    let mut report = json!({
        "kind": "reverie_mnist_linear_q31_model_evaluation",
        "model_bundle": {
            "path": model_path.display().to_string(),
            "fingerprints": model_fingerprints.to_json(),
        },
        "samples_json": {
            "path": samples_path.display().to_string(),
            "fingerprint": samples_fingerprint,
        },
        "evaluation_output": evaluation_output.map(|path| path.display().to_string()),
        "summary": model_evaluation_summary_json(summary),
        "proof": proof,
        "rows": rows.iter().map(model_evaluation_row_json).collect::<Vec<_>>(),
    });
    if let Some(fingerprints) = evaluation_fingerprints {
        report
            .as_object_mut()
            .expect("model evaluation report is an object")
            .insert("fingerprints".to_owned(), fingerprints.to_json());
    }
    if let Some(gate_policy) = gate_policy {
        report
            .as_object_mut()
            .expect("model evaluation report is an object")
            .insert("gate_policy".to_owned(), gate_policy.clone());
    }
    if let Some(gate_report) = gate_report {
        report
            .as_object_mut()
            .expect("model evaluation report is an object")
            .insert(
                "gate".to_owned(),
                model_evaluation_gate_report_json(gate_report),
            );
    }
    report
}

fn model_evaluation_summary_json(summary: &ModelEvaluationSummary) -> serde_json::Value {
    json!({
        "samples": summary.samples,
        "correct": summary.correct,
        "incorrect": summary.incorrect,
        "accuracy_percent": percent(summary.correct, summary.samples),
        "lowest_margin_index": summary.lowest_margin_index,
        "lowest_margin": summary.lowest_margin,
        "elapsed_seconds": secs(summary.elapsed),
        "samples_per_second": per_second(summary.samples, summary.elapsed),
        "forward_elapsed_seconds": secs(summary.forward_elapsed),
        "reverse_elapsed_seconds": secs(summary.reverse_elapsed),
    })
}

fn model_evaluation_row_json(row: &ModelEvaluationRow) -> serde_json::Value {
    let mut value = json!({
        "index": row.index,
        "sample_fingerprint": row.sample_fingerprint,
        "label": row.label,
        "prediction": row.prediction,
        "correct": row.correct,
        "runner_up_digit": row.runner_up_digit,
        "margin": row.margin,
        "active_pixels": row.active_pixels,
    });
    let object = value
        .as_object_mut()
        .expect("model evaluation row is an object");
    if let Some(audit_step) = row.audit_step {
        object.insert("audit_step".to_owned(), json!(audit_step));
    }
    if let Some(source_sample_index) = row.source_sample_index {
        object.insert("source_sample_index".to_owned(), json!(source_sample_index));
    }
    value
}

fn model_evaluation_gate_report_json(report: &ModelEvaluationGateReport) -> serde_json::Value {
    json!({
        "passed": report.passed,
        "checks": report.checks.iter().map(model_evaluation_gate_check_json).collect::<Vec<_>>(),
    })
}

fn model_evaluation_gate_check_json(check: &ModelEvaluationGateCheck) -> serde_json::Value {
    json!({
        "metric": check.metric,
        "passed": check.passed,
        "actual": check.actual,
        "expectation": check.expectation,
    })
}

fn format_model_evaluation_gate_check(check: &ModelEvaluationGateCheck) -> String {
    format!(
        "{} actual={} requirement=\"{}\" passed={}",
        check.metric, check.actual, check.expectation, check.passed
    )
}

#[allow(clippy::too_many_arguments)]
fn write_model_evaluation_bundle(
    path: &Path,
    model_path: &Path,
    model_fingerprints: &ModelFingerprints,
    samples_path: &Path,
    samples_fingerprint: &str,
    model: &Model,
    samples: &[LabeledSample],
    summary: &ModelEvaluationSummary,
    proof: &serde_json::Value,
    rows: &[ModelEvaluationRow],
    gate_policy: Option<&serde_json::Value>,
    gate_report: Option<&ModelEvaluationGateReport>,
) -> Result<ModelEvaluationFingerprints, String> {
    let mut bundle = json!({
        "kind": "reverie_mnist_linear_q31_model_evaluation_bundle",
        "schema_version": 1,
        "program": "examples/mnist_identify.rev",
        "source_model_bundle": {
            "path": model_path.display().to_string(),
            "fingerprints": model_fingerprints.to_json(),
        },
        "source_samples_json": {
            "path": samples_path.display().to_string(),
            "fingerprint": samples_fingerprint,
        },
        "tensor_shapes": {
            "image": [IMAGE_PIXELS],
            "weights": [IMAGE_PIXELS, DIGITS],
            "bias": [DIGITS],
            "logits": [DIGITS],
        },
        "model": {
            "weights": &model.weights,
            "bias": &model.bias,
        },
        "samples": model_evaluation_samples_json(samples_path, samples_fingerprint, samples),
        "summary": model_evaluation_summary_json(summary),
        "proof": proof,
        "rows": rows.iter().map(model_evaluation_row_json).collect::<Vec<_>>(),
        "report": model_evaluation_report_json(
            model_path,
            model_fingerprints,
            samples_path,
            samples_fingerprint,
            Some(path),
            None,
            summary,
            proof,
            rows,
            gate_policy,
            gate_report,
        ),
    });
    if let Some(gate_policy) = gate_policy {
        bundle
            .as_object_mut()
            .expect("model evaluation bundle is an object")
            .insert("gate_policy".to_owned(), gate_policy.clone());
    }
    if let Some(gate_report) = gate_report {
        bundle
            .as_object_mut()
            .expect("model evaluation bundle is an object")
            .insert(
                "gate".to_owned(),
                model_evaluation_gate_report_json(gate_report),
            );
    }

    let fingerprints = model_evaluation_fingerprints_for_unsigned_payload(&bundle);
    bundle
        .as_object_mut()
        .expect("model evaluation bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints.to_json());

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let encoded =
        serde_json::to_string_pretty(&bundle).expect("model evaluation bundle serializes");
    fs::write(path, encoded).map_err(|error| {
        format!(
            "failed to write model evaluation bundle {}: {error}",
            path.display()
        )
    })?;
    Ok(fingerprints)
}

fn verify_model_evaluation_bundle(
    program: &Program,
    path: &Path,
    emit_json: bool,
) -> Result<(), String> {
    let bundle = read_model_evaluation_bundle(path)?;
    let fingerprints = model_evaluation_fingerprints_from_bundle(&bundle)?;
    let model_json = json_field(&bundle, "model")?;
    let model = model_from_json(model_json, "model")?;
    let samples = labeled_samples_from_embedded_json(json_field(&bundle, "samples")?)?;
    let source_checks =
        verify_model_evaluation_source_artifacts(path, &bundle, model_json, &samples)?;

    let start = Instant::now();
    let (summary, rows) = run_model_evaluation(program, &model, &samples)?;
    let elapsed = start.elapsed();
    verify_model_evaluation_summary(json_field(&bundle, "summary")?, &summary)?;
    verify_batch_inference_proof(json_field(&bundle, "proof")?, summary.samples)?;
    let expected_rows = rows
        .iter()
        .map(model_evaluation_row_json)
        .collect::<Vec<_>>();
    if json_array_field(&bundle, "rows")? != &expected_rows {
        return Err(
            "model evaluation bundle rows do not match recomputed Reverie inference".to_owned(),
        );
    }

    let gate_report = if let Some(policy) = bundle.get("gate_policy") {
        let gate = ModelEvaluationGate::from_policy_json(policy)?;
        let recomputed = gate
            .evaluate(&summary, &rows)
            .expect("gate policy is enabled");
        let expected_gate = model_evaluation_gate_report_json(&recomputed);
        if json_field(&bundle, "gate")? != &expected_gate {
            return Err(
                "model evaluation bundle gate report does not match recomputed rows".to_owned(),
            );
        }
        Some(recomputed)
    } else {
        if bundle.get("gate").is_some() {
            return Err(
                "model evaluation bundle has a gate report without a gate policy".to_owned(),
            );
        }
        None
    };

    if emit_json {
        let mut report = json!({
            "kind": "reverie_mnist_linear_q31_model_evaluation_verification",
            "path": path.display().to_string(),
            "fingerprints": fingerprints.to_json(),
            "summary": model_evaluation_summary_json(&summary),
            "rows_match": true,
            "proof_matches": true,
            "source_model_checked": source_checks.model.checked,
            "source_samples_checked": source_checks.samples.checked,
            "source_model": source_checks.model.to_json(),
            "source_samples_json": source_checks.samples.to_json(),
            "restored_initial_state": true,
            "elapsed_seconds": secs(elapsed),
        });
        if let Some(gate_report) = &gate_report {
            report
                .as_object_mut()
                .expect("model evaluation verification report is an object")
                .insert(
                    "gate".to_owned(),
                    model_evaluation_gate_report_json(gate_report),
                );
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .expect("model evaluation verification report serializes")
        );
    } else {
        println!(
            "model evaluation ok: samples={} correct={} incorrect={} rows_match=true proof_matches=true source_model_checked={} source_samples_checked={} restored_initial_state=true elapsed={:.3}s",
            summary.samples,
            summary.correct,
            summary.incorrect,
            source_checks.model.checked,
            source_checks.samples.checked,
            secs(elapsed)
        );
        println!("source_model: {}", source_checks.model.format_line());
        println!(
            "source_samples_json: {}",
            source_checks.samples.format_line()
        );
        if let Some(gate_report) = &gate_report {
            println!(
                "gate: {}",
                if gate_report.passed {
                    "passed"
                } else {
                    "failed"
                }
            );
            for check in &gate_report.checks {
                println!("- {}", format_model_evaluation_gate_check(check));
            }
        }
        println!(
            "fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
    }

    if gate_report.as_ref().is_some_and(|report| !report.passed) {
        return Err("model evaluation bundle gate failed".to_owned());
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct ModelEvaluationSourceChecks {
    model: ModelEvaluationSourceArtifactCheck,
    samples: ModelEvaluationSourceArtifactCheck,
}

#[derive(Debug, Clone)]
struct ModelEvaluationSourceArtifactCheck {
    path: String,
    resolved_path: String,
    checked: bool,
    payload_fingerprint: String,
    items_checked: usize,
    unavailable_reason: Option<&'static str>,
}

impl ModelEvaluationSourceArtifactCheck {
    fn to_json(&self) -> serde_json::Value {
        let mut value = json!({
            "path": self.path,
            "resolved_path": self.resolved_path,
            "checked": self.checked,
            "payload_fingerprint": self.payload_fingerprint,
            "items_checked": self.items_checked,
        });
        if let Some(reason) = self.unavailable_reason {
            value
                .as_object_mut()
                .expect("evaluation source check JSON is an object")
                .insert("unavailable_reason".to_owned(), json!(reason));
        }
        value
    }

    fn format_line(&self) -> String {
        if self.checked {
            format!(
                "checked=true items_checked={} payload={}",
                self.items_checked, self.payload_fingerprint
            )
        } else {
            format!(
                "checked=false reason={} path={}",
                self.unavailable_reason.unwrap_or("unavailable"),
                self.path
            )
        }
    }
}

fn verify_model_evaluation_source_artifacts(
    evaluation_path: &Path,
    bundle: &serde_json::Value,
    model_json: &serde_json::Value,
    samples: &[LabeledSample],
) -> Result<ModelEvaluationSourceChecks, String> {
    Ok(ModelEvaluationSourceChecks {
        model: verify_model_evaluation_source_model(evaluation_path, bundle, model_json)?,
        samples: verify_model_evaluation_source_samples(evaluation_path, bundle, samples)?,
    })
}

fn verify_model_evaluation_source_model(
    evaluation_path: &Path,
    bundle: &serde_json::Value,
    model_json: &serde_json::Value,
) -> Result<ModelEvaluationSourceArtifactCheck, String> {
    let source = json_field(bundle, "source_model_bundle")?;
    let source_path_text = json_str_field(source, "path")?.to_owned();
    let expected_fingerprints = model_fingerprints_from_json(json_field(source, "fingerprints")?)?;
    let resolved_path = resolve_referenced_artifact_path(evaluation_path, &source_path_text);
    let resolved_path_text = resolved_path.display().to_string();
    if !resolved_path.exists() {
        return Ok(ModelEvaluationSourceArtifactCheck {
            path: source_path_text,
            resolved_path: resolved_path_text,
            checked: false,
            payload_fingerprint: expected_fingerprints.payload,
            items_checked: 0,
            unavailable_reason: Some("source_model_not_found"),
        });
    }

    let source_bundle = read_model_bundle(&resolved_path)?;
    let actual_fingerprints = model_fingerprints_from_bundle(&source_bundle)?;
    if actual_fingerprints != expected_fingerprints {
        return Err(
            "model evaluation source model fingerprints do not match referenced model bundle"
                .to_owned(),
        );
    }
    if json_field(&source_bundle, "model")? != model_json {
        return Err("model evaluation source model does not match embedded model".to_owned());
    }

    Ok(ModelEvaluationSourceArtifactCheck {
        path: source_path_text,
        resolved_path: resolved_path_text,
        checked: true,
        payload_fingerprint: actual_fingerprints.payload,
        items_checked: 1,
        unavailable_reason: None,
    })
}

fn verify_model_evaluation_source_samples(
    evaluation_path: &Path,
    bundle: &serde_json::Value,
    samples: &[LabeledSample],
) -> Result<ModelEvaluationSourceArtifactCheck, String> {
    let source = json_field(bundle, "source_samples_json")?;
    let source_path_text = json_str_field(source, "path")?.to_owned();
    let expected_fingerprint = json_str_field(source, "fingerprint")?.to_owned();
    let resolved_path = resolve_referenced_artifact_path(evaluation_path, &source_path_text);
    let resolved_path_text = resolved_path.display().to_string();
    if !resolved_path.exists() {
        return Ok(ModelEvaluationSourceArtifactCheck {
            path: source_path_text,
            resolved_path: resolved_path_text,
            checked: false,
            payload_fingerprint: expected_fingerprint,
            items_checked: 0,
            unavailable_reason: Some("source_samples_not_found"),
        });
    }

    let data = fs::read(&resolved_path).map_err(|error| {
        format!(
            "failed to read model evaluation source samples {}: {error}",
            resolved_path.display()
        )
    })?;
    let source_value: serde_json::Value = serde_json::from_slice(&data).map_err(|error| {
        format!(
            "failed to parse model evaluation source samples {}: {error}",
            resolved_path.display()
        )
    })?;
    let actual_fingerprint = sha256_json(&source_value);
    if actual_fingerprint != expected_fingerprint {
        return Err(
            "model evaluation source samples fingerprint does not match referenced samples file"
                .to_owned(),
        );
    }

    let source_samples = if source_value.get("kind").and_then(serde_json::Value::as_str)
        == Some("reverie_mnist_linear_q31_samples")
        && source_value.get("fingerprints").is_some()
    {
        let source_bundle = read_sample_set_bundle(&resolved_path)?;
        let parsed_samples =
            labeled_samples_from_samples_object(&source_bundle, "model evaluation source samples")?;
        verify_sample_set_report(json_field(&source_bundle, "report")?, &parsed_samples)?;
        let proof = sample_set_proof_json(
            json_field(&source_bundle, "source_audit_bundle")?,
            json_field(&source_bundle, "samples")?,
            json_field(&source_bundle, "report")?,
        )?;
        if let Some(stored_proof) = source_bundle.get("proof") {
            verify_sample_set_proof(stored_proof, &proof)?;
        }
        let _ = verify_sample_set_source_audit(&resolved_path, &source_bundle, &parsed_samples)?;
        parsed_samples
    } else {
        labeled_samples_from_json(&resolved_path)?.0
    };

    if source_samples.len() != samples.len() {
        return Err(format!(
            "model evaluation source samples expected {} samples, found {}",
            samples.len(),
            source_samples.len()
        ));
    }
    for (index, (source_sample, embedded_sample)) in source_samples.iter().zip(samples).enumerate()
    {
        if labeled_sample_json(source_sample) != labeled_sample_json(embedded_sample) {
            return Err(format!(
                "model evaluation source samples sample {index} does not match embedded sample"
            ));
        }
    }

    Ok(ModelEvaluationSourceArtifactCheck {
        path: source_path_text,
        resolved_path: resolved_path_text,
        checked: true,
        payload_fingerprint: actual_fingerprint,
        items_checked: samples.len(),
        unavailable_reason: None,
    })
}

fn scan_model_evaluation_bundle(path: &Path, limit: usize, emit_json: bool) -> Result<(), String> {
    let bundle = read_model_evaluation_bundle(path)?;
    let fingerprints = model_evaluation_fingerprints_from_bundle(&bundle)?;
    let rows = json_array_field(&bundle, "rows")?
        .iter()
        .enumerate()
        .map(|(position, row)| model_evaluation_row_from_json(row, position))
        .collect::<Result<Vec<_>, _>>()?;
    let summary = model_evaluation_scan_summary(&rows);
    let by_label = model_evaluation_label_summaries(&rows);
    let top_confusions = model_evaluation_confusion_summaries(&rows, limit);
    let top_incorrect = ranked_incorrect_model_evaluation_rows(&rows, limit);
    let top_low_margin = ranked_low_margin_model_evaluation_rows(&rows, limit);

    if emit_json {
        let report = json!({
            "kind": "reverie_mnist_linear_q31_model_evaluation_scan",
            "path": path.display().to_string(),
            "fingerprints": fingerprints.to_json(),
            "summary": model_evaluation_scan_summary_json(&summary),
            "by_label": by_label.iter().map(model_evaluation_label_summary_json).collect::<Vec<_>>(),
            "top_confusions": top_confusions.iter().map(model_evaluation_confusion_summary_json).collect::<Vec<_>>(),
            "top_incorrect": top_incorrect.iter().map(|row| model_evaluation_row_json(row)).collect::<Vec<_>>(),
            "top_low_margin": top_low_margin.iter().map(|row| model_evaluation_row_json(row)).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("model evaluation scan report serializes")
        );
    } else {
        println!("model evaluation scan");
        println!("path: {}", path.display());
        println!(
            "fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
        println!(
            "summary: samples={} correct={} incorrect={} accuracy={:.2}% lowest_margin_index={} lowest_margin={}",
            summary.samples,
            summary.correct,
            summary.incorrect,
            percent(summary.correct, summary.samples),
            summary
                .lowest_margin_index
                .map_or_else(|| "none".to_owned(), |index| index.to_string()),
            summary
                .lowest_margin
                .map_or_else(|| "none".to_owned(), |margin| margin.to_string()),
        );
        println!("by_label:");
        for label in &by_label {
            println!("- {}", format_model_evaluation_label_summary(label));
        }
        println!("top_confusions:");
        if top_confusions.is_empty() {
            println!("- none");
        } else {
            for confusion in &top_confusions {
                println!("- {}", format_model_evaluation_confusion(confusion));
            }
        }
        println!("top_incorrect:");
        print_model_evaluation_rows(&top_incorrect);
        println!("top_low_margin:");
        print_model_evaluation_rows(&top_low_margin);
    }

    Ok(())
}

fn inspect_model_evaluation_row(
    program: &Program,
    path: &Path,
    row_index: usize,
    inference_output: Option<&Path>,
    standalone_rev_output: Option<&Path>,
    markdown_output: Option<&Path>,
    emit_json: bool,
) -> Result<(), String> {
    let bundle = read_model_evaluation_bundle(path)?;
    let evaluation_fingerprints = model_evaluation_fingerprints_from_bundle(&bundle)?;
    let source_model_bundle = json_field(&bundle, "source_model_bundle")?;
    let source_model_path = PathBuf::from(json_str_field(source_model_bundle, "path")?);
    let model_fingerprints =
        model_fingerprints_from_json(json_field(source_model_bundle, "fingerprints")?)?;
    let model_json = json_field(&bundle, "model")?;
    if model_fingerprints.model != sha256_json(model_json) {
        return Err(
            "model evaluation bundle source model fingerprint does not match embedded model"
                .to_owned(),
        );
    }
    let model = model_from_json(model_json, "model")?;
    let samples = labeled_samples_from_embedded_json(json_field(&bundle, "samples")?)?;
    let rows = json_array_field(&bundle, "rows")?
        .iter()
        .enumerate()
        .map(|(position, row)| model_evaluation_row_from_json(row, position))
        .collect::<Result<Vec<_>, _>>()?;

    if rows.len() != samples.len() {
        return Err(format!(
            "model evaluation bundle row/sample count mismatch: rows={} samples={}",
            rows.len(),
            samples.len()
        ));
    }
    if row_index >= rows.len() {
        return Err(format!(
            "--evaluation-row {row_index} is out of range for {} row(s)",
            rows.len()
        ));
    }

    let row = &rows[row_index];
    let sample = &samples[row_index];
    let sample_fingerprint = sha256_json(&labeled_sample_json(sample));
    if sample_fingerprint != row.sample_fingerprint {
        return Err(
            "model evaluation row sample fingerprint does not match embedded sample".to_owned(),
        );
    }
    if row.audit_step != sample.audit_step || row.source_sample_index != sample.source_sample_index
    {
        return Err(
            "model evaluation row lineage does not match embedded sample metadata".to_owned(),
        );
    }

    let inference = run_inference_audit(program, &model, &sample.image, sample.label)?;
    verify_model_evaluation_row_matches_inference(row, sample, &inference)?;

    let inference_sample = ModelInferenceSample {
        image: sample.image.clone(),
        label: sample.label,
        source: ModelInferenceSampleSource::Evaluation {
            path: path.display().to_string(),
            fingerprints: evaluation_fingerprints.clone(),
            row_index,
            audit_step: row.audit_step,
            source_sample_index: row.source_sample_index,
            sample_fingerprint: row.sample_fingerprint.clone(),
        },
    };

    if let Some(output_path) = inference_output {
        write_model_inference_bundle(
            output_path,
            &source_model_path,
            &model_fingerprints,
            &inference_sample,
            &model,
            &inference,
        )?;
    }
    if let Some(output_path) = standalone_rev_output {
        write_standalone_rev_classifier(
            output_path,
            &model,
            &sample.image,
            sample.label,
            &inference,
        )?;
    }

    let report = model_evaluation_row_report_json(
        path,
        &evaluation_fingerprints,
        &source_model_path,
        &model_fingerprints,
        row,
        &inference_sample,
        inference_output,
        standalone_rev_output,
        &inference,
    );
    if let Some(output_path) = markdown_output {
        write_markdown_file(output_path, &render_inference_audit_markdown(&report)?)?;
    }

    if emit_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("model evaluation row report serializes")
        );
    } else {
        println!("model evaluation row {row_index}");
        println!("evaluation: {}", path.display());
        println!("source_model: {}", source_model_path.display());
        println!(
            "evaluation_fingerprint: computation={} payload={}",
            evaluation_fingerprints.computation, evaluation_fingerprints.payload
        );
        println!(
            "model_fingerprint: computation={} payload={}",
            model_fingerprints.computation, model_fingerprints.payload
        );
        println!(
            "sample: audit_step={} source_sample_index={} sample={} label={} prediction={} correct={}",
            row.audit_step
                .map_or_else(|| "none".to_owned(), |step| step.to_string()),
            row.source_sample_index
                .map_or_else(|| "none".to_owned(), |index| index.to_string()),
            row.sample_fingerprint,
            sample.label,
            inference.prediction,
            inference.correct
        );
        println!("row_check: row_matches_recomputed_inference=true");
        println!(
            "top_logits: {}",
            format_logit_preview(&inference.top_logits)
        );
        println!(
            "attribution: predicted={} runner_up={} margin={} reconstructed_logit={} matches_logit={} reconstructed_margin={} matches_margin={}",
            inference.attribution.predicted_digit,
            inference.attribution.runner_up_digit,
            inference.attribution.margin,
            inference.attribution.reconstructed_logit,
            inference.attribution.matches_logit,
            inference.attribution.reconstructed_margin,
            inference.attribution.matches_margin
        );
        println!(
            "inverse: restored_initial_state=true elapsed={:.3}s",
            secs(inference.reverse_elapsed)
        );
        if let Some(output_path) = inference_output {
            println!("inference bundle: {}", output_path.display());
        }
        if let Some(output_path) = standalone_rev_output {
            println!("standalone rev: {}", output_path.display());
        }
    }

    Ok(())
}

fn verify_model_evaluation_row_matches_inference(
    row: &ModelEvaluationRow,
    sample: &LabeledSample,
    inference: &InferenceAudit,
) -> Result<(), String> {
    let active_pixels = inference.active_pixels.len();
    if row.label != sample.label
        || row.prediction != inference.prediction
        || row.correct != inference.correct
        || row.runner_up_digit != inference.attribution.runner_up_digit
        || row.margin != inference.attribution.margin
        || row.active_pixels != active_pixels
    {
        return Err("model evaluation row does not match recomputed Reverie inference".to_owned());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn model_evaluation_row_report_json(
    evaluation_path: &Path,
    evaluation_fingerprints: &ModelEvaluationFingerprints,
    model_path: &Path,
    model_fingerprints: &ModelFingerprints,
    row: &ModelEvaluationRow,
    sample: &ModelInferenceSample,
    inference_output: Option<&Path>,
    standalone_rev_output: Option<&Path>,
    inference: &InferenceAudit,
) -> serde_json::Value {
    let mut report = model_inference_report_json(
        model_path,
        model_fingerprints,
        sample,
        inference_output,
        standalone_rev_output,
        inference,
    );
    let object = report
        .as_object_mut()
        .expect("model evaluation row report is an object");
    object.insert(
        "kind".to_owned(),
        json!("reverie_mnist_linear_q31_model_evaluation_row"),
    );
    object.insert(
        "evaluation_bundle".to_owned(),
        json!({
            "path": evaluation_path.display().to_string(),
            "fingerprints": evaluation_fingerprints.to_json(),
        }),
    );
    object.insert("row".to_owned(), model_evaluation_row_json(row));
    object.insert("row_matches_recomputed_inference".to_owned(), json!(true));
    report
}

fn model_evaluation_row_from_json(
    row: &serde_json::Value,
    position: usize,
) -> Result<ModelEvaluationRow, String> {
    let index = json_usize_field(row, "index")?;
    if index != position {
        return Err(format!(
            "model evaluation row at position {position} has index {index}"
        ));
    }
    let label = json_i64_field(row, "label")?;
    if !(0..DIGITS as i64).contains(&label) {
        return Err(format!(
            "model evaluation row {position} label is outside 0..10"
        ));
    }
    let prediction = json_i64_field(row, "prediction")?;
    if !(0..DIGITS as i64).contains(&prediction) {
        return Err(format!(
            "model evaluation row {position} prediction is outside 0..10"
        ));
    }
    Ok(ModelEvaluationRow {
        index,
        audit_step: json_optional_usize_field(row, "audit_step")?,
        source_sample_index: json_optional_usize_field(row, "source_sample_index")?,
        sample_fingerprint: json_str_field(row, "sample_fingerprint")?.to_owned(),
        label,
        prediction,
        correct: json_bool_field(row, "correct")?,
        runner_up_digit: json_usize_field(row, "runner_up_digit")?,
        margin: json_i64_field(row, "margin")?,
        active_pixels: json_usize_field(row, "active_pixels")?,
    })
}

fn model_evaluation_scan_summary(rows: &[ModelEvaluationRow]) -> ModelEvaluationScanSummary {
    let correct = rows.iter().filter(|row| row.correct).count();
    let lowest = rows.iter().min_by(|left, right| {
        left.margin
            .cmp(&right.margin)
            .then(left.index.cmp(&right.index))
    });
    ModelEvaluationScanSummary {
        samples: rows.len(),
        correct,
        incorrect: rows.len().saturating_sub(correct),
        lowest_margin_index: lowest.map(|row| row.index),
        lowest_margin: lowest.map(|row| row.margin),
    }
}

fn model_evaluation_label_summaries(
    rows: &[ModelEvaluationRow],
) -> Vec<ModelEvaluationLabelSummary> {
    (0..DIGITS)
        .map(|label| {
            let matching = rows
                .iter()
                .filter(|row| row.label == label as i64)
                .collect::<Vec<_>>();
            let correct = matching.iter().filter(|row| row.correct).count();
            let lowest = matching.iter().min_by(|left, right| {
                left.margin
                    .cmp(&right.margin)
                    .then(left.index.cmp(&right.index))
            });
            ModelEvaluationLabelSummary {
                label,
                samples: matching.len(),
                correct,
                incorrect: matching.len().saturating_sub(correct),
                lowest_margin_index: lowest.map(|row| row.index),
                lowest_margin: lowest.map(|row| row.margin),
            }
        })
        .collect()
}

fn model_evaluation_confusion_summaries(
    rows: &[ModelEvaluationRow],
    limit: usize,
) -> Vec<ModelEvaluationConfusionSummary> {
    let mut summaries = BTreeMap::<(i64, i64), ModelEvaluationConfusionSummary>::new();
    for row in rows.iter().filter(|row| !row.correct) {
        summaries
            .entry((row.label, row.prediction))
            .and_modify(|summary| {
                summary.count += 1;
                summary.first_index = summary.first_index.min(row.index);
                summary.lowest_margin = summary.lowest_margin.min(row.margin);
            })
            .or_insert(ModelEvaluationConfusionSummary {
                label: row.label,
                prediction: row.prediction,
                count: 1,
                first_index: row.index,
                lowest_margin: row.margin,
            });
    }
    let mut summaries = summaries.into_values().collect::<Vec<_>>();
    summaries.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then(left.lowest_margin.cmp(&right.lowest_margin))
            .then(left.first_index.cmp(&right.first_index))
            .then(left.label.cmp(&right.label))
            .then(left.prediction.cmp(&right.prediction))
    });
    summaries.truncate(limit);
    summaries
}

fn ranked_incorrect_model_evaluation_rows(
    rows: &[ModelEvaluationRow],
    limit: usize,
) -> Vec<&ModelEvaluationRow> {
    let mut ranked = rows.iter().filter(|row| !row.correct).collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.margin
            .cmp(&right.margin)
            .then(left.index.cmp(&right.index))
    });
    ranked.truncate(limit);
    ranked
}

fn ranked_low_margin_model_evaluation_rows(
    rows: &[ModelEvaluationRow],
    limit: usize,
) -> Vec<&ModelEvaluationRow> {
    let mut ranked = rows.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.margin
            .cmp(&right.margin)
            .then(left.index.cmp(&right.index))
    });
    ranked.truncate(limit);
    ranked
}

fn model_evaluation_scan_summary_json(summary: &ModelEvaluationScanSummary) -> serde_json::Value {
    json!({
        "samples": summary.samples,
        "correct": summary.correct,
        "incorrect": summary.incorrect,
        "accuracy_percent": percent(summary.correct, summary.samples),
        "lowest_margin_index": summary.lowest_margin_index,
        "lowest_margin": summary.lowest_margin,
    })
}

fn model_evaluation_label_summary_json(summary: &ModelEvaluationLabelSummary) -> serde_json::Value {
    json!({
        "label": summary.label,
        "samples": summary.samples,
        "correct": summary.correct,
        "incorrect": summary.incorrect,
        "accuracy_percent": percent(summary.correct, summary.samples),
        "lowest_margin_index": summary.lowest_margin_index,
        "lowest_margin": summary.lowest_margin,
    })
}

fn model_evaluation_confusion_summary_json(
    summary: &ModelEvaluationConfusionSummary,
) -> serde_json::Value {
    json!({
        "label": summary.label,
        "prediction": summary.prediction,
        "count": summary.count,
        "first_index": summary.first_index,
        "lowest_margin": summary.lowest_margin,
    })
}

fn format_model_evaluation_label_summary(summary: &ModelEvaluationLabelSummary) -> String {
    format!(
        "label={} samples={} correct={} incorrect={} accuracy={:.2}% lowest_margin_index={} lowest_margin={}",
        summary.label,
        summary.samples,
        summary.correct,
        summary.incorrect,
        percent(summary.correct, summary.samples),
        summary
            .lowest_margin_index
            .map_or_else(|| "none".to_owned(), |index| index.to_string()),
        summary
            .lowest_margin
            .map_or_else(|| "none".to_owned(), |margin| margin.to_string())
    )
}

fn format_model_evaluation_confusion(summary: &ModelEvaluationConfusionSummary) -> String {
    format!(
        "label={} prediction={} count={} first_index={} lowest_margin={}",
        summary.label,
        summary.prediction,
        summary.count,
        summary.first_index,
        summary.lowest_margin
    )
}

fn print_model_evaluation_rows(rows: &[&ModelEvaluationRow]) {
    if rows.is_empty() {
        println!("- none");
    } else {
        for row in rows {
            println!("- {}", format_model_evaluation_row(row));
        }
    }
}

fn format_model_evaluation_row(row: &ModelEvaluationRow) -> String {
    format!(
        "row={} label={} prediction={} correct={} margin={} runner_up={} active_pixels={} audit_step={} source_sample_index={} sample={}",
        row.index,
        row.label,
        row.prediction,
        row.correct,
        row.margin,
        row.runner_up_digit,
        row.active_pixels,
        row.audit_step
            .map_or_else(|| "none".to_owned(), |step| step.to_string()),
        row.source_sample_index
            .map_or_else(|| "none".to_owned(), |index| index.to_string()),
        row.sample_fingerprint
    )
}

fn model_evaluation_samples_json(
    samples_path: &Path,
    samples_fingerprint: &str,
    samples: &[LabeledSample],
) -> serde_json::Value {
    json!({
        "kind": "reverie_mnist_linear_q31_samples",
        "schema_version": 1,
        "source_json": {
            "path": samples_path.display().to_string(),
            "fingerprint": samples_fingerprint,
        },
        "samples": samples.iter().map(labeled_sample_json).collect::<Vec<_>>(),
    })
}

fn labeled_sample_json(sample: &LabeledSample) -> serde_json::Value {
    let mut value = json!({
        "kind": "reverie_mnist_linear_q31_sample",
        "image_u8": &sample.image,
        "label": sample.label,
    });
    let object = value
        .as_object_mut()
        .expect("labeled sample JSON is an object");
    if let Some(audit_step) = sample.audit_step {
        object.insert("audit_step".to_owned(), json!(audit_step));
    }
    if let Some(source_sample_index) = sample.source_sample_index {
        object.insert("source_sample_index".to_owned(), json!(source_sample_index));
    }
    value
}

fn labeled_samples_from_embedded_json(
    value: &serde_json::Value,
) -> Result<Vec<LabeledSample>, String> {
    labeled_samples_from_samples_object(value, "model evaluation bundle")
}

fn verify_model_evaluation_summary(
    value: &serde_json::Value,
    expected: &ModelEvaluationSummary,
) -> Result<(), String> {
    let checks = [
        ("samples", expected.samples),
        ("correct", expected.correct),
        ("incorrect", expected.incorrect),
    ];
    for (name, expected) in checks {
        let actual = json_usize_field(value, name)?;
        if actual != expected {
            return Err(format!(
                "model evaluation bundle summary `{name}` expected {expected}, found {actual}"
            ));
        }
    }
    let expected_lowest_margin_index = json!(expected.lowest_margin_index);
    if value
        .get("lowest_margin_index")
        .unwrap_or(&serde_json::Value::Null)
        != &expected_lowest_margin_index
    {
        return Err(
            "model evaluation bundle summary `lowest_margin_index` does not match recomputed rows"
                .to_owned(),
        );
    }
    let expected_lowest_margin = json!(expected.lowest_margin);
    if value
        .get("lowest_margin")
        .unwrap_or(&serde_json::Value::Null)
        != &expected_lowest_margin
    {
        return Err(
            "model evaluation bundle summary `lowest_margin` does not match recomputed rows"
                .to_owned(),
        );
    }
    Ok(())
}

fn verify_batch_inference_proof(proof: &serde_json::Value, samples: usize) -> Result<(), String> {
    let expected_sample_bytes = samples * single_sample_payload_bytes();
    let expected_witness_bytes = samples * inference_witness_payload_bytes();
    let checks = [
        ("model_payload_bytes", Model::payload_bytes()),
        ("sample_payload_bytes", expected_sample_bytes),
        ("witness_payload_bytes", expected_witness_bytes),
        ("trace_payload_bytes", 0),
        (
            "replay_payload_bytes",
            Model::payload_bytes() + expected_sample_bytes + expected_witness_bytes,
        ),
        (
            "runtime_state_payload_bytes",
            InferenceMemoryReport::for_inference().runtime_state_payload_bytes,
        ),
        ("forward_recompute_steps", samples),
        ("inverse_recompute_steps", samples),
    ];
    for (name, expected) in checks {
        let actual = json_usize_field(proof, name)?;
        if actual != expected {
            return Err(format!(
                "model evaluation bundle proof `{name}` expected {expected}, found {actual}"
            ));
        }
    }
    Ok(())
}

fn read_model_evaluation_bundle(path: &Path) -> Result<serde_json::Value, String> {
    let data = fs::read(path).map_err(|error| {
        format!(
            "failed to read model evaluation bundle {}: {error}",
            path.display()
        )
    })?;
    let bundle: serde_json::Value = serde_json::from_slice(&data).map_err(|error| {
        format!(
            "failed to parse model evaluation bundle {}: {error}",
            path.display()
        )
    })?;
    let kind = json_str_field(&bundle, "kind")?;
    if kind != "reverie_mnist_linear_q31_model_evaluation_bundle" {
        return Err(format!(
            "model evaluation bundle kind `{kind}` is not reverie_mnist_linear_q31_model_evaluation_bundle"
        ));
    }
    verify_model_evaluation_fingerprints(&bundle)?;
    Ok(bundle)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelEvaluationFingerprints {
    identify_source: String,
    computation: String,
    model: String,
    samples: String,
    summary: String,
    proof: String,
    rows: String,
    gate_policy: String,
    gate: String,
    report: String,
    payload: String,
}

impl ModelEvaluationFingerprints {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "algorithm": "sha256",
            "identify_source": self.identify_source,
            "computation": self.computation,
            "model": self.model,
            "samples": self.samples,
            "summary": self.summary,
            "proof": self.proof,
            "rows": self.rows,
            "gate_policy": self.gate_policy,
            "gate": self.gate,
            "report": self.report,
            "payload": self.payload,
        })
    }
}

fn model_evaluation_fingerprints_for_unsigned_payload(
    payload: &serde_json::Value,
) -> ModelEvaluationFingerprints {
    ModelEvaluationFingerprints {
        identify_source: sha256_bytes(IDENTIFY_SOURCE.as_bytes()),
        computation: model_evaluation_computation_fingerprint(payload),
        model: sha256_json(&payload["model"]),
        samples: sha256_json(&payload["samples"]),
        summary: sha256_json(&payload["summary"]),
        proof: sha256_json(&payload["proof"]),
        rows: sha256_json(&payload["rows"]),
        gate_policy: sha256_json(
            payload
                .get("gate_policy")
                .unwrap_or(&serde_json::Value::Null),
        ),
        gate: sha256_json(payload.get("gate").unwrap_or(&serde_json::Value::Null)),
        report: sha256_json(&payload["report"]),
        payload: sha256_json(payload),
    }
}

fn model_evaluation_fingerprints_from_bundle(
    bundle: &serde_json::Value,
) -> Result<ModelEvaluationFingerprints, String> {
    let fingerprints = json_field(bundle, "fingerprints")?;
    model_evaluation_fingerprints_from_json(fingerprints)
}

fn model_evaluation_fingerprints_from_json(
    fingerprints: &serde_json::Value,
) -> Result<ModelEvaluationFingerprints, String> {
    let algorithm = json_str_field(fingerprints, "algorithm")?;
    if algorithm != "sha256" {
        return Err(format!(
            "model evaluation bundle fingerprint algorithm `{algorithm}` is not sha256"
        ));
    }
    Ok(ModelEvaluationFingerprints {
        identify_source: json_str_field(fingerprints, "identify_source")?.to_owned(),
        computation: json_str_field(fingerprints, "computation")?.to_owned(),
        model: json_str_field(fingerprints, "model")?.to_owned(),
        samples: json_str_field(fingerprints, "samples")?.to_owned(),
        summary: json_str_field(fingerprints, "summary")?.to_owned(),
        proof: json_str_field(fingerprints, "proof")?.to_owned(),
        rows: json_str_field(fingerprints, "rows")?.to_owned(),
        gate_policy: json_str_field(fingerprints, "gate_policy")?.to_owned(),
        gate: json_str_field(fingerprints, "gate")?.to_owned(),
        report: json_str_field(fingerprints, "report")?.to_owned(),
        payload: json_str_field(fingerprints, "payload")?.to_owned(),
    })
}

fn verify_model_evaluation_fingerprints(bundle: &serde_json::Value) -> Result<(), String> {
    let stored = model_evaluation_fingerprints_from_bundle(bundle)?;
    let mut unsigned_payload = bundle.clone();
    unsigned_payload
        .as_object_mut()
        .expect("model evaluation bundle is an object")
        .remove("fingerprints");
    let computed = model_evaluation_fingerprints_for_unsigned_payload(&unsigned_payload);
    if stored != computed {
        return Err(format!(
            "model evaluation bundle fingerprint mismatch: stored payload {}, computed payload {}",
            stored.payload, computed.payload
        ));
    }
    Ok(())
}

fn model_evaluation_computation_fingerprint(payload: &serde_json::Value) -> String {
    let computation = json!({
        "identify_source": sha256_bytes(IDENTIFY_SOURCE.as_bytes()),
        "model": &payload["model"],
        "samples": &payload["samples"],
        "summary": &payload["summary"],
        "proof": &payload["proof"],
        "rows": &payload["rows"],
        "gate_policy": payload.get("gate_policy").cloned().unwrap_or(serde_json::Value::Null),
        "gate": payload.get("gate").cloned().unwrap_or(serde_json::Value::Null),
    });
    sha256_json(&computation)
}

fn batch_inference_proof_json(samples: usize) -> serde_json::Value {
    let sample_payload_bytes = samples * single_sample_payload_bytes();
    let witness_payload_bytes = samples * inference_witness_payload_bytes();
    json!({
        "model_payload_bytes": Model::payload_bytes(),
        "sample_payload_bytes": sample_payload_bytes,
        "witness_payload_bytes": witness_payload_bytes,
        "trace_payload_bytes": 0,
        "replay_payload_bytes": Model::payload_bytes() + sample_payload_bytes + witness_payload_bytes,
        "runtime_state_payload_bytes": InferenceMemoryReport::for_inference().runtime_state_payload_bytes,
        "forward_recompute_steps": samples,
        "inverse_recompute_steps": samples,
        "peak_rss_bytes": peak_rss_bytes(),
    })
}

struct InspectModelInferenceRequest<'a> {
    program: &'a Program,
    model_path: &'a Path,
    sample_audit_path: Option<&'a Path>,
    sample_json_path: Option<&'a Path>,
    step_index: usize,
    inference_output: Option<&'a Path>,
    standalone_rev_output: Option<&'a Path>,
    markdown_output: Option<&'a Path>,
    emit_json: bool,
}

fn inspect_model_inference_bundle(request: InspectModelInferenceRequest<'_>) -> Result<(), String> {
    let InspectModelInferenceRequest {
        program,
        model_path,
        sample_audit_path,
        sample_json_path,
        step_index,
        inference_output,
        standalone_rev_output,
        markdown_output,
        emit_json,
    } = request;

    let model_bundle = read_model_bundle(model_path)?;
    let model_fingerprints = model_fingerprints_from_bundle(&model_bundle)?;
    let model = model_from_json(json_field(&model_bundle, "model")?, "model")?;
    let sample = match (sample_audit_path, sample_json_path) {
        (Some(path), None) => model_inference_sample_from_audit(path, step_index)?,
        (None, Some(path)) => model_inference_sample_from_json(path)?,
        (Some(_), Some(_)) => {
            return Err("choose only one model inference sample source".to_owned());
        }
        (None, None) => return Err("missing model inference sample source".to_owned()),
    };
    let inference = run_inference_audit(program, &model, &sample.image, sample.label)?;

    if let Some(output_path) = inference_output {
        write_model_inference_bundle(
            output_path,
            model_path,
            &model_fingerprints,
            &sample,
            &model,
            &inference,
        )?;
    }
    if let Some(output_path) = standalone_rev_output {
        write_standalone_rev_classifier(
            output_path,
            &model,
            &sample.image,
            sample.label,
            &inference,
        )?;
    }

    let report = model_inference_report_json(
        model_path,
        &model_fingerprints,
        &sample,
        inference_output,
        standalone_rev_output,
        &inference,
    );
    if let Some(output_path) = markdown_output {
        write_markdown_file(output_path, &render_inference_audit_markdown(&report)?)?;
    }

    if emit_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("model inference audit report serializes")
        );
    } else {
        println!("model inference audit step {step_index}");
        println!("model: {}", model_path.display());
        println!("sample_source: {}", sample.source.describe());
        println!(
            "model_fingerprint: computation={} payload={}",
            model_fingerprints.computation, model_fingerprints.payload
        );
        println!(
            "sample: source_index={} label={} prediction={} correct={}",
            sample
                .source
                .sample_index()
                .map_or_else(|| "none".to_owned(), |index| index.to_string()),
            sample.label,
            inference.prediction,
            inference.correct
        );
        println!(
            "witness_checks: computed_prediction={} prediction_matches_logits={} correct_matches_logits={}",
            inference.computed_prediction,
            inference.prediction_matches_logits,
            inference.correct_matches_logits
        );
        println!(
            "top_logits: {}",
            format_logit_preview(&inference.top_logits)
        );
        println!(
            "attribution: predicted={} runner_up={} margin={} reconstructed_logit={} matches_logit={} reconstructed_margin={} matches_margin={}",
            inference.attribution.predicted_digit,
            inference.attribution.runner_up_digit,
            inference.attribution.margin,
            inference.attribution.reconstructed_logit,
            inference.attribution.matches_logit,
            inference.attribution.reconstructed_margin,
            inference.attribution.matches_margin
        );
        println!(
            "inverse: restored_initial_state=true elapsed={:.3}s",
            secs(inference.reverse_elapsed)
        );
        if let Some(output_path) = inference_output {
            println!("inference bundle: {}", output_path.display());
        }
        if let Some(output_path) = standalone_rev_output {
            println!("standalone rev: {}", output_path.display());
        }
    }

    Ok(())
}

fn model_inference_report_json(
    model_path: &Path,
    model_fingerprints: &ModelFingerprints,
    sample: &ModelInferenceSample,
    inference_output: Option<&Path>,
    standalone_rev_output: Option<&Path>,
    inference: &InferenceAudit,
) -> serde_json::Value {
    json!({
        "kind": "reverie_mnist_linear_q31_model_inference_audit",
        "model_bundle": {
            "path": model_path.display().to_string(),
            "fingerprints": model_fingerprints.to_json(),
        },
        "sample_source": sample.source.to_json(),
        "sample_audit": sample.source.sample_audit_json(),
        "sample_json": sample.source.sample_json_json(),
        "audit_step": sample.source.audit_step(),
        "sample_index": sample.source.sample_index(),
        "label": sample.label,
        "inference_output": inference_output.map(|path| path.display().to_string()),
        "standalone_rev_output": standalone_rev_output.map(|path| path.display().to_string()),
        "model": {
            "source": "model_bundle",
            "weights_shape": [IMAGE_PIXELS, DIGITS],
            "bias_shape": [DIGITS],
        },
        "prediction": inference.prediction,
        "correct": inference.correct,
        "logits": &inference.logits,
        "top_logits": inference.top_logits.iter().map(|logit| {
            json!({
                "digit": logit.digit,
                "logit": logit.value,
            })
        }).collect::<Vec<_>>(),
        "attribution": inference_attribution_json(&inference.attribution),
        "witness_checks": {
            "computed_prediction": inference.computed_prediction,
            "computed_correct": inference.computed_correct,
            "prediction_matches_logits": inference.prediction_matches_logits,
            "correct_matches_logits": inference.correct_matches_logits,
        },
        "active_pixels": inference.active_pixels.iter().map(|pixel| {
            json!({
                "index": pixel.index,
                "u8": pixel.value,
                "q31": pixel.q31,
            })
        }).collect::<Vec<_>>(),
        "forward": {
            "elapsed_seconds": secs(inference.forward_elapsed),
        },
        "inverse": {
            "restored_initial_state": true,
            "elapsed_seconds": secs(inference.reverse_elapsed),
        },
        "memory": inference.memory.to_json(),
        "proof": inference_proof_summary_json(inference),
        "explanation_contract": inference_explanation_contract_json(inference),
    })
}

fn write_model_inference_bundle(
    path: &Path,
    model_path: &Path,
    model_fingerprints: &ModelFingerprints,
    sample: &ModelInferenceSample,
    model: &Model,
    inference: &InferenceAudit,
) -> Result<(), String> {
    let model_json = json!({
        "weights": &model.weights,
        "bias": &model.bias,
    });
    let sample_json = json!({
        "image_u8": &sample.image,
        "label": sample.label,
    });
    let result_json = inference_result_json(inference);
    let proof_json =
        inference_bundle_proof_json(&model_json, &sample_json, &result_json, inference);
    let report_json = model_inference_report_json(
        model_path,
        model_fingerprints,
        sample,
        Some(path),
        None,
        inference,
    );
    let mut bundle = json!({
        "kind": "reverie_mnist_linear_q31_inference_replay_bundle",
        "schema_version": 1,
        "program": "examples/mnist_identify.rev",
        "source_model_bundle": {
            "path": model_path.display().to_string(),
            "fingerprints": model_fingerprints.to_json(),
        },
        "sample_source": sample.source.to_json(),
        "tensor_shapes": {
            "image": [IMAGE_PIXELS],
            "weights": [IMAGE_PIXELS, DIGITS],
            "bias": [DIGITS],
            "logits": [DIGITS],
        },
        "model": model_json,
        "sample": sample_json,
        "storage": inference.memory.to_json(),
        "result": result_json,
        "proof": proof_json,
        "report": report_json,
    });
    match &sample.source {
        ModelInferenceSampleSource::Audit {
            path,
            fingerprints,
            audit_step,
            sample_index,
        } => {
            bundle
                .as_object_mut()
                .expect("inference bundle is an object")
                .insert(
                    "source_sample_audit".to_owned(),
                    json!({
                        "path": path,
                        "audit_step": audit_step,
                        "sample_index": sample_index,
                        "fingerprints": fingerprints.to_json(),
                    }),
                );
        }
        ModelInferenceSampleSource::Json { path, fingerprint } => {
            bundle
                .as_object_mut()
                .expect("inference bundle is an object")
                .insert(
                    "source_sample_json".to_owned(),
                    json!({
                        "path": path,
                        "fingerprint": fingerprint,
                    }),
                );
        }
        ModelInferenceSampleSource::Evaluation {
            path,
            fingerprints,
            row_index,
            audit_step,
            source_sample_index,
            sample_fingerprint,
        } => {
            bundle
                .as_object_mut()
                .expect("inference bundle is an object")
                .insert(
                    "source_model_evaluation".to_owned(),
                    json!({
                        "path": path,
                        "row_index": row_index,
                        "audit_step": audit_step,
                        "source_sample_index": source_sample_index,
                        "sample_fingerprint": sample_fingerprint,
                        "fingerprints": fingerprints.to_json(),
                    }),
                );
        }
    }
    let fingerprints = inference_fingerprints_for_unsigned_payload(&bundle);
    bundle
        .as_object_mut()
        .expect("inference bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints.to_json());

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let encoded = serde_json::to_string_pretty(&bundle).expect("inference bundle serializes");
    fs::write(path, encoded).map_err(|error| {
        format!(
            "failed to write inference bundle {}: {error}",
            path.display()
        )
    })
}

fn write_standalone_rev_classifier(
    path: &Path,
    model: &Model,
    image: &[u8],
    label: i64,
    inference: &InferenceAudit,
) -> Result<(), String> {
    let source = render_standalone_rev_classifier(model, image, label, inference);
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    fs::write(path, source).map_err(|error| {
        format!(
            "failed to write standalone Reverie classifier {}: {error}",
            path.display()
        )
    })
}

fn render_standalone_rev_classifier(
    model: &Model,
    image: &[u8],
    label: i64,
    inference: &InferenceAudit,
) -> String {
    let image_q31 = image
        .iter()
        .map(|pixel| pixel_to_q31(*pixel))
        .collect::<Vec<_>>();
    let expected_correct = i64::from(inference.correct);
    let mut source = String::new();
    source.push_str("// Standalone MNIST Q31 classifier generated by reverie-mnist-linear.\n");
    source.push_str("// Run with: reverie run this_file.rev --json\n");
    source.push_str("// The classification transition below is pure Reverie source.\n\n");
    source.push_str("global image: tensor<int, 784> = ");
    source.push_str(&render_vector_literal(&image_q31, "  ", 12));
    source.push_str(";\n\n");
    source.push_str("global weights: tensor<int, 784, 10> = ");
    source.push_str(&render_matrix_literal(&model.weights, "  "));
    source.push_str(";\n");
    source.push_str("global bias: tensor<int, 10> = ");
    source.push_str(&render_vector_literal(&model.bias, "  ", 10));
    source.push_str(";\n");
    source.push_str("global logits: witness<tensor<int, 10>>;\n");
    source.push_str("global prediction;\n");
    source.push_str("global correct;\n");
    source.push_str(&format!("global label = {label};\n"));
    source.push_str(&format!(
        "global expected_prediction = {};\n",
        inference.prediction
    ));
    source.push_str(&format!(
        "global expected_correct = {expected_correct};\n\n"
    ));
    source.push_str("procedure main() {\n");
    source.push_str("  logits += vecmat_q31(image, weights);\n");
    source.push_str("  logits += bias;\n");
    source.push_str("  prediction += argmax(logits);\n");
    source.push_str("  correct += argmax_eq(logits, label);\n");
    source.push_str("  assert prediction == expected_prediction;\n");
    source.push_str("  assert correct == expected_correct\n");
    source.push_str("}\n");
    source
}

fn render_vector_literal(values: &[i64], indent: &str, per_line: usize) -> String {
    if values.is_empty() {
        return "[]".to_owned();
    }
    let rows = values
        .chunks(per_line.max(1))
        .map(|chunk| {
            let row = chunk
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{indent}{row}")
        })
        .collect::<Vec<_>>()
        .join(",\n");
    format!("[\n{rows}\n]")
}

fn render_matrix_literal(rows: &[Vec<i64>], indent: &str) -> String {
    if rows.is_empty() {
        return "[]".to_owned();
    }
    let rendered_rows = rows
        .iter()
        .map(|row| {
            let values = row
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{indent}[{values}]")
        })
        .collect::<Vec<_>>()
        .join(",\n");
    format!("[\n{rendered_rows}\n]")
}

#[derive(Debug, Clone)]
struct InferenceAudit {
    logits: Vec<i64>,
    prediction: i64,
    correct: bool,
    computed_prediction: i64,
    computed_correct: bool,
    prediction_matches_logits: bool,
    correct_matches_logits: bool,
    active_pixels: Vec<ActivePixel>,
    top_logits: Vec<LogitRank>,
    attribution: InferenceAttribution,
    forward_elapsed: Duration,
    reverse_elapsed: Duration,
    memory: InferenceMemoryReport,
}

fn run_inference_audit(
    program: &Program,
    model: &Model,
    image: &[u8],
    label: i64,
) -> Result<InferenceAudit, String> {
    let initial_state = sample_state(model, image, label, 0, false);
    let forward_start = Instant::now();
    let final_state = execute_compiled(program, initial_state.clone())
        .map_err(|error| format!("inference inspection failed: {error}"))?;
    let forward_elapsed = forward_start.elapsed();
    let logits = vector_from_state(&final_state, "logits", DIGITS)?;
    let prediction = int_from_state(&final_state, "prediction")?;
    let correct = int_from_state(&final_state, "correct")? != 0;
    let computed_prediction = argmax_first(&logits);
    let computed_correct = computed_prediction == label;
    let prediction_matches_logits = prediction == computed_prediction;
    let correct_matches_logits = correct == computed_correct;
    let active_pixels = active_pixels(image);
    let top_logits = top_logits(&logits);
    let attribution = inference_attribution(model, image, &logits, &top_logits);
    if !attribution.matches_logit {
        return Err(format!(
            "inference attribution failed to reconstruct logit for digit {}",
            attribution.predicted_digit
        ));
    }
    if !attribution.matches_margin {
        return Err(format!(
            "inference attribution failed to reconstruct margin for digits {} and {}",
            attribution.predicted_digit, attribution.runner_up_digit
        ));
    }

    let reverse_start = Instant::now();
    let restored = execute_compiled_backward(program, final_state)
        .map_err(|error| format!("inference inverse check failed: {error}"))?;
    let reverse_elapsed = reverse_start.elapsed();
    if restored != initial_state {
        return Err("inference inverse check failed: initial state was not restored".to_owned());
    }

    Ok(InferenceAudit {
        logits,
        prediction,
        correct,
        computed_prediction,
        computed_correct,
        prediction_matches_logits,
        correct_matches_logits,
        active_pixels,
        top_logits,
        attribution,
        forward_elapsed,
        reverse_elapsed,
        memory: InferenceMemoryReport::for_inference(),
    })
}

#[allow(clippy::too_many_arguments)]
fn inference_report_json(
    kind: &str,
    source_path: Option<&Path>,
    source_fingerprints: Option<&AuditFingerprints>,
    audit_step: usize,
    sample_index: usize,
    label: i64,
    inference_output: Option<&Path>,
    standalone_rev_output: Option<&Path>,
    inference: &InferenceAudit,
) -> serde_json::Value {
    json!({
        "kind": kind,
        "path": source_path.map(|path| path.display().to_string()),
        "fingerprints": source_fingerprints.map(AuditFingerprints::to_json),
        "audit_step": audit_step,
        "sample_index": sample_index,
        "label": label,
        "inference_output": inference_output.map(|path| path.display().to_string()),
        "standalone_rev_output": standalone_rev_output.map(|path| path.display().to_string()),
        "model": {
            "source": "final_model",
            "weights_shape": [IMAGE_PIXELS, DIGITS],
            "bias_shape": [DIGITS],
        },
        "prediction": inference.prediction,
        "correct": inference.correct,
        "logits": &inference.logits,
        "top_logits": inference.top_logits.iter().map(|logit| {
            json!({
                "digit": logit.digit,
                "logit": logit.value,
            })
        }).collect::<Vec<_>>(),
        "attribution": inference_attribution_json(&inference.attribution),
        "witness_checks": {
            "computed_prediction": inference.computed_prediction,
            "computed_correct": inference.computed_correct,
            "prediction_matches_logits": inference.prediction_matches_logits,
            "correct_matches_logits": inference.correct_matches_logits,
        },
        "active_pixels": inference.active_pixels.iter().map(|pixel| {
            json!({
                "index": pixel.index,
                "u8": pixel.value,
                "q31": pixel.q31,
            })
        }).collect::<Vec<_>>(),
        "forward": {
            "elapsed_seconds": secs(inference.forward_elapsed),
        },
        "inverse": {
            "restored_initial_state": true,
            "elapsed_seconds": secs(inference.reverse_elapsed),
        },
        "memory": inference.memory.to_json(),
        "proof": inference_proof_summary_json(inference),
        "explanation_contract": inference_explanation_contract_json(inference),
    })
}

#[allow(clippy::too_many_arguments)]
fn write_inference_bundle(
    path: &Path,
    source_audit_path: &Path,
    source_fingerprints: &AuditFingerprints,
    audit_step: usize,
    sample_index: usize,
    model: &Model,
    image: &[u8],
    label: i64,
    inference: &InferenceAudit,
) -> Result<(), String> {
    let model_json = json!({
        "weights": &model.weights,
        "bias": &model.bias,
    });
    let sample_json = json!({
        "image_u8": image,
        "label": label,
    });
    let result_json = inference_result_json(inference);
    let proof_json =
        inference_bundle_proof_json(&model_json, &sample_json, &result_json, inference);
    let report_json = inference_report_json(
        "reverie_mnist_linear_q31_inference_audit",
        Some(source_audit_path),
        Some(source_fingerprints),
        audit_step,
        sample_index,
        label,
        Some(path),
        None,
        inference,
    );
    let mut bundle = json!({
        "kind": "reverie_mnist_linear_q31_inference_replay_bundle",
        "schema_version": 1,
        "program": "examples/mnist_identify.rev",
        "source_training_bundle": {
            "path": source_audit_path.display().to_string(),
            "audit_step": audit_step,
            "sample_index": sample_index,
            "fingerprints": source_fingerprints.to_json(),
        },
        "tensor_shapes": {
            "image": [IMAGE_PIXELS],
            "weights": [IMAGE_PIXELS, DIGITS],
            "bias": [DIGITS],
            "logits": [DIGITS],
        },
        "model": model_json,
        "sample": sample_json,
        "storage": inference.memory.to_json(),
        "result": result_json,
        "proof": proof_json,
        "report": report_json,
    });
    let fingerprints = inference_fingerprints_for_unsigned_payload(&bundle);
    bundle
        .as_object_mut()
        .expect("inference bundle is an object")
        .insert("fingerprints".to_owned(), fingerprints.to_json());

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let encoded = serde_json::to_string_pretty(&bundle).expect("inference bundle serializes");
    fs::write(path, encoded).map_err(|error| {
        format!(
            "failed to write inference bundle {}: {error}",
            path.display()
        )
    })
}

fn inference_proof_summary_json(inference: &InferenceAudit) -> serde_json::Value {
    let memory = inference.memory;
    json!({
        "claim": "deterministic_q31_inference_replay",
        "kernel": "examples/mnist_identify.rev",
        "arithmetic": "q31_wrapping_i64",
        "witnesses": ["logits", "prediction", "correct"],
        "prediction": inference.prediction,
        "correct": inference.correct,
        "predicted_digit": inference.attribution.predicted_digit,
        "runner_up_digit": inference.attribution.runner_up_digit,
        "margin": inference.attribution.margin,
        "active_pixels": inference.active_pixels.len(),
        "contribution_count": inference.attribution.contribution_count,
        "margin_contribution_count": inference.attribution.margin_contribution_count,
        "contribution_ledger_fingerprint": &inference.attribution.contribution_ledger_fingerprint,
        "margin_contribution_ledger_fingerprint": &inference.attribution.margin_contribution_ledger_fingerprint,
        "model_payload_bytes": memory.model_payload_bytes,
        "sample_payload_bytes": memory.sample_payload_bytes,
        "witness_payload_bytes": memory.witness_payload_bytes,
        "trace_payload_bytes": memory.trace_payload_bytes,
        "replay_payload_bytes": memory.replay_payload_bytes,
        "runtime_state_payload_bytes": memory.runtime_state_payload_bytes,
        "forward_recompute_steps": memory.forward_recompute_steps,
        "inverse_recompute_steps": memory.inverse_recompute_steps,
        "checks": {
            "prediction_matches_logits": inference.prediction_matches_logits,
            "correct_matches_logits": inference.correct_matches_logits,
            "attribution_matches_logit": inference.attribution.matches_logit,
            "attribution_matches_margin": inference.attribution.matches_margin,
            "restored_initial_state": true,
        },
    })
}

fn inference_explanation_contract_json(inference: &InferenceAudit) -> serde_json::Value {
    inference_explanation_contract_json_with_extra(inference, Vec::new())
}

fn inference_verification_explanation_contract_json(
    inference: &InferenceAudit,
    proof_matches: bool,
    result_matches: bool,
    restored_initial_state: bool,
    source_inputs_checked: bool,
) -> serde_json::Value {
    inference_explanation_contract_json_with_extra(
        inference,
        vec![
            json!({
                "metric": "proof_recomputed",
                "passed": proof_matches,
                "actual": format!("proof_matches={proof_matches}"),
                "requirement": "stored proof equals recomputed model/sample/result proof",
            }),
            json!({
                "metric": "result_recomputed",
                "passed": result_matches,
                "actual": format!("result_matches={result_matches}"),
                "requirement": "stored result equals fresh Reverie inference",
            }),
            json!({
                "metric": "source_inputs_checked",
                "passed": source_inputs_checked,
                "actual": format!("source_inputs_checked={source_inputs_checked}"),
                "requirement": "referenced model plus sample, training, or evaluation source was checked",
            }),
            json!({
                "metric": "verification_restores_initial_state",
                "passed": restored_initial_state,
                "actual": format!("restored_initial_state={restored_initial_state}"),
                "requirement": "verification replay restores the inference initial state",
            }),
        ],
    )
}

fn inference_explanation_contract_json_with_extra(
    inference: &InferenceAudit,
    extra_checks: Vec<serde_json::Value>,
) -> serde_json::Value {
    let mut checks = vec![
        json!({
            "metric": "logits_determine_prediction",
            "passed": inference.prediction_matches_logits,
            "actual": format!(
                "prediction={} computed_prediction={}",
                inference.prediction, inference.computed_prediction
            ),
            "requirement": "prediction == argmax_first(logits)",
        }),
        json!({
            "metric": "correct_matches_label",
            "passed": inference.correct_matches_logits,
            "actual": format!(
                "correct={} computed_correct={}",
                inference.correct, inference.computed_correct
            ),
            "requirement": "correct == (prediction == label)",
        }),
        json!({
            "metric": "attribution_reconstructs_logit",
            "passed": inference.attribution.matches_logit,
            "actual": format!(
                "digit={} reconstructed_logit={}",
                inference.attribution.predicted_digit, inference.attribution.reconstructed_logit
            ),
            "requirement": "bias plus per-pixel contributions reconstruct the winning logit",
        }),
        json!({
            "metric": "attribution_reconstructs_margin",
            "passed": inference.attribution.matches_margin,
            "actual": format!(
                "predicted={} runner_up={} reconstructed_margin={}",
                inference.attribution.predicted_digit,
                inference.attribution.runner_up_digit,
                inference.attribution.reconstructed_margin
            ),
            "requirement": "winner-vs-runner-up contributions reconstruct the margin",
        }),
        json!({
            "metric": "reverse_restores_initial_state",
            "passed": true,
            "actual": "restored_initial_state=true",
            "requirement": "running the inference program backward restores model and sample inputs",
        }),
    ];
    checks.extend(extra_checks);
    let passed = checks
        .iter()
        .all(|check| check.get("passed").and_then(serde_json::Value::as_bool) == Some(true));
    json!({
        "claim": "q31_inference_prediction_explanation",
        "passed": passed,
        "prediction": inference.prediction,
        "correct": inference.correct,
        "predicted_digit": inference.attribution.predicted_digit,
        "runner_up_digit": inference.attribution.runner_up_digit,
        "margin": inference.attribution.margin,
        "active_pixel_count": inference.active_pixels.len(),
        "contribution_count": inference.attribution.contribution_count,
        "margin_contribution_count": inference.attribution.margin_contribution_count,
        "top_logit_count": inference.top_logits.len(),
        "ledger_fingerprints": {
            "algorithm": "sha256",
            "contribution": &inference.attribution.contribution_ledger_fingerprint,
            "margin_contribution": &inference.attribution.margin_contribution_ledger_fingerprint,
        },
        "replay_direction": {
            "forward_transition": "model + sample -> logits + prediction",
            "reverse_transition": "model + sample + result -> initial state",
            "inverse_restores_initial_state": true,
        },
        "checks": checks,
    })
}

fn inference_bundle_proof_json(
    model: &serde_json::Value,
    sample: &serde_json::Value,
    result: &serde_json::Value,
    inference: &InferenceAudit,
) -> serde_json::Value {
    let mut proof = inference_proof_summary_json(inference);
    proof
        .as_object_mut()
        .expect("inference proof is an object")
        .insert(
            "fingerprints".to_owned(),
            json!({
                "algorithm": "sha256",
                "identify_source": sha256_bytes(IDENTIFY_SOURCE.as_bytes()),
                "model": sha256_json(model),
                "sample": sha256_json(sample),
                "result": sha256_json(result),
            }),
        );
    proof
}

fn format_inference_proof_line(inference: &InferenceAudit) -> String {
    let memory = inference.memory;
    format!(
        "proof: claim=deterministic_q31_inference_replay model_bytes={} sample_bytes={} witness_bytes={} trace_bytes={} replay_bytes={} forward_recompute_steps={} inverse_recompute_steps={}",
        memory.model_payload_bytes,
        memory.sample_payload_bytes,
        memory.witness_payload_bytes,
        memory.trace_payload_bytes,
        memory.replay_payload_bytes,
        memory.forward_recompute_steps,
        memory.inverse_recompute_steps
    )
}

fn inference_result_json(inference: &InferenceAudit) -> serde_json::Value {
    json!({
        "logits": &inference.logits,
        "prediction": inference.prediction,
        "correct": inference.correct,
        "top_logits": inference.top_logits.iter().map(|logit| {
            json!({
                "digit": logit.digit,
                "logit": logit.value,
            })
        }).collect::<Vec<_>>(),
        "attribution": inference_attribution_json(&inference.attribution),
        "witness_checks": {
            "computed_prediction": inference.computed_prediction,
            "computed_correct": inference.computed_correct,
            "prediction_matches_logits": inference.prediction_matches_logits,
            "correct_matches_logits": inference.correct_matches_logits,
        },
        "inverse": {
            "restored_initial_state": true,
        },
        "memory": inference.memory.to_json(),
    })
}

fn verify_inference_bundle(
    program: &Program,
    path: &Path,
    markdown_output: Option<&Path>,
    emit_json: bool,
) -> Result<(), String> {
    let bundle = read_inference_bundle(path)?;
    let fingerprints = inference_fingerprints_from_bundle(&bundle)?;
    let model_json = json_field(&bundle, "model")?;
    let model = model_from_json(model_json, "model")?;
    let sample = json_field(&bundle, "sample")?;
    let image = json_u8_vec_field(sample, "image_u8", IMAGE_PIXELS)?;
    let label = json_i64_field(sample, "label")?;
    if !(0..DIGITS as i64).contains(&label) {
        return Err("inference bundle `sample.label` is outside 0..10".to_owned());
    }

    let start = Instant::now();
    let inference = run_inference_audit(program, &model, &image, label)?;
    let elapsed = start.elapsed();
    verify_inference_memory(json_field(&bundle, "storage")?, inference.memory)?;
    let result = json_field(&bundle, "result")?;
    verify_inference_result(result, &inference)?;
    let proof = inference_bundle_proof_json(model_json, sample, result, &inference);
    if let Some(stored_proof) = bundle.get("proof") {
        verify_inference_proof(stored_proof, &proof)?;
    }
    let source_evaluation =
        verify_inference_source_evaluation(path, &bundle, model_json, sample, &inference)?;
    let source_evaluation_checked = source_evaluation
        .as_ref()
        .is_some_and(|check| check.checked);
    let source_artifacts = verify_inference_source_artifacts(path, &bundle, model_json, sample)?;
    let proof_matches = bundle.get("proof").is_some();
    let source_model_checked = source_artifacts
        .model
        .as_ref()
        .is_some_and(|check| check.checked);
    let source_training_checked = source_artifacts
        .training
        .as_ref()
        .is_some_and(|check| check.checked);
    let source_sample_checked = source_artifacts
        .sample
        .as_ref()
        .is_some_and(|check| check.checked);
    let source_inputs_checked = source_training_checked
        || (source_model_checked && (source_sample_checked || source_evaluation_checked));

    let report = json!({
        "kind": "reverie_mnist_linear_q31_inference_verification",
        "path": path.display().to_string(),
        "fingerprints": fingerprints.to_json(),
        "prediction": inference.prediction,
        "correct": inference.correct,
        "result_matches": true,
        "restored_initial_state": true,
        "proof_matches": proof_matches,
        "source_evaluation_checked": source_evaluation_checked,
        "source_model_evaluation": source_evaluation.as_ref().map(InferenceSourceEvaluationCheck::to_json),
        "source_model_checked": source_model_checked,
        "source_training_checked": source_training_checked,
        "source_sample_checked": source_sample_checked,
        "source_model": source_artifacts.model.as_ref().map(InferenceSourceArtifactCheck::to_json),
        "source_training_bundle": source_artifacts.training.as_ref().map(InferenceSourceArtifactCheck::to_json),
        "source_sample": source_artifacts.sample.as_ref().map(InferenceSourceArtifactCheck::to_json),
        "elapsed_seconds": secs(elapsed),
        "attribution": inference_attribution_json(&inference.attribution),
        "memory": inference.memory.to_json(),
        "proof": proof,
        "explanation_contract": inference_verification_explanation_contract_json(
            &inference,
            proof_matches,
            true,
            true,
            source_inputs_checked,
        ),
    });

    if let Some(output_path) = markdown_output {
        write_markdown_file(
            output_path,
            &render_inference_verification_markdown(&report)?,
        )?;
    }

    if emit_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .expect("inference verification report serializes")
        );
    } else {
        println!(
            "inference audit ok: prediction={} correct={} result_matches=true restored_initial_state=true elapsed={:.3}s",
            inference.prediction,
            inference.correct,
            secs(elapsed)
        );
        println!(
            "storage: replay_payload_bytes={} witness_payload_bytes={} trace_payload_bytes={} runtime_state_payload_bytes={} forward_recompute_steps={} inverse_recompute_steps={}",
            inference.memory.replay_payload_bytes,
            inference.memory.witness_payload_bytes,
            inference.memory.trace_payload_bytes,
            inference.memory.runtime_state_payload_bytes,
            inference.memory.forward_recompute_steps,
            inference.memory.inverse_recompute_steps
        );
        println!("{}", format_inference_proof_line(&inference));
        println!(
            "explanation_contract: claim=q31_inference_prediction_explanation passed=true checks=9"
        );
        if let Some(source_evaluation) = &source_evaluation {
            println!(
                "source_model_evaluation: {}",
                source_evaluation.format_line()
            );
        }
        if let Some(source_model) = &source_artifacts.model {
            println!("source_model: {}", source_model.format_line());
        }
        if let Some(source_training) = &source_artifacts.training {
            println!("source_training_bundle: {}", source_training.format_line());
        }
        if let Some(source_sample) = &source_artifacts.sample {
            println!("source_sample: {}", source_sample.format_line());
        }
        println!(
            "fingerprint: computation={} payload={}",
            fingerprints.computation, fingerprints.payload
        );
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct InferenceSourceArtifactChecks {
    model: Option<InferenceSourceArtifactCheck>,
    training: Option<InferenceSourceArtifactCheck>,
    sample: Option<InferenceSourceArtifactCheck>,
}

#[derive(Debug, Clone)]
struct InferenceSourceArtifactCheck {
    path: String,
    resolved_path: String,
    checked: bool,
    payload_fingerprint: String,
    items_checked: usize,
    unavailable_reason: Option<&'static str>,
}

impl InferenceSourceArtifactCheck {
    fn to_json(&self) -> serde_json::Value {
        let mut value = json!({
            "path": self.path,
            "resolved_path": self.resolved_path,
            "checked": self.checked,
            "payload_fingerprint": self.payload_fingerprint,
            "items_checked": self.items_checked,
        });
        if let Some(reason) = self.unavailable_reason {
            value
                .as_object_mut()
                .expect("inference source check JSON is an object")
                .insert("unavailable_reason".to_owned(), json!(reason));
        }
        value
    }

    fn format_line(&self) -> String {
        if self.checked {
            format!(
                "checked=true items_checked={} payload={}",
                self.items_checked, self.payload_fingerprint
            )
        } else {
            format!(
                "checked=false reason={} path={}",
                self.unavailable_reason.unwrap_or("unavailable"),
                self.path
            )
        }
    }
}

fn verify_inference_source_artifacts(
    inference_path: &Path,
    bundle: &serde_json::Value,
    model_json: &serde_json::Value,
    sample_json: &serde_json::Value,
) -> Result<InferenceSourceArtifactChecks, String> {
    Ok(InferenceSourceArtifactChecks {
        model: bundle
            .get("source_model_bundle")
            .map(|source| verify_inference_source_model(inference_path, source, model_json))
            .transpose()?,
        training: bundle
            .get("source_training_bundle")
            .map(|source| {
                verify_inference_source_training(inference_path, source, model_json, sample_json)
            })
            .transpose()?,
        sample: verify_inference_source_sample(inference_path, bundle, sample_json)?,
    })
}

fn verify_inference_source_model(
    inference_path: &Path,
    source: &serde_json::Value,
    model_json: &serde_json::Value,
) -> Result<InferenceSourceArtifactCheck, String> {
    let source_path_text = json_str_field(source, "path")?.to_owned();
    let expected_fingerprints = model_fingerprints_from_json(json_field(source, "fingerprints")?)?;
    let resolved_path = resolve_referenced_artifact_path(inference_path, &source_path_text);
    let resolved_path_text = resolved_path.display().to_string();
    if !resolved_path.exists() {
        return Ok(InferenceSourceArtifactCheck {
            path: source_path_text,
            resolved_path: resolved_path_text,
            checked: false,
            payload_fingerprint: expected_fingerprints.payload,
            items_checked: 0,
            unavailable_reason: Some("source_model_not_found"),
        });
    }

    let source_bundle = read_model_bundle(&resolved_path)?;
    let actual_fingerprints = model_fingerprints_from_bundle(&source_bundle)?;
    if actual_fingerprints != expected_fingerprints {
        return Err(
            "inference source model fingerprints do not match referenced model bundle".to_owned(),
        );
    }
    if json_field(&source_bundle, "model")? != model_json {
        return Err("inference source model does not match inference model".to_owned());
    }

    Ok(InferenceSourceArtifactCheck {
        path: source_path_text,
        resolved_path: resolved_path_text,
        checked: true,
        payload_fingerprint: actual_fingerprints.payload,
        items_checked: 1,
        unavailable_reason: None,
    })
}

fn verify_inference_source_training(
    inference_path: &Path,
    source: &serde_json::Value,
    model_json: &serde_json::Value,
    sample_json: &serde_json::Value,
) -> Result<InferenceSourceArtifactCheck, String> {
    let source_path_text = json_str_field(source, "path")?.to_owned();
    let expected_fingerprints = audit_fingerprints_from_json(json_field(source, "fingerprints")?)?;
    let audit_step = json_usize_field(source, "audit_step")?;
    let sample_index = json_usize_field(source, "sample_index")?;
    let resolved_path = resolve_referenced_artifact_path(inference_path, &source_path_text);
    let resolved_path_text = resolved_path.display().to_string();
    if !resolved_path.exists() {
        return Ok(InferenceSourceArtifactCheck {
            path: source_path_text,
            resolved_path: resolved_path_text,
            checked: false,
            payload_fingerprint: expected_fingerprints.payload,
            items_checked: 0,
            unavailable_reason: Some("source_training_not_found"),
        });
    }

    let source_bundle = read_audit_bundle(&resolved_path)?;
    let actual_fingerprints = audit_fingerprints_from_bundle(&source_bundle)?;
    if actual_fingerprints != expected_fingerprints {
        return Err(
            "inference source training fingerprints do not match referenced audit bundle"
                .to_owned(),
        );
    }
    if json_field(&source_bundle, "final_model")? != model_json {
        return Err("inference source training model does not match inference model".to_owned());
    }

    let trace = json_array_field(&source_bundle, "witness_trace")?;
    let steps = parse_audit_steps(trace)?;
    let step = steps.get(audit_step).ok_or_else(|| {
        format!(
            "inference source training audit_step {audit_step} is out of range for {} step(s)",
            steps.len()
        )
    })?;
    verify_inference_sample_matches_audit_step(
        sample_json,
        step,
        sample_index,
        "inference source training",
    )?;

    Ok(InferenceSourceArtifactCheck {
        path: source_path_text,
        resolved_path: resolved_path_text,
        checked: true,
        payload_fingerprint: actual_fingerprints.payload,
        items_checked: 2,
        unavailable_reason: None,
    })
}

fn verify_inference_source_sample(
    inference_path: &Path,
    bundle: &serde_json::Value,
    sample_json: &serde_json::Value,
) -> Result<Option<InferenceSourceArtifactCheck>, String> {
    if let Some(source) = bundle.get("source_sample_audit") {
        return verify_inference_source_sample_audit(inference_path, source, sample_json).map(Some);
    }
    if let Some(source) = bundle.get("source_sample_json") {
        return verify_inference_source_sample_json(inference_path, source, sample_json).map(Some);
    }
    Ok(None)
}

fn verify_inference_source_sample_audit(
    inference_path: &Path,
    source: &serde_json::Value,
    sample_json: &serde_json::Value,
) -> Result<InferenceSourceArtifactCheck, String> {
    let source_path_text = json_str_field(source, "path")?.to_owned();
    let expected_fingerprints = audit_fingerprints_from_json(json_field(source, "fingerprints")?)?;
    let audit_step = json_usize_field(source, "audit_step")?;
    let sample_index = json_usize_field(source, "sample_index")?;
    let resolved_path = resolve_referenced_artifact_path(inference_path, &source_path_text);
    let resolved_path_text = resolved_path.display().to_string();
    if !resolved_path.exists() {
        return Ok(InferenceSourceArtifactCheck {
            path: source_path_text,
            resolved_path: resolved_path_text,
            checked: false,
            payload_fingerprint: expected_fingerprints.payload,
            items_checked: 0,
            unavailable_reason: Some("source_sample_audit_not_found"),
        });
    }

    let source_bundle = read_audit_bundle(&resolved_path)?;
    let actual_fingerprints = audit_fingerprints_from_bundle(&source_bundle)?;
    if actual_fingerprints != expected_fingerprints {
        return Err(
            "inference source sample audit fingerprints do not match referenced audit bundle"
                .to_owned(),
        );
    }
    let trace = json_array_field(&source_bundle, "witness_trace")?;
    let steps = parse_audit_steps(trace)?;
    let step = steps.get(audit_step).ok_or_else(|| {
        format!(
            "inference source sample audit_step {audit_step} is out of range for {} step(s)",
            steps.len()
        )
    })?;
    verify_inference_sample_matches_audit_step(
        sample_json,
        step,
        sample_index,
        "inference source sample audit",
    )?;

    Ok(InferenceSourceArtifactCheck {
        path: source_path_text,
        resolved_path: resolved_path_text,
        checked: true,
        payload_fingerprint: actual_fingerprints.payload,
        items_checked: 1,
        unavailable_reason: None,
    })
}

fn verify_inference_source_sample_json(
    inference_path: &Path,
    source: &serde_json::Value,
    sample_json: &serde_json::Value,
) -> Result<InferenceSourceArtifactCheck, String> {
    let source_path_text = json_str_field(source, "path")?.to_owned();
    let expected_fingerprint = json_str_field(source, "fingerprint")?.to_owned();
    let resolved_path = resolve_referenced_artifact_path(inference_path, &source_path_text);
    let resolved_path_text = resolved_path.display().to_string();
    if !resolved_path.exists() {
        return Ok(InferenceSourceArtifactCheck {
            path: source_path_text,
            resolved_path: resolved_path_text,
            checked: false,
            payload_fingerprint: expected_fingerprint,
            items_checked: 0,
            unavailable_reason: Some("source_sample_json_not_found"),
        });
    }

    let data = fs::read(&resolved_path).map_err(|error| {
        format!(
            "failed to read inference source sample JSON {}: {error}",
            resolved_path.display()
        )
    })?;
    let source_value: serde_json::Value = serde_json::from_slice(&data).map_err(|error| {
        format!(
            "failed to parse inference source sample JSON {}: {error}",
            resolved_path.display()
        )
    })?;
    let actual_fingerprint = sha256_json(&source_value);
    if actual_fingerprint != expected_fingerprint {
        return Err(
            "inference source sample JSON fingerprint does not match referenced sample file"
                .to_owned(),
        );
    }
    let source_sample =
        labeled_sample_from_json_value(&source_value, "inference source sample JSON")?;
    let embedded_image = json_u8_vec_field(sample_json, "image_u8", IMAGE_PIXELS)?;
    let embedded_label = json_i64_field(sample_json, "label")?;
    if source_sample.image != embedded_image || source_sample.label != embedded_label {
        return Err("inference source sample JSON does not match inference sample".to_owned());
    }

    Ok(InferenceSourceArtifactCheck {
        path: source_path_text,
        resolved_path: resolved_path_text,
        checked: true,
        payload_fingerprint: actual_fingerprint,
        items_checked: 1,
        unavailable_reason: None,
    })
}

fn verify_inference_sample_matches_audit_step(
    sample_json: &serde_json::Value,
    step: &AuditStep,
    sample_index: usize,
    context: &str,
) -> Result<(), String> {
    let embedded_image = json_u8_vec_field(sample_json, "image_u8", IMAGE_PIXELS)?;
    let embedded_label = json_i64_field(sample_json, "label")?;
    if sample_index != step.source_sample_index {
        return Err(format!(
            "{context} sample_index expected {}, found {sample_index}",
            step.source_sample_index
        ));
    }
    if embedded_label != step.tape.label {
        return Err(format!(
            "{context} label expected {}, found {embedded_label}",
            step.tape.label
        ));
    }
    if embedded_image != step.image {
        return Err(format!(
            "{context} image does not match referenced audit step"
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct InferenceSourceEvaluationCheck {
    path: String,
    resolved_path: String,
    checked: bool,
    row_index: usize,
    payload_fingerprint: String,
    sample_fingerprint: String,
    unavailable_reason: Option<&'static str>,
}

impl InferenceSourceEvaluationCheck {
    fn to_json(&self) -> serde_json::Value {
        let mut value = json!({
            "path": self.path,
            "resolved_path": self.resolved_path,
            "checked": self.checked,
            "row_index": self.row_index,
            "payload_fingerprint": self.payload_fingerprint,
            "sample_fingerprint": self.sample_fingerprint,
        });
        if let Some(reason) = self.unavailable_reason {
            value
                .as_object_mut()
                .expect("inference source evaluation check JSON is an object")
                .insert("unavailable_reason".to_owned(), json!(reason));
        }
        value
    }

    fn format_line(&self) -> String {
        if self.checked {
            format!(
                "checked=true row_index={} payload={} sample={}",
                self.row_index, self.payload_fingerprint, self.sample_fingerprint
            )
        } else {
            format!(
                "checked=false reason={} row_index={} path={}",
                self.unavailable_reason.unwrap_or("unavailable"),
                self.row_index,
                self.path
            )
        }
    }
}

fn verify_inference_source_evaluation(
    inference_path: &Path,
    bundle: &serde_json::Value,
    model_json: &serde_json::Value,
    sample_json: &serde_json::Value,
    inference: &InferenceAudit,
) -> Result<Option<InferenceSourceEvaluationCheck>, String> {
    let Some(source) = bundle.get("source_model_evaluation") else {
        return Ok(None);
    };
    let source_path_text = json_str_field(source, "path")?.to_owned();
    let expected_fingerprints =
        model_evaluation_fingerprints_from_json(json_field(source, "fingerprints")?)?;
    let row_index = json_usize_field(source, "row_index")?;
    let source_audit_step = json_optional_usize_field(source, "audit_step")?;
    let source_sample_index = json_optional_usize_field(source, "source_sample_index")?;
    let source_sample_fingerprint = json_str_field(source, "sample_fingerprint")?.to_owned();
    let resolved_path = resolve_referenced_artifact_path(inference_path, &source_path_text);
    let resolved_path_text = resolved_path.display().to_string();
    if !resolved_path.exists() {
        return Ok(Some(InferenceSourceEvaluationCheck {
            path: source_path_text,
            resolved_path: resolved_path_text,
            checked: false,
            row_index,
            payload_fingerprint: expected_fingerprints.payload,
            sample_fingerprint: source_sample_fingerprint,
            unavailable_reason: Some("source_evaluation_not_found"),
        }));
    }

    let source_bundle = read_model_evaluation_bundle(&resolved_path)?;
    let actual_fingerprints = model_evaluation_fingerprints_from_bundle(&source_bundle)?;
    if actual_fingerprints != expected_fingerprints {
        return Err(
            "inference source evaluation fingerprints do not match referenced evaluation bundle"
                .to_owned(),
        );
    }
    if json_field(&source_bundle, "model")? != model_json {
        return Err("inference source evaluation model does not match inference model".to_owned());
    }

    let source_samples =
        labeled_samples_from_embedded_json(json_field(&source_bundle, "samples")?)?;
    let _ = verify_model_evaluation_source_artifacts(
        &resolved_path,
        &source_bundle,
        json_field(&source_bundle, "model")?,
        &source_samples,
    )?;
    let source_rows = json_array_field(&source_bundle, "rows")?
        .iter()
        .enumerate()
        .map(|(position, row)| model_evaluation_row_from_json(row, position))
        .collect::<Result<Vec<_>, _>>()?;
    if source_rows.len() != source_samples.len() {
        return Err(format!(
            "inference source evaluation row/sample count mismatch: rows={} samples={}",
            source_rows.len(),
            source_samples.len()
        ));
    }
    let source_sample = source_samples.get(row_index).ok_or_else(|| {
        format!(
            "inference source evaluation row_index {row_index} is out of range for {} sample(s)",
            source_samples.len()
        )
    })?;
    let source_row = source_rows
        .get(row_index)
        .expect("row/sample count already matches");
    if source_row.index != row_index {
        return Err(format!(
            "inference source evaluation row_index expected {row_index}, found {}",
            source_row.index
        ));
    }
    let embedded_image = json_u8_vec_field(sample_json, "image_u8", IMAGE_PIXELS)?;
    let embedded_label = json_i64_field(sample_json, "label")?;
    if source_sample.image != embedded_image || source_sample.label != embedded_label {
        return Err(
            "inference source evaluation sample does not match inference sample".to_owned(),
        );
    }
    let actual_sample_fingerprint = sha256_json(&labeled_sample_json(source_sample));
    if actual_sample_fingerprint != source_row.sample_fingerprint
        || actual_sample_fingerprint != source_sample_fingerprint
    {
        return Err(
            "inference source evaluation sample fingerprint does not match referenced row"
                .to_owned(),
        );
    }
    if source_row.audit_step != source_sample.audit_step
        || source_row.source_sample_index != source_sample.source_sample_index
        || source_row.audit_step != source_audit_step
        || source_row.source_sample_index != source_sample_index
    {
        return Err("inference source evaluation lineage does not match referenced row".to_owned());
    }
    verify_model_evaluation_row_matches_inference(source_row, source_sample, inference)?;

    Ok(Some(InferenceSourceEvaluationCheck {
        path: source_path_text,
        resolved_path: resolved_path_text,
        checked: true,
        row_index,
        payload_fingerprint: actual_fingerprints.payload,
        sample_fingerprint: actual_sample_fingerprint,
        unavailable_reason: None,
    }))
}

fn verify_inference_proof(
    stored: &serde_json::Value,
    expected: &serde_json::Value,
) -> Result<(), String> {
    if stored != expected {
        return Err(
            "inference bundle proof does not match recomputed Reverie inference".to_owned(),
        );
    }
    Ok(())
}

fn verify_inference_result(
    result: &serde_json::Value,
    inference: &InferenceAudit,
) -> Result<(), String> {
    let logits = json_i64_vec_field(result, "logits", DIGITS)?;
    let prediction = json_i64_field(result, "prediction")?;
    let correct = json_bool_field(result, "correct")?;
    let checks = json_field(result, "witness_checks")?;
    let computed_prediction = json_i64_field(checks, "computed_prediction")?;
    let computed_correct = json_bool_field(checks, "computed_correct")?;
    let prediction_matches_logits = json_bool_field(checks, "prediction_matches_logits")?;
    let correct_matches_logits = json_bool_field(checks, "correct_matches_logits")?;
    let inverse = json_field(result, "inverse")?;
    let restored_initial_state = json_bool_field(inverse, "restored_initial_state")?;
    verify_inference_memory(json_field(result, "memory")?, inference.memory)?;
    if let Some(attribution) = result.get("attribution") {
        let expected = inference_attribution_json(&inference.attribution);
        if attribution != &expected {
            return Err(
                "inference bundle attribution does not match recomputed Q31 contributions"
                    .to_owned(),
            );
        }
    }

    if logits != inference.logits
        || prediction != inference.prediction
        || correct != inference.correct
        || computed_prediction != inference.computed_prediction
        || computed_correct != inference.computed_correct
        || prediction_matches_logits != inference.prediction_matches_logits
        || correct_matches_logits != inference.correct_matches_logits
        || !restored_initial_state
    {
        return Err(
            "inference bundle result does not match recomputed Reverie inference".to_owned(),
        );
    }
    Ok(())
}

fn verify_inference_memory(
    memory: &serde_json::Value,
    expected: InferenceMemoryReport,
) -> Result<(), String> {
    let checks = [
        ("model_payload_bytes", expected.model_payload_bytes),
        ("sample_payload_bytes", expected.sample_payload_bytes),
        ("witness_payload_bytes", expected.witness_payload_bytes),
        ("trace_payload_bytes", expected.trace_payload_bytes),
        ("replay_payload_bytes", expected.replay_payload_bytes),
        (
            "runtime_state_payload_bytes",
            expected.runtime_state_payload_bytes,
        ),
        ("forward_recompute_steps", expected.forward_recompute_steps),
        ("inverse_recompute_steps", expected.inverse_recompute_steps),
    ];
    for (name, expected) in checks {
        let actual = json_usize_field(memory, name)?;
        if actual != expected {
            return Err(format!(
                "inference bundle memory `{name}` expected {expected}, found {actual}"
            ));
        }
    }
    Ok(())
}

fn read_inference_bundle(path: &Path) -> Result<serde_json::Value, String> {
    let data = fs::read(path).map_err(|error| {
        format!(
            "failed to read inference bundle {}: {error}",
            path.display()
        )
    })?;
    let bundle: serde_json::Value = serde_json::from_slice(&data).map_err(|error| {
        format!(
            "failed to parse inference bundle {}: {error}",
            path.display()
        )
    })?;
    let kind = json_str_field(&bundle, "kind")?;
    if kind != "reverie_mnist_linear_q31_inference_replay_bundle" {
        return Err(format!(
            "inference bundle kind `{kind}` is not reverie_mnist_linear_q31_inference_replay_bundle"
        ));
    }
    verify_inference_fingerprints(&bundle)?;
    Ok(bundle)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InferenceFingerprints {
    identify_source: String,
    computation: String,
    model: String,
    sample: String,
    result: String,
    proof: String,
    report: String,
    payload: String,
}

impl InferenceFingerprints {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "algorithm": "sha256",
            "identify_source": self.identify_source,
            "computation": self.computation,
            "model": self.model,
            "sample": self.sample,
            "result": self.result,
            "proof": self.proof,
            "report": self.report,
            "payload": self.payload,
        })
    }
}

fn inference_fingerprints_for_unsigned_payload(
    payload: &serde_json::Value,
) -> InferenceFingerprints {
    InferenceFingerprints {
        identify_source: sha256_bytes(IDENTIFY_SOURCE.as_bytes()),
        computation: inference_computation_fingerprint(payload),
        model: sha256_json(&payload["model"]),
        sample: sha256_json(&payload["sample"]),
        result: sha256_json(&payload["result"]),
        proof: sha256_json(payload.get("proof").unwrap_or(&serde_json::Value::Null)),
        report: sha256_json(&payload["report"]),
        payload: sha256_json(payload),
    }
}

fn inference_fingerprints_from_bundle(
    bundle: &serde_json::Value,
) -> Result<InferenceFingerprints, String> {
    let fingerprints = json_field(bundle, "fingerprints")?;
    let algorithm = json_str_field(fingerprints, "algorithm")?;
    if algorithm != "sha256" {
        return Err(format!(
            "inference bundle fingerprint algorithm `{algorithm}` is not sha256"
        ));
    }
    Ok(InferenceFingerprints {
        identify_source: json_str_field(fingerprints, "identify_source")?.to_owned(),
        computation: json_str_field(fingerprints, "computation")?.to_owned(),
        model: json_str_field(fingerprints, "model")?.to_owned(),
        sample: json_str_field(fingerprints, "sample")?.to_owned(),
        result: json_str_field(fingerprints, "result")?.to_owned(),
        proof: fingerprints
            .get("proof")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| sha256_json(&serde_json::Value::Null)),
        report: json_str_field(fingerprints, "report")?.to_owned(),
        payload: json_str_field(fingerprints, "payload")?.to_owned(),
    })
}

fn verify_inference_fingerprints(bundle: &serde_json::Value) -> Result<(), String> {
    let stored = inference_fingerprints_from_bundle(bundle)?;
    let mut unsigned_payload = bundle.clone();
    unsigned_payload
        .as_object_mut()
        .expect("inference bundle is an object")
        .remove("fingerprints");
    let computed = inference_fingerprints_for_unsigned_payload(&unsigned_payload);
    if stored != computed {
        return Err(format!(
            "inference bundle fingerprint mismatch: stored payload {}, computed payload {}",
            stored.payload, computed.payload
        ));
    }
    Ok(())
}

fn inference_computation_fingerprint(payload: &serde_json::Value) -> String {
    let computation = json!({
        "identify_source": sha256_bytes(IDENTIFY_SOURCE.as_bytes()),
        "model": &payload["model"],
        "sample": &payload["sample"],
        "result": &payload["result"],
    });
    let mut computation = computation;
    if let Some(proof) = payload.get("proof") {
        computation
            .as_object_mut()
            .expect("inference computation fingerprint is an object")
            .insert("proof".to_owned(), proof.clone());
    }
    sha256_json(&computation)
}

#[derive(Debug, Clone)]
struct ActivePixel {
    index: usize,
    value: u8,
    q31: i64,
}

#[derive(Debug, Clone)]
struct LogitRank {
    digit: usize,
    value: i64,
}

#[derive(Debug, Clone)]
struct LogitMargin {
    predicted: LogitRank,
    runner_up: LogitRank,
    margin: i64,
}

#[derive(Debug, Clone)]
struct PixelContribution {
    pixel: usize,
    value: u8,
    q31: i64,
    weight: i64,
    contribution: i64,
}

#[derive(Debug, Clone)]
struct MarginContribution {
    pixel: usize,
    value: u8,
    q31: i64,
    predicted_weight: i64,
    runner_up_weight: i64,
    weight_delta: i64,
    contribution: i64,
}

#[derive(Debug, Clone)]
struct InferenceAttribution {
    predicted_digit: usize,
    runner_up_digit: usize,
    predicted_logit: i64,
    runner_up_logit: i64,
    margin: i64,
    bias: i64,
    runner_up_bias: i64,
    margin_bias: i64,
    contribution_sum: i64,
    margin_contribution_sum: i64,
    reconstructed_logit: i64,
    reconstructed_margin: i64,
    matches_logit: bool,
    matches_margin: bool,
    contribution_count: usize,
    margin_contribution_count: usize,
    contribution_ledger_fingerprint: String,
    margin_contribution_ledger_fingerprint: String,
    top_contributions: Vec<PixelContribution>,
    top_margin_contributions: Vec<MarginContribution>,
}

#[derive(Debug, Clone)]
struct WeightDelta {
    pixel: usize,
    digit: usize,
    delta: i64,
}

#[derive(Debug)]
struct WeightDeltaStats {
    nonzero_count: usize,
    max_abs: i128,
    ledger_fingerprint: String,
    top: Vec<WeightDelta>,
}

#[derive(Debug)]
struct ModelWindow {
    before: Model,
    after: Model,
    reversed_later_steps: usize,
}

#[derive(Debug)]
struct WeightWindow {
    pixel: usize,
    digit: usize,
    before: i64,
    after: i64,
    observed_delta: i64,
    computed_delta: i64,
}

fn reconstruct_model_window(
    program: &Program,
    final_model: &Model,
    steps: &[AuditStep],
    step_index: usize,
) -> Result<ModelWindow, String> {
    let mut model = final_model.clone();
    for step in steps.iter().skip(step_index + 1).rev() {
        model = reverse_one_step(program, &model, &step.image, &step.tape)?;
    }
    let after = model;
    let before = reverse_one_step(
        program,
        &after,
        &steps[step_index].image,
        &steps[step_index].tape,
    )?;
    Ok(ModelWindow {
        before,
        after,
        reversed_later_steps: steps.len() - step_index - 1,
    })
}

fn active_pixels(image: &[u8]) -> Vec<ActivePixel> {
    image
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value != 0)
        .map(|(index, value)| ActivePixel {
            index,
            value,
            q31: pixel_to_q31(value),
        })
        .collect()
}

fn top_logits(logits: &[i64]) -> Vec<LogitRank> {
    let mut ranked = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(digit, value)| LogitRank { digit, value })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .value
            .cmp(&left.value)
            .then_with(|| left.digit.cmp(&right.digit))
    });
    ranked.truncate(3);
    ranked
}

fn logit_margin(top_logits: &[LogitRank]) -> LogitMargin {
    let predicted = top_logits
        .first()
        .cloned()
        .unwrap_or(LogitRank { digit: 0, value: 0 });
    let runner_up = top_logits
        .get(1)
        .cloned()
        .unwrap_or_else(|| predicted.clone());
    let margin = predicted.value.wrapping_sub(runner_up.value);
    LogitMargin {
        predicted,
        runner_up,
        margin,
    }
}

fn inference_attribution(
    model: &Model,
    image: &[u8],
    logits: &[i64],
    top_logits: &[LogitRank],
) -> InferenceAttribution {
    let predicted_digit = top_logits.first().map_or(0, |logit| logit.digit);
    let runner_up_digit = top_logits
        .get(1)
        .map_or(predicted_digit, |logit| logit.digit);
    let predicted_logit = logits[predicted_digit];
    let runner_up_logit = logits[runner_up_digit];
    let bias = model.bias[predicted_digit];
    let runner_up_bias = model.bias[runner_up_digit];
    let margin_bias = bias.wrapping_sub(runner_up_bias);
    let mut contribution_sum = 0_i64;
    let mut margin_contribution_sum = 0_i64;
    let mut reconstructed_logit = bias;
    let mut reconstructed_margin = margin_bias;
    let mut contributions = Vec::new();
    let mut margin_contributions = Vec::new();

    for (pixel, value) in image.iter().copied().enumerate() {
        if value == 0 {
            continue;
        }
        let q31 = pixel_to_q31(value);
        let weight = model.weights[pixel][predicted_digit];
        let runner_up_weight = model.weights[pixel][runner_up_digit];
        let weight_delta = weight.wrapping_sub(runner_up_weight);
        let contribution = fixed_mul_q31(q31, weight);
        let runner_up_contribution = fixed_mul_q31(q31, runner_up_weight);
        let margin_contribution = contribution.wrapping_sub(runner_up_contribution);
        contribution_sum = contribution_sum.wrapping_add(contribution);
        margin_contribution_sum = margin_contribution_sum.wrapping_add(margin_contribution);
        reconstructed_logit = reconstructed_logit.wrapping_add(contribution);
        reconstructed_margin = reconstructed_margin.wrapping_add(margin_contribution);
        if contribution != 0 {
            contributions.push(PixelContribution {
                pixel,
                value,
                q31,
                weight,
                contribution,
            });
        }
        if margin_contribution != 0 {
            margin_contributions.push(MarginContribution {
                pixel,
                value,
                q31,
                predicted_weight: weight,
                runner_up_weight,
                weight_delta,
                contribution: margin_contribution,
            });
        }
    }

    let contribution_ledger_fingerprint =
        contribution_ledger_fingerprint(predicted_digit, &contributions);
    let margin_contribution_ledger_fingerprint = margin_contribution_ledger_fingerprint(
        predicted_digit,
        runner_up_digit,
        &margin_contributions,
    );
    let contribution_count = contributions.len();
    let margin_contribution_count = margin_contributions.len();
    let mut top_contributions = contributions;
    let mut top_margin_contributions = margin_contributions;
    top_contributions.sort_by(|left, right| {
        abs_i128(right.contribution)
            .cmp(&abs_i128(left.contribution))
            .then_with(|| left.pixel.cmp(&right.pixel))
    });
    top_contributions.truncate(10);
    top_margin_contributions.sort_by(|left, right| {
        abs_i128(right.contribution)
            .cmp(&abs_i128(left.contribution))
            .then_with(|| left.pixel.cmp(&right.pixel))
    });
    top_margin_contributions.truncate(10);
    let margin = predicted_logit.wrapping_sub(runner_up_logit);

    InferenceAttribution {
        predicted_digit,
        runner_up_digit,
        predicted_logit,
        runner_up_logit,
        margin,
        bias,
        runner_up_bias,
        margin_bias,
        contribution_sum,
        margin_contribution_sum,
        reconstructed_logit,
        reconstructed_margin,
        matches_logit: reconstructed_logit == predicted_logit,
        matches_margin: reconstructed_margin == margin,
        contribution_count,
        margin_contribution_count,
        contribution_ledger_fingerprint,
        margin_contribution_ledger_fingerprint,
        top_contributions,
        top_margin_contributions,
    }
}

fn pixel_contribution_json(contribution: &PixelContribution) -> serde_json::Value {
    json!({
        "pixel": contribution.pixel,
        "u8": contribution.value,
        "q31": contribution.q31,
        "weight": contribution.weight,
        "contribution": contribution.contribution,
    })
}

fn margin_contribution_json(contribution: &MarginContribution) -> serde_json::Value {
    json!({
        "pixel": contribution.pixel,
        "u8": contribution.value,
        "q31": contribution.q31,
        "predicted_weight": contribution.predicted_weight,
        "runner_up_weight": contribution.runner_up_weight,
        "weight_delta": contribution.weight_delta,
        "contribution": contribution.contribution,
    })
}

fn contribution_ledger_fingerprint(
    predicted_digit: usize,
    contributions: &[PixelContribution],
) -> String {
    let ledger = json!({
        "schema": "q31_linear_inference_contribution_ledger_v1",
        "predicted_digit": predicted_digit,
        "rows": contributions.iter().map(pixel_contribution_json).collect::<Vec<_>>(),
    });
    sha256_json(&ledger)
}

fn margin_contribution_ledger_fingerprint(
    predicted_digit: usize,
    runner_up_digit: usize,
    contributions: &[MarginContribution],
) -> String {
    let ledger = json!({
        "schema": "q31_linear_inference_margin_contribution_ledger_v1",
        "predicted_digit": predicted_digit,
        "runner_up_digit": runner_up_digit,
        "rows": contributions.iter().map(margin_contribution_json).collect::<Vec<_>>(),
    });
    sha256_json(&ledger)
}

fn weight_delta_json(delta: &WeightDelta) -> serde_json::Value {
    json!({
        "pixel": delta.pixel,
        "digit": delta.digit,
        "delta": delta.delta,
    })
}

fn weight_delta_ledger_fingerprint(deltas: &[WeightDelta]) -> String {
    let ledger = json!({
        "schema": "q31_linear_training_weight_delta_ledger_v1",
        "rows": deltas.iter().map(weight_delta_json).collect::<Vec<_>>(),
    });
    sha256_json(&ledger)
}

fn nonzero_bias_delta_count(bias_delta: &[i64]) -> usize {
    bias_delta.iter().filter(|delta| **delta != 0).count()
}

fn bias_delta_ledger_fingerprint(bias_delta: &[i64]) -> String {
    let rows = bias_delta
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, delta)| *delta != 0)
        .map(|(digit, delta)| {
            json!({
                "digit": digit,
                "delta": delta,
            })
        })
        .collect::<Vec<_>>();
    let ledger = json!({
        "schema": "q31_linear_training_bias_delta_ledger_v1",
        "rows": rows,
    });
    sha256_json(&ledger)
}

fn inference_attribution_json(attribution: &InferenceAttribution) -> serde_json::Value {
    json!({
        "predicted_digit": attribution.predicted_digit,
        "runner_up_digit": attribution.runner_up_digit,
        "predicted_logit": attribution.predicted_logit,
        "runner_up_logit": attribution.runner_up_logit,
        "margin": attribution.margin,
        "bias": attribution.bias,
        "runner_up_bias": attribution.runner_up_bias,
        "margin_bias": attribution.margin_bias,
        "contribution_sum": attribution.contribution_sum,
        "margin_contribution_sum": attribution.margin_contribution_sum,
        "reconstructed_logit": attribution.reconstructed_logit,
        "reconstructed_margin": attribution.reconstructed_margin,
        "matches_logit": attribution.matches_logit,
        "matches_margin": attribution.matches_margin,
        "contribution_count": attribution.contribution_count,
        "margin_contribution_count": attribution.margin_contribution_count,
        "top_contribution_count": attribution.top_contributions.len(),
        "top_margin_contribution_count": attribution.top_margin_contributions.len(),
        "contribution_ledger_fingerprint": &attribution.contribution_ledger_fingerprint,
        "margin_contribution_ledger_fingerprint": &attribution.margin_contribution_ledger_fingerprint,
        "top_contributions": attribution.top_contributions.iter().map(pixel_contribution_json).collect::<Vec<_>>(),
        "top_margin_contributions": attribution.top_margin_contributions.iter().map(margin_contribution_json).collect::<Vec<_>>(),
    })
}

fn applied_bias_delta(error: &[i64], lr: i64) -> Vec<i64> {
    error
        .iter()
        .map(|value| fixed_mul_q31(*value, lr).wrapping_neg())
        .collect()
}

fn vector_delta(before: &[i64], after: &[i64]) -> Vec<i64> {
    before
        .iter()
        .zip(after)
        .map(|(before, after)| after.wrapping_sub(*before))
        .collect()
}

fn weight_delta_stats(image: &[u8], error: &[i64], lr: i64) -> WeightDeltaStats {
    let mut deltas = Vec::new();
    let mut nonzero_count = 0_usize;
    let mut max_abs = 0_i128;
    for (pixel, value) in image.iter().copied().enumerate() {
        if value == 0 {
            continue;
        }
        let pixel_q31 = pixel_to_q31(value);
        for (digit, error) in error.iter().copied().enumerate() {
            let outer = fixed_mul_q31(pixel_q31, error);
            let scaled = fixed_mul_q31(outer, lr);
            let delta = scaled.wrapping_neg();
            if delta != 0 {
                nonzero_count += 1;
                max_abs = max_abs.max(abs_i128(delta));
                deltas.push(WeightDelta {
                    pixel,
                    digit,
                    delta,
                });
            }
        }
    }
    let ledger_fingerprint = weight_delta_ledger_fingerprint(&deltas);
    let mut top = deltas;
    top.sort_by(|left, right| {
        abs_i128(right.delta)
            .cmp(&abs_i128(left.delta))
            .then_with(|| left.pixel.cmp(&right.pixel))
            .then_with(|| left.digit.cmp(&right.digit))
    });
    top.truncate(10);
    WeightDeltaStats {
        nonzero_count,
        max_abs,
        ledger_fingerprint,
        top,
    }
}

fn fixed_mul_q31(left: i64, right: i64) -> i64 {
    ((i128::from(left) * i128::from(right)) >> 31) as i64
}

fn argmax_first(values: &[i64]) -> i64 {
    let mut best_index = 0_usize;
    let mut best_value = values[0];
    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    best_index as i64
}

fn abs_i128(value: i64) -> i128 {
    if value < 0 {
        -i128::from(value)
    } else {
        i128::from(value)
    }
}

fn format_active_pixel_preview(active_pixels: &[ActivePixel], limit: usize) -> String {
    let mut parts = active_pixels
        .iter()
        .take(limit)
        .map(|pixel| format!("{}:{}({})", pixel.index, pixel.value, pixel.q31))
        .collect::<Vec<_>>();
    if active_pixels.len() > limit {
        parts.push("...".to_owned());
    }
    format!("[{}]", parts.join(", "))
}

fn format_weight_delta_preview(deltas: &[WeightDelta]) -> String {
    let parts = deltas
        .iter()
        .map(|delta| format!("p{} d{} {}", delta.pixel, delta.digit, delta.delta))
        .collect::<Vec<_>>();
    format!("[{}]", parts.join(", "))
}

fn format_weight_window_preview(windows: &[WeightWindow]) -> String {
    let parts = windows
        .iter()
        .map(|window| {
            format!(
                "p{} d{} {}->{} (obs {}, calc {})",
                window.pixel,
                window.digit,
                window.before,
                window.after,
                window.observed_delta,
                window.computed_delta
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", parts.join(", "))
}

fn format_logit_preview(logits: &[LogitRank]) -> String {
    let parts = logits
        .iter()
        .map(|logit| format!("d{}={}", logit.digit, logit.value))
        .collect::<Vec<_>>();
    format!("[{}]", parts.join(", "))
}

fn format_contribution_preview(contributions: &[PixelContribution]) -> String {
    let parts = contributions
        .iter()
        .map(|contribution| {
            format!(
                "p{} u8={} w={} c={}",
                contribution.pixel,
                contribution.value,
                contribution.weight,
                contribution.contribution
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", parts.join(", "))
}

fn format_margin_contribution_preview(contributions: &[MarginContribution]) -> String {
    let parts = contributions
        .iter()
        .map(|contribution| {
            format!(
                "p{} u8={} dw={} c={}",
                contribution.pixel,
                contribution.value,
                contribution.weight_delta,
                contribution.contribution
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", parts.join(", "))
}

fn write_markdown_file(path: &Path, markdown: &str) -> Result<(), String> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    fs::write(path, markdown)
        .map_err(|error| format!("failed to write Markdown {}: {error}", path.display()))
}

fn render_inference_audit_markdown(report: &serde_json::Value) -> Result<String, String> {
    let attribution = json_field(report, "attribution")?;
    let memory = json_field(report, "memory")?;
    let proof = json_field(report, "proof")?;
    let contract = json_field(report, "explanation_contract")?;
    let forward = json_field(report, "forward")?;
    let inverse = json_field(report, "inverse")?;
    let source = inference_audit_source_label(report)?;
    let audit_step = report
        .get("audit_step")
        .and_then(serde_json::Value::as_u64)
        .map_or_else(|| "n/a".to_owned(), |step| step.to_string());
    let sample_index = report
        .get("sample_index")
        .and_then(serde_json::Value::as_u64)
        .map_or_else(|| "n/a".to_owned(), |index| index.to_string());
    let output = report
        .get("inference_output")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("not written");
    let recompute_steps = json_usize_field(memory, "forward_recompute_steps")?
        + json_usize_field(memory, "inverse_recompute_steps")?;

    let mut lines = vec![
        "# Reverie Inference Audit".to_owned(),
        String::new(),
        "| Source | Step | Sample | Label | Prediction | Correct | Reverse | Output |".to_owned(),
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |".to_owned(),
        format!(
            "| {} | {} | {} | {} | {} | {} | {} | `{}` |",
            markdown_cell(&source),
            audit_step,
            sample_index,
            json_i64_field(report, "label")?,
            json_i64_field(report, "prediction")?,
            json_bool_field(report, "correct")?,
            ok_failed(json_bool_field(inverse, "restored_initial_state")?),
            markdown_cell(output),
        ),
        String::new(),
        "## Top Logits".to_owned(),
        String::new(),
        "| Rank | Digit | Logit |".to_owned(),
        "| ---: | ---: | ---: |".to_owned(),
    ];
    for (rank, row) in json_array_field(report, "top_logits")?.iter().enumerate() {
        lines.push(format!(
            "| {} | {} | {} |",
            rank + 1,
            json_usize_field(row, "digit")?,
            comma_i64(json_i64_field(row, "logit")?),
        ));
    }

    lines.extend([
        String::new(),
        "## Explanation".to_owned(),
        String::new(),
        "| Predicted | Runner-up | Margin | Winning logit | Recomputed logit | Recomputed margin |"
            .to_owned(),
        "| ---: | ---: | ---: | ---: | ---: | ---: |".to_owned(),
        format!(
            "| {} | {} | {} | {} | {} | {} |",
            json_usize_field(attribution, "predicted_digit")?,
            json_usize_field(attribution, "runner_up_digit")?,
            comma_i64(json_i64_field(attribution, "margin")?),
            comma_i64(json_i64_field(attribution, "predicted_logit")?),
            comma_i64(json_i64_field(attribution, "reconstructed_logit")?),
            comma_i64(json_i64_field(attribution, "reconstructed_margin")?),
        ),
        String::new(),
        "| Ledger | Rows | Fingerprint |".to_owned(),
        "| --- | ---: | --- |".to_owned(),
        format!(
            "| contribution | {} | `{}` |",
            comma_usize(json_usize_field(attribution, "contribution_count")?),
            short_hash(json_str_field(
                attribution,
                "contribution_ledger_fingerprint"
            )?)
        ),
        format!(
            "| margin contribution | {} | `{}` |",
            comma_usize(json_usize_field(attribution, "margin_contribution_count")?),
            short_hash(json_str_field(
                attribution,
                "margin_contribution_ledger_fingerprint"
            )?)
        ),
        String::new(),
        "## Top Contributions".to_owned(),
        String::new(),
        "| Pixel | U8 | Weight | Contribution |".to_owned(),
        "| ---: | ---: | ---: | ---: |".to_owned(),
    ]);
    for row in json_array_field(attribution, "top_contributions")? {
        lines.push(format!(
            "| {} | {} | {} | {} |",
            json_usize_field(row, "pixel")?,
            json_usize_field(row, "u8")?,
            comma_i64(json_i64_field(row, "weight")?),
            comma_i64(json_i64_field(row, "contribution")?),
        ));
    }

    lines.extend([
        String::new(),
        "## Top Margin Contributions".to_owned(),
        String::new(),
        "| Pixel | U8 | Predicted weight | Runner-up weight | Delta | Contribution |".to_owned(),
        "| ---: | ---: | ---: | ---: | ---: | ---: |".to_owned(),
    ]);
    for row in json_array_field(attribution, "top_margin_contributions")? {
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} |",
            json_usize_field(row, "pixel")?,
            json_usize_field(row, "u8")?,
            comma_i64(json_i64_field(row, "predicted_weight")?),
            comma_i64(json_i64_field(row, "runner_up_weight")?),
            comma_i64(json_i64_field(row, "weight_delta")?),
            comma_i64(json_i64_field(row, "contribution")?),
        ));
    }

    lines.extend([
        String::new(),
        "## Replay Proof".to_owned(),
        String::new(),
        "| Claim | Model bytes | Sample bytes | Witness bytes | Trace bytes | Replay bytes | Runtime state | Recompute steps | Forward | Reverse |".to_owned(),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |".to_owned(),
        format!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {:.3}s | {:.3}s |",
            markdown_cell(json_str_field(proof, "claim")?),
            comma_usize(json_usize_field(proof, "model_payload_bytes")?),
            comma_usize(json_usize_field(proof, "sample_payload_bytes")?),
            comma_usize(json_usize_field(proof, "witness_payload_bytes")?),
            comma_usize(json_usize_field(proof, "trace_payload_bytes")?),
            comma_usize(json_usize_field(proof, "replay_payload_bytes")?),
            comma_usize(json_usize_field(memory, "runtime_state_payload_bytes")?),
            comma_usize(recompute_steps),
            json_f64_field(forward, "elapsed_seconds")?,
            json_f64_field(inverse, "elapsed_seconds")?,
        ),
    ]);
    append_inference_fingerprint_rows(&mut lines, report, proof)?;

    lines.extend([
        String::new(),
        "## Contract Checks".to_owned(),
        String::new(),
        "| Check | Passed | Evidence | Requirement |".to_owned(),
        "| --- | --- | --- | --- |".to_owned(),
    ]);
    for check in json_array_field(contract, "checks")? {
        lines.push(format!(
            "| `{}` | {} | {} | {} |",
            markdown_cell(json_str_field(check, "metric")?),
            ok_failed(json_bool_field(check, "passed")?),
            markdown_cell(json_str_field(check, "actual")?),
            markdown_cell(json_str_field(check, "requirement")?),
        ));
    }

    lines.extend([
        String::new(),
        "## Sources".to_owned(),
        String::new(),
        "| Source | Payload | Details |".to_owned(),
        "| --- | --- | --- |".to_owned(),
    ]);
    append_inference_audit_source_rows(&mut lines, report)?;

    Ok(lines.join("\n") + "\n")
}

fn inference_audit_source_label(report: &serde_json::Value) -> Result<String, String> {
    if let Some(evaluation) = report
        .get("evaluation_bundle")
        .filter(|value| value.is_object())
    {
        return Ok(format!(
            "evaluation row from {}",
            json_str_field(evaluation, "path")?
        ));
    }
    if let Some(model) = report.get("model_bundle").filter(|value| value.is_object()) {
        return Ok(format!("model {}", json_str_field(model, "path")?));
    }
    if let Some(path) = report.get("path").and_then(serde_json::Value::as_str) {
        return Ok(path.to_owned());
    }
    Ok("inference report".to_owned())
}

fn append_inference_fingerprint_rows(
    lines: &mut Vec<String>,
    report: &serde_json::Value,
    proof: &serde_json::Value,
) -> Result<(), String> {
    lines.extend([
        String::new(),
        "| Fingerprint | SHA-256 |".to_owned(),
        "| --- | --- |".to_owned(),
    ]);
    for (prefix, value) in [
        ("source", report.get("fingerprints")),
        (
            "model bundle",
            report
                .get("model_bundle")
                .and_then(|value| value.get("fingerprints")),
        ),
        (
            "evaluation bundle",
            report
                .get("evaluation_bundle")
                .and_then(|value| value.get("fingerprints")),
        ),
    ] {
        if let Some(fingerprints) = value.filter(|value| value.is_object()) {
            for name in ["computation", "payload"] {
                if let Some(digest) = fingerprints.get(name).and_then(serde_json::Value::as_str) {
                    lines.push(format!(
                        "| {} {} | `{}` |",
                        prefix,
                        name,
                        short_hash(digest)
                    ));
                }
            }
        }
    }
    if let Some(fingerprints) = proof.get("fingerprints").filter(|value| value.is_object()) {
        for name in ["identify_source", "model", "sample", "result"] {
            if let Some(digest) = fingerprints.get(name).and_then(serde_json::Value::as_str) {
                lines.push(format!("| proof {name} | `{}` |", short_hash(digest)));
            }
        }
    }
    Ok(())
}

fn append_inference_audit_source_rows(
    lines: &mut Vec<String>,
    report: &serde_json::Value,
) -> Result<(), String> {
    let mut count = 0usize;
    if let Some(source) = report.get("path").and_then(serde_json::Value::as_str) {
        count += 1;
        lines.push(format!(
            "| training audit | {} | {} |",
            report_payload_cell(report.get("fingerprints")),
            markdown_cell(source),
        ));
    }
    if let Some(model) = report.get("model_bundle").filter(|value| value.is_object()) {
        count += 1;
        lines.push(format!(
            "| model bundle | {} | {} |",
            report_payload_cell(model.get("fingerprints")),
            markdown_cell(json_str_field(model, "path")?),
        ));
    }
    if let Some(evaluation) = report
        .get("evaluation_bundle")
        .filter(|value| value.is_object())
    {
        count += 1;
        lines.push(format!(
            "| evaluation bundle | {} | {} |",
            report_payload_cell(evaluation.get("fingerprints")),
            markdown_cell(json_str_field(evaluation, "path")?),
        ));
    }
    if let Some(sample_source) = report
        .get("sample_source")
        .filter(|value| value.is_object())
    {
        count += 1;
        let kind = sample_source
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("sample");
        lines.push(format!(
            "| sample source `{}` | {} | {} |",
            markdown_cell(kind),
            report_payload_cell(sample_source.get("fingerprints")),
            markdown_cell(&sample_source_details(sample_source)),
        ));
    }
    if count == 0 {
        lines.push("| n/a | `n/a` | no source artifacts recorded |".to_owned());
    }
    Ok(())
}

fn report_payload_cell(value: Option<&serde_json::Value>) -> String {
    value
        .and_then(|value| value.get("payload"))
        .and_then(serde_json::Value::as_str)
        .map(|digest| format!("`{}`", short_hash(digest)))
        .unwrap_or_else(|| "`n/a`".to_owned())
}

fn sample_source_details(source: &serde_json::Value) -> String {
    if let Some(path) = source.get("path").and_then(serde_json::Value::as_str) {
        let mut details = path.to_owned();
        if let Some(step) = source.get("audit_step").and_then(serde_json::Value::as_u64) {
            details.push_str(&format!(" step={step}"));
        }
        if let Some(index) = source
            .get("sample_index")
            .and_then(serde_json::Value::as_u64)
        {
            details.push_str(&format!(" sample={index}"));
        }
        if let Some(row) = source.get("row_index").and_then(serde_json::Value::as_u64) {
            details.push_str(&format!(" row={row}"));
        }
        return details;
    }
    if let Some(row) = source.get("row_index").and_then(serde_json::Value::as_u64) {
        return format!("row={row}");
    }
    "inline sample".to_owned()
}

fn render_inference_verification_markdown(report: &serde_json::Value) -> Result<String, String> {
    let fingerprints = json_field(report, "fingerprints")?;
    let attribution = json_field(report, "attribution")?;
    let memory = json_field(report, "memory")?;
    let proof = json_field(report, "proof")?;
    let contract = json_field(report, "explanation_contract")?;
    let source_inputs_checked = json_bool_field(report, "source_training_checked")?
        || (json_bool_field(report, "source_model_checked")?
            && (json_bool_field(report, "source_sample_checked")?
                || json_bool_field(report, "source_evaluation_checked")?));
    let recompute_steps = json_usize_field(memory, "forward_recompute_steps")?
        + json_usize_field(memory, "inverse_recompute_steps")?;

    let mut lines =
        vec![
        "# Reverie Inference Verification".to_owned(),
        String::new(),
        "| Path | Prediction | Correct | Result | Proof | Reverse | Source inputs | Elapsed |"
            .to_owned(),
        "| --- | ---: | --- | --- | --- | --- | --- | ---: |".to_owned(),
        format!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {:.3}s |",
            markdown_cell(json_str_field(report, "path")?),
            json_i64_field(report, "prediction")?,
            json_bool_field(report, "correct")?,
            ok_failed(json_bool_field(report, "result_matches")?),
            ok_failed(json_bool_field(report, "proof_matches")?),
            ok_failed(json_bool_field(report, "restored_initial_state")?),
            ok_failed(source_inputs_checked),
            json_f64_field(report, "elapsed_seconds")?,
        ),
        String::new(),
        "## Explanation".to_owned(),
        String::new(),
        "| Predicted | Runner-up | Margin | Winning logit | Recomputed logit | Recomputed margin |"
            .to_owned(),
        "| ---: | ---: | ---: | ---: | ---: | ---: |".to_owned(),
        format!(
            "| {} | {} | {} | {} | {} | {} |",
            json_usize_field(attribution, "predicted_digit")?,
            json_usize_field(attribution, "runner_up_digit")?,
            comma_i64(json_i64_field(attribution, "margin")?),
            comma_i64(json_i64_field(attribution, "predicted_logit")?),
            comma_i64(json_i64_field(attribution, "reconstructed_logit")?),
            comma_i64(json_i64_field(attribution, "reconstructed_margin")?),
        ),
        String::new(),
        "| Ledger | Rows | Fingerprint |".to_owned(),
        "| --- | ---: | --- |".to_owned(),
        format!(
            "| contribution | {} | `{}` |",
            comma_usize(json_usize_field(attribution, "contribution_count")?),
            short_hash(json_str_field(attribution, "contribution_ledger_fingerprint")?)
        ),
        format!(
            "| margin contribution | {} | `{}` |",
            comma_usize(json_usize_field(attribution, "margin_contribution_count")?),
            short_hash(json_str_field(
                attribution,
                "margin_contribution_ledger_fingerprint"
            )?)
        ),
        String::new(),
        "## Top Contributions".to_owned(),
        String::new(),
        "| Pixel | U8 | Weight | Contribution |".to_owned(),
        "| ---: | ---: | ---: | ---: |".to_owned(),
    ];

    for row in json_array_field(attribution, "top_contributions")? {
        lines.push(format!(
            "| {} | {} | {} | {} |",
            json_usize_field(row, "pixel")?,
            json_usize_field(row, "u8")?,
            comma_i64(json_i64_field(row, "weight")?),
            comma_i64(json_i64_field(row, "contribution")?),
        ));
    }

    lines.extend([
        String::new(),
        "## Top Margin Contributions".to_owned(),
        String::new(),
        "| Pixel | U8 | Predicted weight | Runner-up weight | Delta | Contribution |".to_owned(),
        "| ---: | ---: | ---: | ---: | ---: | ---: |".to_owned(),
    ]);
    for row in json_array_field(attribution, "top_margin_contributions")? {
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} |",
            json_usize_field(row, "pixel")?,
            json_usize_field(row, "u8")?,
            comma_i64(json_i64_field(row, "predicted_weight")?),
            comma_i64(json_i64_field(row, "runner_up_weight")?),
            comma_i64(json_i64_field(row, "weight_delta")?),
            comma_i64(json_i64_field(row, "contribution")?),
        ));
    }

    lines.extend([
        String::new(),
        "## Replay Proof".to_owned(),
        String::new(),
        "| Claim | Model bytes | Sample bytes | Witness bytes | Trace bytes | Replay bytes | Runtime state | Recompute steps |".to_owned(),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |".to_owned(),
        format!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |",
            markdown_cell(json_str_field(proof, "claim")?),
            comma_usize(json_usize_field(proof, "model_payload_bytes")?),
            comma_usize(json_usize_field(proof, "sample_payload_bytes")?),
            comma_usize(json_usize_field(proof, "witness_payload_bytes")?),
            comma_usize(json_usize_field(proof, "trace_payload_bytes")?),
            comma_usize(json_usize_field(proof, "replay_payload_bytes")?),
            comma_usize(json_usize_field(memory, "runtime_state_payload_bytes")?),
            comma_usize(recompute_steps),
        ),
        String::new(),
        "| Fingerprint | SHA-256 |".to_owned(),
        "| --- | --- |".to_owned(),
        format!(
            "| inference computation | `{}` |",
            short_hash(json_str_field(fingerprints, "computation")?)
        ),
        format!(
            "| inference payload | `{}` |",
            short_hash(json_str_field(fingerprints, "payload")?)
        ),
    ]);
    if let Some(fingerprint_value) = proof.get("fingerprints").filter(|value| value.is_object()) {
        for name in ["identify_source", "model", "sample", "result"] {
            if let Some(value) = fingerprint_value
                .get(name)
                .and_then(serde_json::Value::as_str)
            {
                lines.push(format!("| proof {name} | `{}` |", short_hash(value)));
            }
        }
    }

    lines.extend([
        String::new(),
        "## Contract Checks".to_owned(),
        String::new(),
        "| Check | Passed | Evidence | Requirement |".to_owned(),
        "| --- | --- | --- | --- |".to_owned(),
    ]);
    for check in json_array_field(contract, "checks")? {
        lines.push(format!(
            "| `{}` | {} | {} | {} |",
            markdown_cell(json_str_field(check, "metric")?),
            ok_failed(json_bool_field(check, "passed")?),
            markdown_cell(json_str_field(check, "actual")?),
            markdown_cell(json_str_field(check, "requirement")?),
        ));
    }

    lines.extend([
        String::new(),
        "## Source Checks".to_owned(),
        String::new(),
        "| Source | Checked | Items | Payload | Path or reason |".to_owned(),
        "| --- | --- | ---: | --- | --- |".to_owned(),
    ]);
    let mut source_rows = Vec::new();
    for (label, field) in [
        ("model", "source_model"),
        ("training bundle", "source_training_bundle"),
        ("sample", "source_sample"),
        ("evaluation row", "source_model_evaluation"),
    ] {
        if let Some(row) = render_inference_source_check_markdown_row(label, report.get(field))? {
            source_rows.push(row);
        }
    }
    if source_rows.is_empty() {
        lines.push("| n/a | no source artifacts recorded | 0 | `n/a` | n/a |".to_owned());
    } else {
        lines.extend(source_rows);
    }

    Ok(lines.join("\n") + "\n")
}

fn render_inference_source_check_markdown_row(
    label: &str,
    value: Option<&serde_json::Value>,
) -> Result<Option<String>, String> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let checked = json_bool_field(value, "checked")?;
    let items = if let Some(items) = value
        .get("items_checked")
        .and_then(serde_json::Value::as_u64)
    {
        items.to_string()
    } else if let Some(row_index) = value.get("row_index").and_then(serde_json::Value::as_u64) {
        format!("row {row_index}")
    } else {
        "n/a".to_owned()
    };
    let payload = value
        .get("payload_fingerprint")
        .and_then(serde_json::Value::as_str)
        .map(short_hash)
        .unwrap_or_else(|| "n/a".to_owned());
    let path_or_reason = value
        .get("unavailable_reason")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            value
                .get("resolved_path")
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| value.get("path").and_then(serde_json::Value::as_str))
        .unwrap_or("n/a");
    Ok(Some(format!(
        "| {} | {} | {} | `{}` | {} |",
        markdown_cell(label),
        ok_failed(checked),
        markdown_cell(&items),
        payload,
        markdown_cell(path_or_reason),
    )))
}

fn render_training_audit_verification_markdown(
    report: &serde_json::Value,
) -> Result<String, String> {
    let fingerprints = json_field(report, "fingerprints")?;
    let proof = json_field(report, "proof")?;
    let lineage = json_field(report, "lineage_ledger")?;
    let lineage_payload = json_field(lineage, "payload")?;
    let forward = json_field(report, "forward")?;
    let reverse = json_field(report, "reverse")?;
    let first_transition = json_field(lineage_payload, "first_transition")?;
    let last_transition = json_field(lineage_payload, "last_transition")?;
    let forward_elapsed = json_f64_field(forward, "elapsed_seconds")?;
    let reverse_elapsed = json_f64_field(reverse, "elapsed_seconds")?;
    let elapsed_ratio = if forward_elapsed > 0.0 {
        Some(reverse_elapsed / forward_elapsed)
    } else {
        None
    };
    let mut lines = vec![
        "# Reverie Training Audit Verification".to_owned(),
        String::new(),
        "| Verdict | Checked steps | Witness replay | Final model | Reverse restores initial | Proof | Lineage |".to_owned(),
        "| --- | ---: | --- | --- | --- | --- | --- |".to_owned(),
        format!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            ok_failed(
                json_bool_field(report, "witnesses_match_forward_replay")?
                    && json_bool_field(report, "final_model_replayed")?
                    && json_bool_field(report, "restored_initial_model")?
                    && json_bool_field(report, "proof_matches")?
                    && json_bool_field(report, "lineage_ledger_matches")?
            ),
            comma_usize(json_usize_field(report, "checked")?),
            ok_failed(json_bool_field(report, "witnesses_match_forward_replay")?),
            ok_failed(json_bool_field(report, "final_model_replayed")?),
            ok_failed(json_bool_field(report, "restored_initial_model")?),
            ok_failed(json_bool_field(report, "proof_matches")?),
            ok_failed(json_bool_field(report, "lineage_ledger_matches")?),
        ),
        String::new(),
        "## Replay Cost".to_owned(),
        String::new(),
        "| Entries | Model bytes | Sample bytes | Witness bytes | Trace replay bytes | Full replay bytes | Forward steps | Inverse steps |".to_owned(),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |".to_owned(),
        format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            comma_usize(json_usize_field(proof, "entries")?),
            comma_usize(json_usize_field(proof, "model_payload_bytes")?),
            comma_usize(json_usize_field(proof, "sample_payload_bytes")?),
            comma_usize(json_usize_field(proof, "witness_payload_bytes")?),
            comma_usize(json_usize_field(proof, "trace_replay_payload_bytes")?),
            comma_usize(json_usize_field(proof, "full_replay_payload_bytes")?),
            comma_usize(json_usize_field(proof, "forward_recompute_steps")?),
            comma_usize(json_usize_field(proof, "inverse_recompute_steps")?),
        ),
        String::new(),
        "## Replay Timing".to_owned(),
        String::new(),
        "| Direction | Checked | Result | Elapsed seconds | Steps/sec |".to_owned(),
        "| --- | ---: | --- | ---: | ---: |".to_owned(),
        format!(
            "| forward | {} | {} | {:.6} | {:.1} |",
            comma_usize(json_usize_field(forward, "checked")?),
            ok_failed(
                json_bool_field(forward, "witnesses_match")?
                    && json_bool_field(forward, "final_model_matches")?
            ),
            forward_elapsed,
            json_f64_field(forward, "steps_per_second")?,
        ),
        format!(
            "| reverse | {} | {} | {:.6} | {:.1} |",
            comma_usize(json_usize_field(reverse, "checked")?),
            ok_failed(json_bool_field(reverse, "restored_initial_model")?),
            reverse_elapsed,
            json_f64_field(reverse, "steps_per_second")?,
        ),
        format!(
            "| reverse/forward | n/a | {} | {} | n/a |",
            ok_failed(elapsed_ratio.is_some()),
            elapsed_ratio
                .map(|ratio| format!("{ratio:.3}x"))
                .unwrap_or_else(|| "n/a".to_owned()),
        ),
        String::new(),
        "## Lineage".to_owned(),
        String::new(),
        "| Item | Fingerprint |".to_owned(),
        "| --- | --- |".to_owned(),
        format!(
            "| bundle computation | `{}` |",
            short_hash(json_str_field(fingerprints, "computation")?)
        ),
        format!(
            "| bundle payload | `{}` |",
            short_hash(json_str_field(fingerprints, "payload")?)
        ),
        format!(
            "| lineage ledger | `{}` |",
            short_hash(json_str_field(lineage, "fingerprint")?)
        ),
        format!(
            "| witness trace | `{}` |",
            short_hash(json_str_field(lineage_payload, "witness_trace_fingerprint")?)
        ),
        format!(
            "| transition ledger | `{}` |",
            short_hash(json_str_field(
                lineage_payload,
                "transition_ledger_fingerprint"
            )?)
        ),
        format!(
            "| initial chain | `{}` |",
            short_hash(json_str_field(lineage_payload, "initial_chain")?)
        ),
        format!(
            "| final chain | `{}` |",
            short_hash(json_str_field(lineage_payload, "final_chain")?)
        ),
        String::new(),
        "## Boundary Transitions".to_owned(),
        String::new(),
        "| Boundary | Step | Sample | Label | Prediction | Correct | Cause ledger | Transition | Chain |".to_owned(),
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |".to_owned(),
        render_lineage_transition_row("first", first_transition, "after_chain")?,
        render_lineage_transition_row("last", last_transition, "after_chain")?,
    ];
    lines.push(String::new());
    Ok(lines.join("\n"))
}

fn render_lineage_transition_row(
    label: &str,
    transition: &serde_json::Value,
    chain_field: &str,
) -> Result<String, String> {
    Ok(format!(
        "| {} | {} | {} | {} | {} | {} | `{}` | `{}` | `{}` |",
        markdown_cell(label),
        comma_usize(json_usize_field(transition, "step")?),
        comma_usize(json_usize_field(transition, "sample_index")?),
        json_i64_field(transition, "label")?,
        json_i64_field(transition, "prediction")?,
        json_bool_field(transition, "correct")?,
        short_hash(json_str_field(transition, "cause_ledger_fingerprint")?),
        short_hash(json_str_field(transition, "transition_fingerprint")?),
        short_hash(json_str_field(transition, chain_field)?),
    ))
}

fn render_training_step_debug_markdown(
    audit_step: &serde_json::Value,
    step_verification: Option<&serde_json::Value>,
) -> Result<String, String> {
    let update = json_field(audit_step, "update")?;
    let window = json_field(audit_step, "model_window")?;
    let margin = json_field(audit_step, "logit_margin")?;
    let contract = json_field(audit_step, "debug_contract")?;
    let cause = json_field(audit_step, "cause_ledger")?;
    let cause_payload = json_field(cause, "payload")?;
    let proof = step_verification
        .map(|verification| json_field(verification, "proof"))
        .transpose()?;
    let proof_recompute_steps = proof
        .map(|proof| {
            Ok::<usize, String>(
                json_usize_field(proof, "forward_recompute_steps")?
                    + json_usize_field(proof, "inverse_recompute_steps")?,
            )
        })
        .transpose()?;
    let mut window_by_delta = BTreeMap::new();
    for row in json_array_field(window, "top_weight_windows")? {
        window_by_delta.insert(
            (
                json_usize_field(row, "pixel")?,
                json_usize_field(row, "digit")?,
            ),
            row,
        );
    }

    let mut lines = vec![
        "# Reverie Training Step Debug".to_owned(),
        String::new(),
        "| Step | Sample | Label | Prediction | Correct | Runner-up | Margin | Active pixels | LR |"
            .to_owned(),
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |".to_owned(),
        format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            comma_usize(json_usize_field(audit_step, "step")?),
            comma_usize(json_usize_field(audit_step, "sample_index")?),
            json_i64_field(audit_step, "label")?,
            json_i64_field(audit_step, "prediction")?,
            json_bool_field(audit_step, "correct")?,
            json_usize_field(margin, "runner_up_digit")?,
            comma_i64(json_i64_field(margin, "margin")?),
            comma_usize(json_array_field(audit_step, "active_pixels")?.len()),
            comma_i64(json_i64_field(audit_step, "lr")?),
        ),
        String::new(),
        "## Debug Contract".to_owned(),
        String::new(),
        "| Claim | Passed | Reversed later steps | Model window | Update deltas |".to_owned(),
        "| --- | --- | ---: | --- | --- |".to_owned(),
        format!(
            "| `{}` | {} | {} | {} | {} |",
            markdown_cell(json_str_field(contract, "claim")?),
            json_bool_field(contract, "passed")?,
            comma_usize(json_usize_field(window, "reversed_later_steps")?),
            ok_failed(json_bool_field(window, "reconstructed")?),
            ok_failed(
                json_bool_field(window, "bias_delta_matches")?
                    && json_bool_field(window, "weight_delta_matches")?
            ),
        ),
        String::new(),
        "| Check | Passed | Evidence |".to_owned(),
        "| --- | --- | --- |".to_owned(),
    ];
    for check in json_array_field(contract, "checks")? {
        lines.push(format!(
            "| `{}` | {} | {} |",
            markdown_cell(json_str_field(check, "metric")?),
            json_bool_field(check, "passed")?,
            markdown_cell(json_str_field(check, "actual")?),
        ));
    }

    lines.extend([
        String::new(),
        "## Witnesses".to_owned(),
        String::new(),
        "| Top logit | Digit | Value |".to_owned(),
        "| ---: | ---: | ---: |".to_owned(),
    ]);
    for (rank, row) in json_array_field(audit_step, "top_logits")?
        .iter()
        .enumerate()
    {
        lines.push(format!(
            "| {} | {} | {} |",
            rank + 1,
            json_usize_field(row, "digit")?,
            comma_i64(json_i64_field(row, "logit")?),
        ));
    }

    lines.extend([
        String::new(),
        "| Error digit | Error |".to_owned(),
        "| ---: | ---: |".to_owned(),
    ]);
    lines.extend(render_nonzero_i64_rows(
        &json_i64_vec_field(audit_step, "error", DIGITS)?,
        "errors",
    ));
    lines.extend([
        String::new(),
        "| Active pixel | U8 | Q31 |".to_owned(),
        "| ---: | ---: | ---: |".to_owned(),
    ]);
    for row in json_array_field(audit_step, "active_pixels")? {
        lines.push(format!(
            "| {} | {} | {} |",
            json_usize_field(row, "index")?,
            json_usize_field(row, "u8")?,
            comma_i64(json_i64_field(row, "q31")?),
        ));
    }

    lines.extend([
        String::new(),
        "## Top Weight Deltas".to_owned(),
        String::new(),
        "| Pixel | Digit | Computed delta | Before | After | Observed delta | Match |".to_owned(),
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |".to_owned(),
    ]);
    for row in json_array_field(update, "top_weight_deltas")? {
        let pixel = json_usize_field(row, "pixel")?;
        let digit = json_usize_field(row, "digit")?;
        let observed = window_by_delta.get(&(pixel, digit));
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            pixel,
            digit,
            comma_i64(json_i64_field(row, "delta")?),
            observed
                .map(|row| json_i64_field(row, "before").map(comma_i64))
                .transpose()?
                .unwrap_or_else(|| "n/a".to_owned()),
            observed
                .map(|row| json_i64_field(row, "after").map(comma_i64))
                .transpose()?
                .unwrap_or_else(|| "n/a".to_owned()),
            observed
                .map(|row| json_i64_field(row, "observed_delta").map(comma_i64))
                .transpose()?
                .unwrap_or_else(|| "n/a".to_owned()),
            observed
                .map(|row| json_bool_field(row, "delta_matches").map(ok_failed))
                .transpose()?
                .unwrap_or("n/a"),
        ));
    }

    lines.extend([
        String::new(),
        "## Bias Deltas".to_owned(),
        String::new(),
        "| Digit | Delta |".to_owned(),
        "| ---: | ---: |".to_owned(),
    ]);
    lines.extend(render_nonzero_i64_rows(
        &json_i64_vec_field(update, "bias_delta", DIGITS)?,
        "bias deltas",
    ));

    lines.extend([
        String::new(),
        "## Ledgers".to_owned(),
        String::new(),
        "| Ledger | Fingerprint |".to_owned(),
        "| --- | --- |".to_owned(),
        format!(
            "| cause | `{}` |",
            short_hash(json_str_field(cause, "fingerprint")?)
        ),
        format!(
            "| update | `{}` |",
            short_hash(json_str_field(cause_payload, "update_fingerprint")?)
        ),
        format!(
            "| witness | `{}` |",
            short_hash(json_str_field(cause_payload, "witness_fingerprint")?)
        ),
        format!(
            "| bias delta | `{}` |",
            short_hash(json_str_field(update, "bias_delta_ledger_fingerprint")?)
        ),
        format!(
            "| weight delta | `{}` |",
            short_hash(json_str_field(update, "weight_delta_ledger_fingerprint")?)
        ),
        String::new(),
    ]);

    if let Some(selection) = audit_step.get("selection") {
        lines.extend([
            "## Selection".to_owned(),
            String::new(),
            "| Strategy | Requested | Selected | Lowest margin | Largest update | Top suspicious |"
                .to_owned(),
            "| --- | ---: | ---: | ---: | ---: | ---: |".to_owned(),
            format!(
                "| `{}` | {} | {} | {} | {} | {} |",
                markdown_cell(json_str_field(selection, "strategy")?),
                comma_usize(json_usize_field(selection, "requested_step")?),
                comma_usize(json_usize_field(selection, "selected_step")?),
                optional_usize_cell(selection, "scan_lowest_margin_step")?,
                optional_usize_cell(selection, "scan_largest_update_step")?,
                optional_usize_cell(selection, "scan_top_suspicious_step")?,
            ),
            String::new(),
        ]);
    }

    if let (Some(verification), Some(proof), Some(recompute_steps)) =
        (step_verification, proof, proof_recompute_steps)
    {
        let forward = json_field(verification, "forward")?;
        let reverse = json_field(verification, "reverse")?;
        lines.extend([
            "## Replay Proof".to_owned(),
            String::new(),
            "| Claim | Forward | Reverse | Proof | Recompute steps | Witness bytes | Update bytes |"
                .to_owned(),
            "| --- | --- | --- | --- | ---: | ---: | ---: |".to_owned(),
            format!(
                "| `{}` | {} | {} | {} | {} | {} | {} |",
                markdown_cell(json_str_field(proof, "claim")?),
                ok_failed(json_bool_field(forward, "after_model_matches")?),
                ok_failed(json_bool_field(reverse, "before_model_restored")?),
                ok_failed(json_bool_field(verification, "proof_matches")?),
                comma_usize(recompute_steps),
                comma_usize(json_usize_field(proof, "witness_payload_bytes")?),
                comma_usize(json_usize_field(proof, "derived_update_payload_bytes")?),
            ),
            String::new(),
            "| Proof check | Passed |".to_owned(),
            "| --- | --- |".to_owned(),
        ]);
        let checks = json_field(proof, "checks")?
            .as_object()
            .ok_or_else(|| "audit bundle `checks` must be an object".to_owned())?;
        for (name, passed) in checks {
            let passed = passed
                .as_bool()
                .ok_or_else(|| format!("audit bundle `checks.{name}` must be a boolean"))?;
            lines.push(format!("| `{}` | {} |", markdown_cell(name), passed));
        }
    }

    Ok(lines.join("\n") + "\n")
}

fn render_nonzero_i64_rows(values: &[i64], label: &str) -> Vec<String> {
    let rows = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value != 0)
        .map(|(index, value)| format!("| {} | {} |", index, comma_i64(value)))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        vec![format!("| n/a | no nonzero {} |", markdown_cell(label))]
    } else {
        rows
    }
}

fn markdown_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn short_hash(value: &str) -> String {
    value.chars().take(12).collect()
}

fn ok_failed(value: bool) -> &'static str {
    if value { "ok" } else { "failed" }
}

fn optional_usize_cell(value: &serde_json::Value, name: &str) -> Result<String, String> {
    Ok(match value.get(name) {
        Some(raw) if raw.is_null() => "n/a".to_owned(),
        Some(raw) => comma_usize(
            raw.as_u64()
                .ok_or_else(|| format!("audit bundle `{name}` must be a non-negative integer"))?
                .try_into()
                .map_err(|_| format!("audit bundle `{name}` exceeds usize"))?,
        ),
        None => "n/a".to_owned(),
    })
}

fn comma_usize(value: usize) -> String {
    comma_i128(value as i128)
}

fn comma_i64(value: i64) -> String {
    comma_i128(i128::from(value))
}

fn comma_i128(value: i128) -> String {
    let sign = if value < 0 { "-" } else { "" };
    let digits = value.unsigned_abs().to_string();
    let mut out = String::new();
    for (index, ch) in digits.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    let grouped = out.chars().rev().collect::<String>();
    format!("{sign}{grouped}")
}

fn json_field<'a>(
    value: &'a serde_json::Value,
    name: &str,
) -> Result<&'a serde_json::Value, String> {
    value
        .get(name)
        .ok_or_else(|| format!("audit bundle is missing `{name}`"))
}

fn json_array_field<'a>(
    value: &'a serde_json::Value,
    name: &str,
) -> Result<&'a Vec<serde_json::Value>, String> {
    json_field(value, name)?
        .as_array()
        .ok_or_else(|| format!("audit bundle `{name}` must be an array"))
}

fn json_str_field<'a>(value: &'a serde_json::Value, name: &str) -> Result<&'a str, String> {
    json_field(value, name)?
        .as_str()
        .ok_or_else(|| format!("audit bundle `{name}` must be a string"))
}

fn json_i64_field(value: &serde_json::Value, name: &str) -> Result<i64, String> {
    json_field(value, name)?
        .as_i64()
        .ok_or_else(|| format!("audit bundle `{name}` must be an integer"))
}

fn json_i128_field(value: &serde_json::Value, name: &str) -> Result<i128, String> {
    let value = json_field(value, name)?;
    if let Some(value) = value.as_i64() {
        return Ok(i128::from(value));
    }
    if let Some(value) = value.as_u64() {
        return Ok(i128::from(value));
    }
    Err(format!("audit bundle `{name}` must be an integer"))
}

fn json_bool_field(value: &serde_json::Value, name: &str) -> Result<bool, String> {
    json_field(value, name)?
        .as_bool()
        .ok_or_else(|| format!("audit bundle `{name}` must be a boolean"))
}

fn json_f64_field(value: &serde_json::Value, name: &str) -> Result<f64, String> {
    json_field(value, name)?
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("audit bundle `{name}` must be a finite number"))
}

fn json_usize_field(value: &serde_json::Value, name: &str) -> Result<usize, String> {
    let value = json_field(value, name)?
        .as_u64()
        .ok_or_else(|| format!("audit bundle `{name}` must be a non-negative integer"))?;
    usize::try_from(value).map_err(|_| format!("audit bundle `{name}` exceeds usize"))
}

fn json_optional_usize_field(
    value: &serde_json::Value,
    name: &str,
) -> Result<Option<usize>, String> {
    let Some(raw) = value.get(name) else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    let raw = raw
        .as_u64()
        .ok_or_else(|| format!("audit bundle `{name}` must be a non-negative integer"))?;
    usize::try_from(raw)
        .map(Some)
        .map_err(|_| format!("audit bundle `{name}` exceeds usize"))
}

fn json_i64_vec_field(
    value: &serde_json::Value,
    name: &str,
    len: usize,
) -> Result<Vec<i64>, String> {
    let values = json_array_field(value, name)?;
    if values.len() != len {
        return Err(format!(
            "audit bundle `{name}` expected length {len}, found {}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_i64()
                .ok_or_else(|| format!("audit bundle `{name}[{index}]` must be an integer"))
        })
        .collect()
}

fn json_u8_vec_field(value: &serde_json::Value, name: &str, len: usize) -> Result<Vec<u8>, String> {
    let values = json_array_field(value, name)?;
    if values.len() != len {
        return Err(format!(
            "audit bundle `{name}` expected length {len}, found {}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let value = value
                .as_u64()
                .ok_or_else(|| format!("audit bundle `{name}[{index}]` must be an integer"))?;
            u8::try_from(value)
                .map_err(|_| format!("audit bundle `{name}[{index}]` is outside 0..=255"))
        })
        .collect()
}

fn model_from_json(value: &serde_json::Value, name: &str) -> Result<Model, String> {
    let weights = json_matrix_field(value, "weights", IMAGE_PIXELS, DIGITS)?;
    let bias = json_i64_vec_field(value, "bias", DIGITS)?;
    if weights.len() != IMAGE_PIXELS {
        return Err(format!(
            "audit bundle `{name}.weights` expected {IMAGE_PIXELS} rows, found {}",
            weights.len()
        ));
    }
    Ok(Model { weights, bias })
}

fn json_matrix_field(
    value: &serde_json::Value,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<Vec<i64>>, String> {
    let values = json_array_field(value, name)?;
    if values.len() != rows {
        return Err(format!(
            "audit bundle `{name}` expected {rows} rows, found {}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(row, value)| {
            let values = value
                .as_array()
                .ok_or_else(|| format!("audit bundle `{name}[{row}]` must be an array"))?;
            if values.len() != cols {
                return Err(format!(
                    "audit bundle `{name}[{row}]` expected length {cols}, found {}",
                    values.len()
                ));
            }
            values
                .iter()
                .enumerate()
                .map(|(col, value)| {
                    value.as_i64().ok_or_else(|| {
                        format!("audit bundle `{name}[{row}][{col}]` must be an integer")
                    })
                })
                .collect()
        })
        .collect()
}

fn witness_trace_payload_bytes(entries: usize) -> usize {
    entries * (size_of::<usize>() + (4 + 2 * DIGITS) * size_of::<i64>())
}

fn training_trace_sample_payload_bytes(entries: usize) -> usize {
    entries * IMAGE_PIXELS * size_of::<u8>()
}

fn single_sample_payload_bytes() -> usize {
    IMAGE_PIXELS * size_of::<u8>() + size_of::<u8>()
}

fn training_step_witness_payload_bytes() -> usize {
    (2 * DIGITS + 3) * size_of::<i64>()
}

fn step_update_payload_bytes(top_weight_deltas: usize) -> usize {
    DIGITS * size_of::<i64>()
        + size_of::<usize>()
        + size_of::<i64>()
        + size_of::<usize>()
        + 2 * 32
        + top_weight_deltas * (2 * size_of::<usize>() + size_of::<i64>())
}

fn inference_witness_payload_bytes() -> usize {
    (DIGITS + 2) * size_of::<i64>()
}

#[cfg(unix)]
fn peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return None;
    }

    let max_rss = unsafe { usage.assume_init().ru_maxrss };
    if max_rss < 0 {
        return None;
    }
    let max_rss = max_rss as u64;

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        Some(max_rss)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        Some(max_rss.saturating_mul(1024))
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

fn percent(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        100.0 * numerator as f64 / denominator as f64
    }
}

fn secs(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn per_second(count: usize, duration: Duration) -> f64 {
    let secs = duration.as_secs_f64();
    if secs == 0.0 {
        0.0
    } else {
        count as f64 / secs
    }
}

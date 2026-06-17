#!/usr/bin/env python3
"""Validate that the external Q31 model import contract is documented."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_IMPORT_DOC = REPO_ROOT / "docs" / "model-import.md"
README = REPO_ROOT / "README.md"
CLI_DOC = REPO_ROOT / "docs" / "cli.md"
CLI_SOURCE = REPO_ROOT / "crates" / "reverie-cli" / "src" / "main.rs"
EXAMPLES_GUIDE = REPO_ROOT / "examples" / "README.md"
RUNNER = REPO_ROOT / "crates" / "reverie-cli" / "src" / "bin" / "reverie-mnist-linear.rs"
EXPORTER_RELATIVE = "scripts/export_q31_linear_model.py"
EXPORTER = REPO_ROOT / EXPORTER_RELATIVE
MLP_EXPORTER_RELATIVE = "scripts/export_q31_mlp_vars.py"
MLP_EXPORTER = REPO_ROOT / MLP_EXPORTER_RELATIVE
TORCH_EXTRACTOR_RELATIVE = "scripts/extract_torch_linear_state_dict.py"
TORCH_EXTRACTOR = REPO_ROOT / TORCH_EXTRACTOR_RELATIVE
REFERENCE_CHECKER_RELATIVE = "scripts/check_q31_inference.py"
REFERENCE_CHECKER = REPO_ROOT / REFERENCE_CHECKER_RELATIVE
MLP_REFERENCE_CHECKER_RELATIVE = "scripts/check_q31_mlp_witness.py"
MLP_REFERENCE_CHECKER = REPO_ROOT / MLP_REFERENCE_CHECKER_RELATIVE

MODEL_IMPORT_DOC_SNIPPETS = {
    "model import flag": ("--import-model-json",),
    "model verifier flag": ("--verify-model",),
    "model inference flag": ("--inspect-model-inference",),
    "evaluation flag": ("--evaluate-model",),
    "artifact comparison flag": ("--compare-artifacts",),
    "external import provenance": ("provenance_kind", "external_import"),
    "source JSON verification field": ("source_model_json_checked",),
    "import format": ("weights_bias_q31_json",),
    "weights field": ('"weights"',),
    "bias field": ('"bias"',),
    "MNIST image width": ("784",),
    "MNIST class count": ("10",),
    "Q31 constant": ("Q31_ONE = 1 << 31",),
    "Q31 integer scale": ("2147483648",),
    "pixel conversion name": ("pixel_q31",),
    "ML profile": ("ml_profile",),
    "Reverie example source": ("examples/mnist_identify.rev",),
    "Q31 exporter": ("scripts/export_q31_linear_model.py",),
    "exporter class-pixel layout": ("--input-layout class-pixel",),
    "exporter float scale": ("--input-scale float",),
    "state dict weight key": ("state_dict", "linear.weight"),
    "PyTorch state dict extractor": ("scripts/extract_torch_linear_state_dict.py",),
    "PyTorch safe loading": ("weights_only=True", "--allow-unsafe-pickle"),
    "PyTorch extraction format": ("torch_linear_state_dict_json",),
    "NPZ exporter": (".npz", "--npz-weights", "--npz-bias"),
    "NPZ C-order contract": ("C-order numeric `.npy` arrays",),
    "Q31 reference checker": ("scripts/check_q31_inference.py",),
    "reference checker expectations": ("--expect-prediction", "--expect-correct"),
    "reference checker Markdown": ("--markdown-output", "prediction, top logits, margin"),
    "reference checker semantics": ("wrapping logit accumulation", "first-maximum `argmax`"),
    "contribution ledger fingerprints": ("contribution-ledger", "SHA-256 fingerprints"),
    "training update ledger fingerprints": ("training-update ledger", "weight-delta ledger fingerprints"),
    "training step debug artifact": ("training-step-debug.md", "selected-update explanation"),
    "north-star readiness table": ("North-Star Readiness", "ml_goal_readiness"),
    "north-star non-goal": ("reversible inspectable deterministic ML kernels",),
    "model capsule artifact": ("model-capsule.json", "MLP witness proof fingerprint"),
    "model capsule profile": ("model-capsule-profile.md", "Model Capsules"),
    "model capsule verifier": ("model-capsule.json", "--verify-pipeline-files"),
    "model capsule verification artifact": ("model-capsule-verification.json", "post-manifest attestation"),
    "ML audit handoff artifact": ("ml-audit-handoff.json", "reviewer entry point"),
    "dedicated model capsule verifier": ("scripts/verify_model_capsule.py", "target/mnist-ml-audit-pipeline"),
    "model capsule JSON verification": ("--json", "--output", "capsule-profile SHA-256"),
    "model capsule profile table": ("Model Capsules", "North-Star Readiness"),
    "generic JSON seed files": ("--vars-json", "floating-point JSON values"),
    "MLP vars exporter": ("scripts/export_q31_mlp_vars.py", "examples/mnist_mlp_witness.rev"),
    "MLP witness checker": ("scripts/check_q31_mlp_witness.py", "--run-output-json"),
}

README_SNIPPETS = {
    "model import docs link": ("[Model import](docs/model-import.md)",),
}

CLI_DOC_SNIPPETS = {
    "import flag": ("--import-model-json",),
    "import format": ("weights_bias_q31_json",),
    "external import provenance": ("external_import",),
    "source JSON verification field": ("source_model_json_checked",),
    "large tensor seed files": ("--vars-json PATH", "Floating-point JSON numbers are rejected"),
    "machine-readable run output": ("--json", "reverie_run_result"),
}

CLI_SOURCE_SNIPPETS = {
    "vars JSON option": ("vars_json", "--vars-json"),
    "vars JSON loader": ("load_vars_json_files", "value_from_json_seed"),
    "run JSON output": ("print_io_state_json", "reverie_run_result"),
    "exact integer JSON seeds": ("floating-point JSON numbers are not supported",),
}

EXAMPLES_GUIDE_SNIPPETS = {
    "import flag": ("--import-model-json",),
    "external import provenance": ("external_import",),
    "artifact profile": ("ml_profile",),
    "MLP vars exporter": ("scripts/export_q31_mlp_vars.py", "--vars-json"),
    "MLP witness checker": ("scripts/check_q31_mlp_witness.py", "--run-output-json"),
}

RUNNER_SNIPPETS = {
    "CLI import argument": ("import_model_json",),
    "model output requirement": ("--import-model-json requires --model-output",),
    "import format": ("weights_bias_q31_json",),
    "external import provenance": ("external_import",),
    "source JSON verification field": ("source_model_json_checked",),
    "Q31 constant": ("Q31_ONE", "1_i64 << 31"),
    "pixel conversion": ("pixel_to_q31", "/ 255"),
    "contribution ledger fingerprint": ("contribution_ledger_fingerprint",),
    "training update ledger fingerprint": ("weight_delta_ledger_fingerprint",),
}

EXPORTER_SNIPPETS = {
    "build-independent self-test": ("--self-test",),
    "import format": ("weights_bias_q31_json",),
    "Q31 constant": ("Q31_ONE = 1 << 31",),
    "pixel-class layout": ("pixel-class",),
    "class-pixel layout": ("class-pixel",),
    "state dict key": ("state_dict", "linear.weight"),
    "NPZ reader": ("read_npz_model", "zipfile.ZipFile"),
    "NPY header parser": ("ast.literal_eval", "fortran_order"),
    "NPZ CLI flags": ("--npz-weights", "--npz-bias"),
    "bool is not numeric": ("isinstance(value, bool)",),
    "i64 range check": ("I64_MIN", "I64_MAX"),
}

MLP_EXPORTER_SNIPPETS = {
    "build-independent self-test": ("--self-test",),
    "vars JSON output": ("--vars-json",),
    "metadata sidecar": ("--metadata-output",),
    "MLP witness target": ("examples/mnist_mlp_witness.rev",),
    "Q31 constant": ("Q31_ONE = 1 << 31",),
    "PyTorch layer keys": ("fc1.weight", "fc2.weight"),
    "layout modes": ("torch", "reverie"),
    "seed keys": ('"w1"', '"b1"', '"w2"', '"b2"'),
    "NPZ reader reuse": ("linear_export.read_npz_member",),
}

MLP_REFERENCE_CHECKER_SNIPPETS = {
    "build-independent self-test": ("--self-test",),
    "MLP witness target": ("examples/mnist_mlp_witness.rev",),
    "vars JSON input": ("--vars-json",),
    "run output JSON input": ("--run-output-json",),
    "Q31 constant": ("Q31_ONE = 1 << 31",),
    "hidden layer width": ("HIDDEN = 16",),
    "fixed multiply": ("fixed_mul_q31",),
    "witness field": ("hidden_delta_tape",),
    "prediction expectation": ("--expect-predictions",),
    "correctness expectation": ("--expect-correct",),
    "proof-cost claim": ("deterministic_q31_mlp_witness_replay",),
    "witness payload bytes": ("witness_payload_bytes",),
    "trace payload bytes": ("trace_payload_bytes",),
    "recomputed update payload bytes": ("recomputed_update_payload_bytes",),
    "run output kind": ("reverie_run_result",),
}

TORCH_EXTRACTOR_SNIPPETS = {
    "build-independent self-test": ("--self-test",),
    "safe checkpoint loading": ("weights_only=True",),
    "unsafe legacy opt-in": ("--allow-unsafe-pickle",),
    "torch loader": ("torch.load",),
    "output format": ("torch_linear_state_dict_json",),
    "state dict key": ("state_dict", "linear.weight"),
    "bias key": ("linear.bias",),
    "tensor conversion": ("detach", "tolist"),
    "bool is not numeric": ("isinstance(value, bool)",),
}

REFERENCE_CHECKER_SNIPPETS = {
    "build-independent self-test": ("--self-test",),
    "Q31 constant": ("Q31_ONE = 1 << 31",),
    "pixel scaling": ("pixel_to_q31", "/ 255"),
    "fixed multiply": ("fixed_mul_q31", ">> 31"),
    "wrapping arithmetic": ("wrap_i64", "U64_MOD"),
    "first argmax": ("argmax_first", "value > best_value"),
    "top logits": ("top_logits",),
    "attribution reconstruction": ("inference_attribution",),
    "margin contribution proof": ("top_margin_contributions", "matches_margin"),
    "sample-set selection": ("--sample-index",),
    "batch sample-set evaluation": ("--all-samples",),
    "batch evaluation kind": ("reverie_q31_linear_reference_evaluation",),
    "Markdown explanation option": ("--markdown-output",),
    "Markdown explanation title": ("# Reverie Q31 Inference Explanation",),
    "low-margin ranking": ("top_low_margin",),
    "label summaries": ("by_label",),
    "sample-set kind": ("reverie_mnist_linear_q31_samples",),
    "prediction expectation": ("--expect-prediction",),
    "correctness expectation": ("--expect-correct",),
    "sample shape": ("image_u8", "784"),
}


def validate_required_snippets(
    label: str, text: str, snippets: dict[str, tuple[str, ...]]
) -> list[str]:
    missing = [
        name
        for name, required in snippets.items()
        if not all(snippet in text for snippet in required)
    ]
    if not missing:
        return []
    return [f"{label} missing required model-import detail: {', '.join(sorted(missing))}"]


def validate_repo() -> list[str]:
    return [
        *validate_required_snippets(
            "docs/model-import.md",
            MODEL_IMPORT_DOC.read_text(encoding="utf-8"),
            MODEL_IMPORT_DOC_SNIPPETS,
        ),
        *validate_required_snippets(
            "README.md",
            README.read_text(encoding="utf-8"),
            README_SNIPPETS,
        ),
        *validate_required_snippets(
            "docs/cli.md",
            CLI_DOC.read_text(encoding="utf-8"),
            CLI_DOC_SNIPPETS,
        ),
        *validate_required_snippets(
            "reverie CLI",
            CLI_SOURCE.read_text(encoding="utf-8"),
            CLI_SOURCE_SNIPPETS,
        ),
        *validate_required_snippets(
            "examples/README.md",
            EXAMPLES_GUIDE.read_text(encoding="utf-8"),
            EXAMPLES_GUIDE_SNIPPETS,
        ),
        *validate_required_snippets(
            "reverie-mnist-linear runner",
            RUNNER.read_text(encoding="utf-8"),
            RUNNER_SNIPPETS,
        ),
        *validate_required_snippets(
            "scripts/export_q31_linear_model.py",
            EXPORTER.read_text(encoding="utf-8"),
            EXPORTER_SNIPPETS,
        ),
        *validate_required_snippets(
            "scripts/export_q31_mlp_vars.py",
            MLP_EXPORTER.read_text(encoding="utf-8"),
            MLP_EXPORTER_SNIPPETS,
        ),
        *validate_required_snippets(
            "scripts/extract_torch_linear_state_dict.py",
            TORCH_EXTRACTOR.read_text(encoding="utf-8"),
            TORCH_EXTRACTOR_SNIPPETS,
        ),
        *validate_required_snippets(
            "scripts/check_q31_inference.py",
            REFERENCE_CHECKER.read_text(encoding="utf-8"),
            REFERENCE_CHECKER_SNIPPETS,
        ),
        *validate_required_snippets(
            "scripts/check_q31_mlp_witness.py",
            MLP_REFERENCE_CHECKER.read_text(encoding="utf-8"),
            MLP_REFERENCE_CHECKER_SNIPPETS,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the documented external Q31 model import contract."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run build-independent model-import checker self-tests and exit.",
    )
    return parser.parse_args()


def run_self_tests() -> int:
    valid = validate_required_snippets(
        "fixture",
        "alpha beta gamma",
        {
            "alpha": ("alpha",),
            "compound": ("beta", "gamma"),
        },
    )
    if valid:
        for error in valid:
            print(f"error: valid self-test fixture failed: {error}", file=sys.stderr)
        return 1

    invalid = validate_required_snippets(
        "fixture",
        "alpha beta",
        {
            "missing single": ("delta",),
            "missing compound": ("alpha", "gamma"),
        },
    )
    expected_needles = ["missing single", "missing compound"]
    if not invalid or not all(needle in invalid[0] for needle in expected_needles):
        print(
            "error: negative self-test fixture did not report all missing details",
            file=sys.stderr,
        )
        return 1

    print("ok: model import docs checker self-tests passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()

    errors = validate_repo()
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print("ok: validated model import docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

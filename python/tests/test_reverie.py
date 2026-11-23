"""Tests for the `reverie` Python package. Expectations mirror
`crates/reverie-cli/tests/cli.rs`."""

from __future__ import annotations

import doctest
import json
import threading
from pathlib import Path
from typing import Any, Dict

import pytest

import reverie
from conftest import REPO_ROOT, var_args

FIB_SEED = {"n": 7, "i": 0, "a": 0, "b": 1}
FIB_DONE = {"a": 13, "b": 21, "i": 7, "n": 7}


# -- load / loads / check ----------------------------------------------------


def test_run_fib_example_accepts_seeded_integer_vars(example):
    assert example("fib").run(n=7, i=0, a=0, b=1) == FIB_DONE


def test_run_fib_example_can_force_tree_engine(example):
    program = example("fib", engine="tree")
    assert program.engine == "tree"
    assert program.run(**FIB_SEED) == FIB_DONE


def test_reverse_fib_example_accepts_seeded_integer_vars(example):
    assert example("fib").reverse(n=7, i=7, a=13, b=21) == {"a": 0, "b": 1, "i": 0, "n": 7}


def test_reverse_undoes_run(example):
    for engine in ("slot", "tree"):
        fib = example("fib", engine=engine)
        assert fib.reverse(fib.run(FIB_SEED)) == FIB_SEED


def test_store_may_be_mapping_kwargs_or_both(example):
    fib = example("fib")
    assert fib.run(FIB_SEED) == FIB_DONE
    assert fib.run({"n": 7, "i": 0}, a=0, b=1) == FIB_DONE
    # Keyword arguments win over the mapping.
    assert fib.run({"n": 7, "i": 0, "a": 0, "b": 99}, b=1) == FIB_DONE


def test_load_accepts_pathlike_and_records_path(examples):
    program = reverie.load(examples / "fib.rev")
    assert program.path == str(examples / "fib.rev")
    assert "a <=> b" in program.source
    assert "fib.rev" in repr(program)


def test_loads_has_no_path():
    program = reverie.loads("x += 1")
    assert program.path is None
    assert program.run(x=41) == {"x": 42}


def test_irreversible_update_raises_check_error(example):
    with pytest.raises(reverie.CheckError) as info:
        example("irreversible_update")
    error = info.value
    assert "irreversible update" in error.message
    assert (error.line, error.column) == (1, 6)
    assert error.span == (5, 10)
    assert "irreversible_update.rev:1:6" in error.rendered
    assert "\x1b[" not in error.rendered, "diagnostics are rendered without color"


def test_x_plus_equals_x_raises_check_error():
    with pytest.raises(reverie.CheckError, match="irreversible update") as info:
        reverie.loads("x += x")
    assert info.value.line == 1
    assert "<string>:1:" in info.value.rendered


def test_parse_errors_are_check_errors():
    with pytest.raises(reverie.ParseError) as info:
        reverie.loads("x += ")
    assert isinstance(info.value, reverie.CheckError)
    assert (info.value.line, info.value.column) == (1, 6)
    assert "the parser got stuck here" in info.value.rendered


def test_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        reverie.load(tmp_path / "missing.rev")


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError, match="unknown engine"):
        reverie.loads("skip", engine="jit")


# -- seeds -------------------------------------------------------------------


def test_missing_seeds_raise_seed_error(example):
    with pytest.raises(reverie.SeedError, match="missing seed"):
        example("fib").run(n=7)


def test_unexpected_seeds_raise_seed_error(example):
    with pytest.raises(reverie.SeedError, match="unexpected seed.*`zzz`"):
        example("fib").run(zzz=1, **FIB_SEED)


def test_bad_seed_values_raise_seed_error(example):
    fib = example("fib")
    with pytest.raises(reverie.SeedError, match="seed `b`"):
        fib.run(n=7, i=0, a=0, b="1")
    with pytest.raises(reverie.SeedError, match="64-bit"):
        fib.run(n=7, i=0, a=0, b=2**70)
    assert isinstance(reverie.SeedError("x"), ValueError)


def test_seed_types_are_checked_against_program(example):
    # `n` is compared with an int, so seeding it with a bool fails the check.
    with pytest.raises(reverie.CheckError):
        example("fib").run(n=True, i=0, a=0, b=1)


def test_heterogeneous_arrays_are_rejected():
    with pytest.raises(reverie.SeedError, match="heterogeneous"):
        reverie.loads("xs[0] += 1").run(xs=[1, True])


# -- values ------------------------------------------------------------------


def test_arrays_round_trip_as_lists(example):
    program = example("array")
    out = program.run(xs=[10, 20, 30], delta=5)
    assert out == {"delta": 5, "xs": [30, 25, 10]}
    assert isinstance(out["xs"], list)
    assert program.reverse(out) == {"delta": 5, "xs": [10, 20, 30]}
    # Tuples are accepted as arrays too.
    assert program.run(xs=(10, 20, 30), delta=5) == out


def test_bools_stay_bools(example):
    out = example("bool_toggle").run(flag=False)
    assert out == {"flag": True}
    assert out["flag"] is True


def test_nested_arrays():
    program = reverie.loads("m[1][0] += 5")
    assert program.run(m=[[1, 2], [3, 4]]) == {"m": [[1, 2], [8, 4]]}


def test_stacks():
    program = reverie.loads("push(x, s)")
    out = program.run(x=3, s=reverie.Stack([2, 1]))
    assert out == {"x": 0, "s": reverie.Stack([3, 2, 1])}
    assert out["s"].top == 3
    assert program.reverse(out) == {"x": 3, "s": reverie.Stack([2, 1])}
    assert program.run(x=7, s=reverie.Stack()) == {"x": 0, "s": reverie.Stack([7])}


def test_stack_type():
    stack = reverie.Stack([3, 2, 1])
    assert list(stack) == [3, 2, 1]
    assert len(stack) == 3 and stack[0] == 3 and stack.top == 3
    assert stack.pop() == (3, reverie.Stack([2, 1]))
    assert stack.push(4) == reverie.Stack([4, 3, 2, 1])
    assert stack != [3, 2, 1] and stack != (3, 2, 1)
    assert hash(stack) == hash(reverie.Stack((3, 2, 1)))
    assert repr(reverie.Stack()) == "Stack()" and not reverie.Stack()
    with pytest.raises(TypeError):
        reverie.Stack([True])


def test_numpy_values_are_accepted():
    np = pytest.importorskip("numpy")
    program = reverie.loads("xs[1] += d")
    assert program.run(xs=np.array([1, 2, 3]), d=np.int64(4)) == {"d": 4, "xs": [1, 6, 3]}
    assert reverie.loads("flag ^= true").run(flag=np.bool_(False)) == {"flag": True}
    grid = reverie.loads("m[1][0] += 5").run(m=np.array([[1, 2], [3, 4]]))
    assert grid == {"m": [[1, 2], [8, 4]]}


# -- runtime errors ----------------------------------------------------------


def test_failed_assertion_raises_execution_error(example):
    with pytest.raises(reverie.ExecutionError, match="assertion") as info:
        example("assert_failure").run(x=2)
    assert (info.value.line, info.value.column) == (1, 1)
    assert "runtime failed here" in info.value.rendered


def test_execution_releases_the_gil(example):
    fib = example("fib")
    results = []

    def work():
        results.append(fib.run(FIB_SEED))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [FIB_DONE] * 8


# -- timeline / inverse ------------------------------------------------------


def test_timeline_frames(example):
    frames = example("fib").timeline(n=2, i=0, a=0, b=1)
    assert frames[0].step == 0 and frames[0].label == "start"
    assert frames[0].store == {"a": 0, "b": 1, "i": 0, "n": 2}
    assert frames[-1].store == {"a": 1, "b": 2, "i": 2, "n": 2}
    assert [frame.step for frame in frames] == list(range(len(frames)))
    assert any(frame.line is not None for frame in frames)
    assert "Frame(step=0" in repr(frames[0])


def test_inverse_source_matches_cli_invert(example):
    inverse = example("fib").inverse_source()
    for fragment in ("from i == n do", "i -= 1;", "a <=> b;", "a -= b", "until i == 0"):
        assert fragment in inverse
    # The inverse is itself a valid program that undoes the original.
    assert reverie.loads(inverse).run(FIB_DONE) == FIB_SEED


def test_formatted_source_is_canonical():
    assert reverie.loads("x+=1").formatted_source() == "x += 1\n"


# -- roundtrip proofs --------------------------------------------------------


def test_roundtrip_proof_passes(example):
    proof = example("fib").roundtrip(FIB_SEED)
    assert proof["kind"] == "reverie_roundtrip_result"
    assert proof["schema"] == "reverie_roundtrip_proof_v1"
    assert proof["passed"] is True
    payload = proof["payload"]
    assert payload["baseline"]["store"] == FIB_SEED
    assert payload["forward"]["store"] == FIB_DONE
    assert payload["restored"]["store"] == FIB_SEED


@pytest.mark.parametrize("engine", ["slot", "tree"])
def test_roundtrip_proof_matches_cli_proof_output(monkeypatch, reverie_cli, engine):
    """The Python proof is the exact JSON `reverie roundtrip --proof-output`
    writes for the same path, seeds and engine."""
    path = "examples/fib.rev"  # relative, as typed on the CLI from the repo root
    cli_proof = reverie_cli(
        "roundtrip", path, *var_args(FIB_SEED), "--engine", engine, "--json"
    )
    monkeypatch.chdir(REPO_ROOT)
    py_proof = reverie.load(path, engine=engine).roundtrip(FIB_SEED)
    assert py_proof == cli_proof
    assert py_proof["fingerprint"] == cli_proof["fingerprint"]


@pytest.mark.parametrize(
    "name, store",
    [
        ("fib", FIB_SEED),
        ("array", {"xs": [10, 20, 30], "delta": 5}),
        ("bool_toggle", {"flag": False}),
        ("stack", {}),
    ],
)
def test_run_matches_cli_run_json(example, reverie_cli, examples, name, store):
    cli = reverie_cli("run", str(examples / f"{name}.rev"), *var_args(store), "--json")
    assert _json_store(example(name).run(store)) == cli["store"]


def test_verify_saved_proof_file(tmp_path, example):
    proof = example("fib").roundtrip(FIB_SEED)
    saved = tmp_path / "proof.json"
    saved.write_text(json.dumps(proof))
    report = reverie.verify(saved)
    assert report["kind"] == "reverie_roundtrip_verification"
    assert report["passed"] is True, report["checks"]
    assert report["proof_path"] == str(saved)
    assert all(report["checks"].values())
    assert reverie.verify(proof)["passed"] is True


def test_verify_detects_tampering(example):
    proof = example("fib").roundtrip(FIB_SEED)
    proof["payload"]["forward"]["store"]["a"] = 14
    report = reverie.verify(proof)
    assert report["passed"] is False
    assert report["checks"]["payload_fingerprint"] is False


def test_verify_in_memory_program_needs_source():
    source = REPO_ROOT.joinpath("examples/fib.rev").read_text()
    program = reverie.loads(source)
    proof = program.roundtrip(FIB_SEED)
    assert proof["payload"]["program"]["file"] == "<string>"
    unreplayed = reverie.verify(proof)
    assert unreplayed["passed"] is False and unreplayed["checks"]["source_readable"] is False
    assert reverie.verify(proof, source=program)["passed"] is True
    assert reverie.verify(proof, source=source)["passed"] is True
    assert reverie.verify(proof, source="skip")["checks"]["source_hash_matches"] is False


# -- package -----------------------------------------------------------------


def test_docstring_examples():
    failures, _ = doctest.testmod(reverie)
    assert failures == 0


def test_public_api():
    assert reverie.__version__
    for name in reverie.__all__:
        assert hasattr(reverie, name), name
    assert issubclass(reverie.ExecutionError, reverie.ReverieError)


def _json_store(store: Dict[str, Any]) -> Dict[str, Any]:
    """Python store -> the CLI's `--json` store shape (stacks as objects)."""

    def convert(value: Any) -> Any:
        if isinstance(value, reverie.Stack):
            return {"stack": list(value)}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return {name: convert(value) for name, value in store.items()}


# -- lab / ledger ------------------------------------------------------------


def _tiny_run(seed=0, n=10, d=5, h=4, c=3, epochs=2):
    import random

    from reverie import lab

    rng = random.Random(seed)
    images = [[lab.to_q31(rng.uniform(-1, 1)) for _ in range(d)] for _ in range(n)]
    labels = [rng.randrange(c) for _ in range(n)]
    order = [i for _ in range(epochs) for i in rng.sample(range(n), n)]
    run = lab.SGDRewind(
        images=images, labels=labels, order=order, hidden=h, classes=c, lr=lab.to_q31(0.1)
    )
    weights = {
        "w1": [[lab.to_q31(rng.gauss(0, 0.4)) for _ in range(h)] for _ in range(d)],
        "b1": [0] * h,
        "w2": [[lab.to_q31(rng.gauss(0, 0.4)) for _ in range(c)] for _ in range(h)],
        "b2": [0] * c,
    }
    return run, weights


@pytest.mark.parametrize("engine", ["slot", "tree"])
def test_sgd_rewinds_exactly_to_any_step(engine):
    from reverie import lab

    run, weights0 = _tiny_run()
    run.program = reverie.loads(run.source, engine=engine)
    final = run.train(weights0)
    assert run.weights(final) != weights0
    for step in (0, 3, run.steps - 1):
        rewound = run.rewind(final, step)
        replayed = run.train(weights0, stop=step - 1) if step else None
        expected = run.weights(replayed) if replayed else weights0
        assert run.weights(rewound) == expected
        assert all(row == [0] * len(row) for row in rewound["err_tape"][step:])
    assert lab.SGDRewind.witness(final)["err_tape"][0] != [0] * 3


def test_unlearning_is_bit_identical_to_retraining():
    run, weights0 = _tiny_run()
    final = run.train(weights0)
    for sample in range(run.n_samples):
        active = [0 if index == sample else 1 for index in run.data["order"]]
        retrained = run.train(weights0, active=active)
        unlearned = run.unlearn(final, sample)
        assert run.weights(unlearned) == run.weights(retrained)
        assert run.weights(unlearned) != run.weights(final)


def test_training_matches_plain_python_sgd():
    """The witness bookkeeping does not change the arithmetic."""
    run, weights0 = _tiny_run()
    final = run.train(weights0)
    fm = lambda a, b: (a * b) >> 31  # noqa: E731
    w1 = [row[:] for row in weights0["w1"]]
    b1, w2, b2 = weights0["b1"][:], [row[:] for row in weights0["w2"]], weights0["b2"][:]
    lr = run.data["lr"]
    for s in run.data["order"]:
        x, y = run.data["images"][s], run.data["labels"][s]
        pre = [sum(fm(x[i], w1[i][j]) for i in range(len(x))) + b1[j] for j in range(len(b1))]
        act = [max(v, 0) for v in pre]
        logits = [sum(fm(act[i], w2[i][j]) for i in range(len(act))) + b2[j] for j in range(len(b2))]
        err = [v - ((1 << 31) if j == y else 0) for j, v in enumerate(logits)]
        back = [sum(fm(w2[i][j], err[j]) for j in range(len(err))) for i in range(len(act))]
        delta = [fm(back[i], (1 << 31) if pre[i] > 0 else 0) for i in range(len(act))]
        for i in range(len(act)):
            for j in range(len(err)):
                w2[i][j] -= fm(fm(act[i], err[j]), lr)
        b2 = [b - fm(e, lr) for b, e in zip(b2, err)]
        for i in range(len(x)):
            for j in range(len(delta)):
                w1[i][j] -= fm(fm(x[i], delta[j]), lr)
        b1 = [b - fm(dl, lr) for b, dl in zip(b1, delta)]
    assert run.weights(final) == {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def test_ledger_accounting():
    from reverie.ledger import code_length_bits, flatten, landauer_joules, ledger

    assert [code_length_bits(v) for v in (0, -1, 1, -2, 2)] == [1, 3, 3, 5, 5]
    assert list(flatten({"a": [1, [2, True]], "s": reverie.Stack([4])})) == [1, 2, 1, 4]
    book = ledger({"w": [0, 0, 5], "v": 1}, ["w", "v"])
    assert book.values == 4 and book.raw_bits == 256
    assert book.entries[0].nonzero == 1
    assert book.coded_bits == 1 + 1 + code_length_bits(5) + code_length_bits(1)
    assert landauer_joules(1, 300) == pytest.approx(2.87e-21, rel=1e-2)


def test_proof_with_inexactly_parsed_floats_verifies():
    """Regression: `verify` compared the replayed ML profile with the parsed
    one by raw equality, and serde_json's default float parser can land one
    ulp off (here on witness_to_state_payload_ratio = 1.6744927793036237)."""
    import random

    from reverie import lab

    rng = random.Random(0)
    n, d, h, c = 600, 64, 32, 10
    images = [[lab.to_q31(rng.uniform(0, 1)) for _ in range(d)] for _ in range(n)]
    order = list(range(n)) * 3
    run = lab.SGDRewind(
        images=images,
        labels=[i % c for i in range(n)],
        order=order,
        hidden=h,
        classes=c,
        lr=lab.to_q31(0.03),
    )
    seeds = dict(run.data)
    seeds.update(
        w1=[[lab.to_q31(rng.gauss(0, 0.2)) for _ in range(h)] for _ in range(d)],
        b1=[0] * h,
        w2=[[lab.to_q31(rng.gauss(0, 0.2)) for _ in range(c)] for _ in range(h)],
        b2=[0] * c,
        active=[1] * len(order),
        start=0,
        stop=len(order) - 1,
    )
    proof = run.program.roundtrip(seeds)
    report = reverie.verify(proof, source=run.program)
    assert report["passed"], [k for k, ok in report["checks"].items() if not ok]

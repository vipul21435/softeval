import json
from pathlib import Path

import pytest

from softeval import cli

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy"
HUMAN = str(TOY / "human.jsonl")
MODEL_A = str(TOY / "model_a.jsonl")
MODEL_B = str(TOY / "model_b.jsonl")
LABELS = "entailment,neutral,contradiction"


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_score_markdown(capsys):
    code, out, _ = run(capsys, "score", HUMAN, MODEL_A, "--labels", LABELS)
    assert code == 0
    assert "Items scored: 7 (no annotations: 1; tied majority: 2, scored as 'split')" in out
    assert "| total variation distance |" in out
    assert "n/a" not in out


def test_score_json_matches_hand_values(capsys):
    code, out, _ = run(capsys, "score", HUMAN, MODEL_B, "--labels", LABELS, "--format", "json")
    assert code == 0
    payload = json.loads(out)
    assert payload["labels"] == ["entailment", "neutral", "contradiction"]
    assert payload["n_items"] == 7
    # model B answers e, e, n, c, n, e, c; q3 and q6 are two-way ties that it lands inside
    assert payload["metrics"]["majority_accuracy"] == pytest.approx((5 + 0.5 + 0.5) / 7)
    # human mass on those answers: 5/5, 3/5, 2/5, 4/5, 3/5, 2/4, 5/5
    assert payload["metrics"]["annotator_match"] == pytest.approx(
        (1 + 0.6 + 0.4 + 0.8 + 0.6 + 0.5 + 1) / 7
    )


def test_score_tie_policy_changes_accuracy(capsys):
    _, out_any, _ = run(
        capsys, "score", HUMAN, MODEL_B, "--labels", LABELS, "--ties", "any", "--format", "json"
    )
    _, out_excl, _ = run(
        capsys, "score", HUMAN, MODEL_B, "--labels", LABELS, "--ties", "exclude", "--format", "json"
    )
    assert json.loads(out_any)["metrics"]["majority_accuracy"] == pytest.approx(1.0)
    assert json.loads(out_excl)["metrics"]["majority_accuracy"] == pytest.approx(1.0)


def test_hard_predictions_note(capsys, tmp_path):
    preds = tmp_path / "hard.csv"
    preds.write_text("id,label\nq1,entailment\nq2,neutral\n", encoding="utf-8")
    code, out, _ = run(capsys, "score", HUMAN, str(preds), "--labels", LABELS, "--allow-missing")
    assert code == 0
    assert "Items scored: 2" in out
    assert "labels without probabilities" in out
    assert "| cross-entropy (nats) | n/a |" in out


def test_input_errors_exit_with_2(capsys, tmp_path):
    preds = tmp_path / "hard.csv"
    preds.write_text("id,label\nq1,entailment\n", encoding="utf-8")
    code, _, err = run(capsys, "score", HUMAN, str(preds), "--labels", LABELS)
    assert code == 2
    assert "have no prediction" in err
    code, _, err = run(capsys, "score", str(tmp_path / "missing.jsonl"), MODEL_A)
    assert code == 2
    assert "softeval:" in err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0
    assert "softeval 0.1.0" in capsys.readouterr().out

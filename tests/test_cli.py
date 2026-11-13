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


def test_score_reports_intervals_and_corrections(capsys):
    code, out, _ = run(capsys, "score", HUMAN, MODEL_A, "--labels", LABELS, "--bootstrap", "200")
    assert code == 0
    assert "Annotations per item: min 4, median 5, max 5" in out
    assert "| metric | value | 95% interval |" in out
    assert "| metric | plug-in | jackknife | split-half | Miller-Madow | estimated bias |" in out
    code, out, _ = run(
        capsys, "score", HUMAN, MODEL_A, "--labels", LABELS, "--format", "json", "--level", "0.9"
    )
    payload = json.loads(out)
    assert payload["level"] == 0.9
    low = payload["intervals"]["total_variation"]["low"]
    high = payload["intervals"]["total_variation"]["high"]
    assert low <= payload["metrics"]["total_variation"] <= high
    row = payload["finite_annotation"]["total_variation"]
    assert row["estimated_bias"] == pytest.approx(row["plug_in"] - row["jackknife"])
    assert payload["annotations_per_item"] == {"min": 4.0, "median": 5.0, "max": 5.0}


def test_score_without_bootstrap_and_with_fewer_annotators(capsys):
    code, out, _ = run(
        capsys,
        "score",
        HUMAN,
        MODEL_A,
        "--labels",
        LABELS,
        "--bootstrap",
        "0",
        "--annotations",
        "3",
    )
    assert code == 0
    assert "| metric | value |" in out and "interval" not in out
    assert "Annotations per item: min 3, median 3, max 3" in out
    code, _, err = run(capsys, "score", HUMAN, MODEL_A, "--labels", LABELS, "--annotations", "0")
    assert code == 2 and "at least 1" in err


def test_score_with_weighted_counts_skips_corrections(capsys, tmp_path):
    human = tmp_path / "h.csv"
    human.write_text("id,yes,no\na,1.5,0.5\nb,0.5,2.5\n", encoding="utf-8")
    preds = tmp_path / "p.csv"
    preds.write_text("id,yes,no\na,0.7,0.3\nb,0.2,0.8\n", encoding="utf-8")
    code, out, _ = run(capsys, "score", str(human), str(preds), "--labels", "yes,no")
    assert code == 0
    assert "jackknife" not in out


def test_ceiling_markdown_and_json(capsys):
    code, out, _ = run(capsys, "ceiling", HUMAN, "--labels", LABELS, "--draws", "20")
    assert code == 0
    assert "Items: 7 annotated, 7 with two or more annotations" in out
    assert "| metric | one annotator vs the others |" in out
    code, out, _ = run(
        capsys, "ceiling", HUMAN, "--labels", LABELS, "--draws", "20", "--fresh", "50",
        "--format", "json",
    )  # fmt: skip
    payload = json.loads(out)
    assert payload["fresh_annotators"] == 50
    one = payload["one_annotator_vs_others"]
    # pairwise agreement by hand: (1 + 0.4 + 0.2 + 0.6 + 0.3 + 1/3 + 1) / 7
    assert one["annotator_match"] == pytest.approx((1 + 0.4 + 0.2 + 0.6 + 0.3 + 1 / 3 + 1) / 7)
    assert one["cross_entropy"] is None
    fresh = payload["full_distribution_vs_fresh_sample"]
    assert 0 <= fresh["total_variation"] < one["total_variation"]


def test_ceiling_rejects_fractional_counts(capsys, tmp_path):
    human = tmp_path / "h.csv"
    human.write_text("id,yes,no\na,1.5,0.5\n", encoding="utf-8")
    code, _, err = run(capsys, "ceiling", str(human), "--labels", "yes,no")
    assert code == 2 and "whole annotation counts" in err


def test_compare_markdown_and_json(capsys):
    code, out, _ = run(
        capsys, "compare", HUMAN, MODEL_A, MODEL_B, "--labels", LABELS, "--bootstrap", "200"
    )
    assert code == 0
    assert "A = " in out and "B = " in out
    assert "| metric | A | B | A - B | 95% interval | share of resamples A better |" in out
    code, out, _ = run(
        capsys, "compare", HUMAN, MODEL_A, MODEL_B, "--labels", LABELS, "--bootstrap", "200",
        "--format", "json",
    )  # fmt: skip
    payload = json.loads(out)
    tv = payload["metrics"]["total_variation"]
    assert tv["diff"] == pytest.approx(tv["a"] - tv["b"])
    assert tv["diff"] < 0 and tv["p_a_better"] > 0.9
    assert payload["metrics"]["majority_accuracy"]["diff"] == pytest.approx(0.0)


def test_compare_needs_matching_items(capsys, tmp_path):
    short = tmp_path / "short.jsonl"
    short.write_text('{"id": "q1", "label": "entailment"}\n', encoding="utf-8")
    code, _, err = run(capsys, "compare", HUMAN, MODEL_A, str(short), "--labels", LABELS)
    assert code == 2 and "have no prediction" in err
    code, out, _ = run(
        capsys, "compare", HUMAN, MODEL_A, str(short), "--labels", LABELS, "--allow-missing",
        "--bootstrap", "20",
    )  # fmt: skip
    assert code == 0 and "Items scored: 1" in out
    other = tmp_path / "other.jsonl"
    other.write_text('{"id": "zz", "label": "entailment"}\n', encoding="utf-8")
    code, _, err = run(capsys, "score", HUMAN, str(other), "--labels", LABELS, "--allow-missing")
    assert code == 2 and "no item ids are shared" in err


def test_bands_markdown_json_and_bad_edges(capsys):
    code, out, _ = run(capsys, "bands", HUMAN, MODEL_A, "--labels", LABELS)
    assert code == 0
    assert "Bands by share of annotators on the most common label" in out
    assert "| [0.9, 1] | 2 |" in out and "| all | 7 |" in out
    code, out, _ = run(
        capsys, "bands", HUMAN, MODEL_A, "--labels", LABELS, "--by", "entropy", "--edges",
        "0,0.5,1", "--format", "json",
    )  # fmt: skip
    payload = json.loads(out)
    # normalised entropy is under 0.5 for the two unanimous items and for the 4-to-1 item
    assert [band["n_items"] for band in payload["bands"]] == [3, 4]
    assert payload["all"]["n_items"] == 7
    code, _, err = run(capsys, "bands", HUMAN, MODEL_A, "--labels", LABELS, "--edges", "a,b")
    assert code == 2 and "comma-separated numbers" in err
    code, _, err = run(capsys, "bands", HUMAN, MODEL_A, "--labels", LABELS, "--edges", "1,0")
    assert code == 2 and "increasing" in err


def test_simulate_markdown_and_json(capsys):
    argv = [
        "simulate",
        "--items",
        "300",
        "--replicates",
        "3",
        "--annotations",
        "3,20",
        "--seed",
        "4",
    ]
    code, out, _ = run(capsys, *argv)
    assert code == 0
    assert "| metric | estimator | truth | bias at 3 | bias at 20 | within 5% from |" in out
    assert "| cross-entropy (nats) | plug-in |" in out
    code, out, _ = run(capsys, *argv, "--format", "json", "--tolerance", "0.1")
    payload = json.loads(out)
    assert payload["tolerance"] == 0.1
    assert payload["trusted_from"]["cross_entropy/plug-in"] == 3
    assert payload["settings"]["annotations"] == [3, 20]
    code, _, err = run(capsys, "simulate", "--annotations", "three")
    assert code == 2 and "whole numbers" in err
    code, _, err = run(capsys, "simulate", "--annotations", "1,5")
    assert code == 2 and "at least 2" in err


def test_quickstart_example_runs(capsys):
    import importlib.util

    path = TOY.parent / "quickstart.py"
    spec = importlib.util.spec_from_file_location("softeval_quickstart", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    out = capsys.readouterr().out
    assert "items scored: 7, without annotations: 1" in out
    assert "model A total_variation: 0.0786" in out
    assert "one annotator vs the others, match: 0.5476" in out


def test_every_command_survives_a_file_without_any_annotation(capsys, tmp_path):
    human = tmp_path / "empty.jsonl"
    human.write_text('{"id": "a", "annotations": []}\n{"id": "b", "annotations": []}\n')
    preds = tmp_path / "p.jsonl"
    preds.write_text('{"id": "a", "probs": [0.5, 0.5]}\n{"id": "b", "label": "y"}\n')
    h, p = str(human), str(preds)
    code, out, _ = run(capsys, "score", h, p, "--labels", "y,n", "--bootstrap", "5")
    assert code == 0 and "Items scored: 0 (no annotations: 2;" in out
    assert "| total variation distance | n/a | n/a |" in out
    code, out, _ = run(capsys, "ceiling", h, "--labels", "y,n", "--draws", "2")
    assert code == 0 and "Items: 0 annotated" in out
    code, out, _ = run(capsys, "bands", h, p, "--labels", "y,n")
    assert code == 0 and "| all | 0 |" in out
    code, out, _ = run(capsys, "compare", h, p, p, "--labels", "y,n", "--bootstrap", "5")
    assert code == 0 and "Items scored: 0" in out
    code, out, _ = run(capsys, "score", h, p, "--labels", "y,n", "--annotations", "3")
    assert code == 0

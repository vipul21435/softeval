"""The ChaosNLI example scripts, run on a tiny made-up folder in the same file formats."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from softeval import io

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "chaosnli"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"chaosnli_{name}", EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def data_dir(tmp_path):
    rows = [
        {
            "uid": "a1",
            "label_counter": {"e": 70, "n": 30},
            "old_label": "e",
            "old_labels": ["entailment", "entailment", "neutral", "entailment", "entailment"],
        },
        {
            "uid": "a2",
            "label_counter": {"n": 50, "c": 45, "e": 5},
            "old_label": "c",
            "old_labels": ["contradiction", "contradiction", "neutral", "neutral", "contradiction"],
        },
    ]
    with (tmp_path / "chaosNLI_snli.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    baseline = {
        "random_baseline": {
            "a1": {"uid": "a1", "predicted_probabilities": [0.2, 0.3, 0.5]},
            "a2": {"uid": "a2", "predicted_probabilities": [0.6, 0.2, 0.2]},
        }
    }
    (tmp_path / "snli_random_baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    models = {
        "model one": {
            "a1": {"uid": "a1", "logits": [2.0, 0.0, 0.0]},
            "a2": {"uid": "a2", "logits": [0.0, 0.0, 0.0]},
        }
    }
    (tmp_path / "model_predictions_for_snli_mnli.json").write_text(
        json.dumps(models), encoding="utf-8"
    )
    return tmp_path


def test_prepare_writes_files_softeval_reads(data_dir, tmp_path):
    prepare = _load("prepare")
    out = tmp_path / "prepared"
    written = prepare.prepare_subset("snli", data_dir, out)
    assert written["items"] == 2 and written["model-one"] == 2
    human = io.read_human(out / "snli" / "human.jsonl", labels="e,n,c")
    assert human.counts.tolist() == [[70, 30, 0], [5, 50, 45]]
    old = io.read_human(out / "snli" / "old_label.jsonl", labels="e,n,c")
    assert old.counts.tolist() == [[1, 0, 0], [0, 0, 1]]
    labels = human.labels
    hard = io.read_predictions(out / "snli" / "pred_old-label.jsonl", labels)
    assert hard.probs.tolist() == [[1, 0, 0], [0, 0, 1]]
    five = io.read_predictions(out / "snli" / "pred_old-five.jsonl", labels)
    assert five.probs[0] == pytest.approx(np.array([4.5, 1.5, 0.5]) / 6.5)
    prior = io.read_predictions(out / "snli" / "pred_old-prior.jsonl", labels)
    assert prior.probs == pytest.approx(np.array([[0.5, 0, 0.5], [0.5, 0, 0.5]]))
    baseline = io.read_predictions(out / "snli" / "pred_random-baseline.jsonl", labels)
    assert baseline.probs[1] == pytest.approx([0.6, 0.2, 0.2])
    model = io.read_predictions(out / "snli" / "pred_model-one.jsonl", labels)
    e = np.exp(2.0)
    assert model.probs[0] == pytest.approx([e / (e + 2), 1 / (e + 2), 1 / (e + 2)])
    assert model.probs[1] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_prepare_main_skips_missing_subsets(data_dir, tmp_path, capsys):
    prepare = _load("prepare")
    assert prepare.main(["prepare.py", str(data_dir), str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "snli: items 2" in out
    assert "mnli_m: chaosNLI_mnli_m.jsonl not found" in out
    assert prepare.main(["prepare.py"]) == 2


def test_tables_run_on_the_prepared_folder(data_dir, tmp_path, capsys, monkeypatch):
    prepare = _load("prepare")
    tables = _load("tables")
    out = tmp_path / "prepared"
    prepare.prepare_subset("snli", data_dir, out)
    monkeypatch.setattr(sys, "argv", ["tables.py", str(out), "--draws", "2"])
    assert tables.main() == 0
    text = capsys.readouterr().out
    assert "### snli: 2 items" in text
    assert "| old-label | 1.0000 | 0.5000 |" in text
    assert "one annotator vs the other 99" in text
    assert tables.ranks([0.9, 0.5, 0.9], higher_is_better=True) == [1, 3, 1]
    assert tables.ranks([0.2, 0.1, 0.3], higher_is_better=False) == [2, 1, 3]

"""Turn the ChaosNLI files into softeval inputs.

    python examples/chaosnli/prepare.py DATA_DIR OUT_DIR

For each subset (snli, mnli_m, alphanli) this writes into OUT_DIR/<subset>/:

- human.jsonl          the 100 new annotations per item, as label counts;
- old_label.jsonl      the label the item had in the original dataset, as a single annotation;
- pred_*.jsonl         predictions to score (see PREDICTORS below).

If saved model predictions in the authors' format are present in DATA_DIR
(model_predictions_for_snli_mnli.json, model_predictions_for_abdnli.json), each model in them is
written as pred_<name>.jsonl too, with logits turned into probabilities by a softmax.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from softeval.metrics import softmax

SUBSETS = {
    "snli": ("chaosNLI_snli.jsonl", ["e", "n", "c"], "snli_random_baseline.json"),
    "mnli_m": ("chaosNLI_mnli_m.jsonl", ["e", "n", "c"], "mnli_random_baseline.json"),
    "alphanli": ("chaosNLI_alphanli.jsonl", ["1", "2"], "abdnli_random_baseline.json"),
}
MODEL_FILES = {
    "snli": "model_predictions_for_snli_mnli.json",
    "mnli_m": "model_predictions_for_snli_mnli.json",
    "alphanli": "model_predictions_for_abdnli.json",
}
LONG_NAMES = {"entailment": "e", "neutral": "n", "contradiction": "c"}

PREDICTORS = {
    "old-label": "all probability on the label the item had in the original dataset",
    "old-five": "distribution of the five original annotations, with 0.5 added to each count",
    "old-prior": "one distribution for every item: the pooled original labels of the subset",
    "random-baseline": "the random-baseline prediction file in the ChaosNLI repository",
    "uniform": "equal probability on every label",
}


def read_items(path: Path, labels: list[str]) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counter = {str(k): int(v) for k, v in row["label_counter"].items()}
            old_five = [LONG_NAMES.get(str(x), str(x)) for x in row.get("old_labels") or []]
            items.append(
                {
                    "id": str(row["uid"]),
                    "counts": {label: counter.get(label, 0) for label in labels},
                    "old_label": str(row["old_label"]),
                    "old_five": [x for x in old_five if x in labels],
                }
            )
    return items


def probability_rows(entries: dict, labels: list[str]) -> dict[str, list[float]]:
    """Per-item probabilities from one model's entries in the authors' prediction format."""
    out = {}
    for uid, entry in entries.items():
        if "predicted_probabilities" in entry:
            probs = np.asarray(entry["predicted_probabilities"], dtype=float)
            probs = probs / probs.sum()
        else:
            probs = softmax(np.asarray([entry["logits"]], dtype=float))[0]
        if probs.size != len(labels):
            raise ValueError(f"{uid}: {probs.size} values for {len(labels)} labels")
        out[str(uid)] = [float(x) for x in probs]
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_predictions(path: Path, items: list[dict], labels: list[str], rows: dict) -> int:
    kept = [
        {"id": item["id"], "probs": dict(zip(labels, rows[item["id"]], strict=True))}
        for item in items
        if item["id"] in rows
    ]
    write_rows(path, kept)
    return len(kept)


def prepare_subset(name: str, data_dir: Path, out_dir: Path) -> dict[str, int]:
    file_name, labels, baseline_name = SUBSETS[name]
    items = read_items(data_dir / file_name, labels)
    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    write_rows(target / "human.jsonl", [{"id": i["id"], "counts": i["counts"]} for i in items])
    write_rows(
        target / "old_label.jsonl",
        [{"id": i["id"], "counts": {i["old_label"]: 1}} for i in items],
    )
    written = {"items": len(items)}
    k = len(labels)
    one_hot = {i["id"]: [float(label == i["old_label"]) for label in labels] for i in items}
    written["old-label"] = write_predictions(
        target / "pred_old-label.jsonl", items, labels, one_hot
    )
    if all(len(i["old_five"]) > 0 for i in items):
        five = {}
        for i in items:
            counts = np.array([i["old_five"].count(label) for label in labels], dtype=float) + 0.5
            five[i["id"]] = [float(x) for x in counts / counts.sum()]
        written["old-five"] = write_predictions(target / "pred_old-five.jsonl", items, labels, five)
    pooled = np.array([sum(i["old_label"] == label for i in items) for label in labels], float)
    prior = [float(x) for x in pooled / pooled.sum()]
    written["old-prior"] = write_predictions(
        target / "pred_old-prior.jsonl", items, labels, {i["id"]: prior for i in items}
    )
    written["uniform"] = write_predictions(
        target / "pred_uniform.jsonl", items, labels, {i["id"]: [1.0 / k] * k for i in items}
    )
    baseline_path = data_dir / baseline_name
    if baseline_path.exists():
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        rows = probability_rows(payload["random_baseline"], labels)
        written["random-baseline"] = write_predictions(
            target / "pred_random-baseline.jsonl", items, labels, rows
        )
    model_path = data_dir / MODEL_FILES[name]
    if model_path.exists():
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        for model, entries in payload.items():
            rows = probability_rows(entries, labels)
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in str(model))
            written[safe] = write_predictions(target / f"pred_{safe}.jsonl", items, labels, rows)
    return written


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    data_dir, out_dir = Path(argv[1]), Path(argv[2])
    for name in SUBSETS:
        if not (data_dir / SUBSETS[name][0]).exists():
            print(f"{name}: {SUBSETS[name][0]} not found in {data_dir}, skipped")
            continue
        written = prepare_subset(name, data_dir, out_dir)
        print(f"{name}: " + ", ".join(f"{key} {value}" for key, value in written.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

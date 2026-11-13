"""Readers for human annotations and model predictions in JSONL or CSV.

Human labels, one of:

- counts per item: ``{"id": "a", "counts": {"yes": 3, "no": 2}}`` (or a list in label order), or a
  CSV with an ``id`` column and one numeric column per label;
- a list of annotations per item: ``{"id": "a", "annotations": ["yes", "no", "yes"]}``;
- one row per annotation: ``{"id": "a", "label": "yes"}`` or a CSV with ``id`` and ``label``
  columns (an ``annotator`` column is allowed and ignored).

Predictions, one of:

- a label: ``{"id": "a", "label": "yes"}`` or CSV ``id,label``;
- probabilities: ``{"id": "a", "probs": {"yes": 0.7, "no": 0.3}}`` (or a list in label order), or
  a CSV with ``id`` and one column per label.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from softeval.metrics import FloatArray

ID_KEYS = ("id", "uid", "item", "item_id")
IGNORED_COLUMNS = ("annotator", "annotator_id", "worker", "worker_id")


class InputError(ValueError):
    """Raised for a file that cannot be read as annotations or predictions."""


@dataclass
class HumanLabels:
    """Label counts: ``counts[i, k]`` annotators gave item ``ids[i]`` the label ``labels[k]``."""

    ids: list[str]
    labels: list[str]
    counts: FloatArray

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class Predictions:
    """Predicted probabilities per item, in the label order of the human file."""

    ids: list[str]
    labels: list[str]
    probs: FloatArray
    hard: bool

    def __len__(self) -> int:
        return len(self.ids)


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".jsonl", ".ndjson", ".json"):
        rows: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise InputError(f"{p}:{line_no}: not valid JSON ({exc.msg})") from exc
                if not isinstance(row, dict):
                    raise InputError(f"{p}:{line_no}: each line must be a JSON object")
                rows.append(row)
        return rows
    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with p.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [dict(row) for row in reader]
    raise InputError(f"{p}: unknown extension {suffix!r}; use .jsonl, .csv or .tsv")


def _item_id(row: dict[str, Any], where: str) -> str:
    for key in ID_KEYS:
        if key in row and row[key] is not None and str(row[key]) != "":
            return str(row[key])
    raise InputError(f"{where}: no id column (looked for {', '.join(ID_KEYS)})")


def _to_float(value: Any, where: str) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        raise InputError(f"{where}: expected a number, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{where}: expected a number, got {value!r}") from exc
    if not np.isfinite(out):
        raise InputError(f"{where}: expected a finite number, got {value!r}")
    return out


def _vector_columns(row: dict[str, Any]) -> list[str]:
    skip = set(ID_KEYS) | set(IGNORED_COLUMNS)
    return [key for key in row if key not in skip]


def _parse_label_list(labels: Sequence[str] | str | None) -> list[str] | None:
    if labels is None:
        return None
    if isinstance(labels, str):
        labels = [part.strip() for part in labels.split(",")]
    out = [str(label) for label in labels if str(label) != ""]
    if len(set(out)) != len(out):
        raise InputError("label names must be distinct")
    if len(out) < 2:
        raise InputError("at least two labels are needed")
    return out


def read_human(path: str | Path, labels: Sequence[str] | str | None = None) -> HumanLabels:
    """Read human annotations into per-item label counts.

    ``labels`` fixes the label set and its order. Without it the labels are those seen in the
    file, sorted. Items listed with no annotations are kept with a row of zeros.
    """
    rows = _read_rows(path)
    if not rows:
        raise InputError(f"{path}: no rows")
    fixed = _parse_label_list(labels)
    per_item: dict[str, dict[str, float]] = {}
    positional: dict[str, list[float]] = {}
    for n, row in enumerate(rows, start=1):
        where = f"{path}: row {n}"
        item = _item_id(row, where)
        bucket = per_item.setdefault(item, {})
        if "counts" in row:
            value = row["counts"]
            if isinstance(value, dict):
                for label, count in value.items():
                    bucket[str(label)] = bucket.get(str(label), 0.0) + _to_float(count, where)
            elif isinstance(value, list):
                vec = [_to_float(v, where) for v in value]
                if item in positional and len(positional[item]) != len(vec):
                    raise InputError(f"{where}: counts list changes length for item {item!r}")
                old = positional.get(item, [0.0] * len(vec))
                positional[item] = [a + b for a, b in zip(old, vec, strict=True)]
            else:
                raise InputError(f"{where}: counts must be an object or a list")
        elif "annotations" in row or "labels" in row:
            value = row.get("annotations", row.get("labels"))
            if not isinstance(value, list):
                raise InputError(f"{where}: annotations must be a list of labels")
            for label in value:
                if label is None or label == "":
                    continue
                bucket[str(label)] = bucket.get(str(label), 0.0) + 1.0
        elif "label" in row:
            label = row["label"]
            if label is not None and str(label) != "":
                bucket[str(label)] = bucket.get(str(label), 0.0) + 1.0
        else:
            for label in _vector_columns(row):
                bucket[label] = bucket.get(label, 0.0) + _to_float(row[label], where)
    if positional:
        if any(per_item[item] for item in per_item):
            raise InputError(f"{path}: counts given both as lists and by label name")
        width = {len(v) for v in positional.values()}
        if len(width) != 1:
            raise InputError(f"{path}: counts lists have different lengths")
        k = width.pop()
        names = fixed if fixed is not None else [str(i) for i in range(k)]
        if len(names) != k:
            raise InputError(f"{path}: counts lists have {k} entries but {len(names)} labels given")
        ids = list(per_item)
        counts = np.array([positional.get(i, [0.0] * k) for i in ids], dtype=np.float64)
    else:
        seen = sorted({label for bucket in per_item.values() for label in bucket})
        names = fixed if fixed is not None else seen
        unknown = [label for label in seen if label not in names]
        if unknown:
            raise InputError(f"{path}: labels not in the given label set: {', '.join(unknown)}")
        ids = list(per_item)
        counts = np.array(
            [[per_item[i].get(label, 0.0) for label in names] for i in ids], dtype=np.float64
        )
    if len(names) < 2:
        raise InputError(f"{path}: found fewer than two labels; pass the label set explicitly")
    if np.any(counts < 0):
        raise InputError(f"{path}: negative count")
    return HumanLabels(ids=ids, labels=list(names), counts=counts)


def read_predictions(path: str | Path, labels: Sequence[str]) -> Predictions:
    """Read predictions as probability rows over ``labels`` (the order of the human file)."""
    rows = _read_rows(path)
    if not rows:
        raise InputError(f"{path}: no rows")
    names = list(labels)
    index = {label: k for k, label in enumerate(names)}
    ids: list[str] = []
    out: list[list[float]] = []
    seen: set[str] = set()
    all_hard = True
    for n, row in enumerate(rows, start=1):
        where = f"{path}: row {n}"
        item = _item_id(row, where)
        if item in seen:
            raise InputError(f"{where}: item {item!r} appears twice")
        seen.add(item)
        vec = [0.0] * len(names)
        if "probs" in row or "probabilities" in row:
            value = row.get("probs", row.get("probabilities"))
            all_hard = False
            if isinstance(value, dict):
                for label, prob in value.items():
                    if str(label) not in index:
                        raise InputError(f"{where}: unknown label {label!r}")
                    vec[index[str(label)]] = _to_float(prob, where)
            elif isinstance(value, list):
                if len(value) != len(names):
                    raise InputError(f"{where}: {len(value)} probabilities for {len(names)} labels")
                vec = [_to_float(v, where) for v in value]
            else:
                raise InputError(f"{where}: probs must be an object or a list")
        elif "label" in row or "prediction" in row:
            label = str(row.get("label", row.get("prediction")))
            if label not in index:
                raise InputError(f"{where}: unknown label {label!r}")
            vec[index[label]] = 1.0
        else:
            columns = _vector_columns(row)
            unknown = [c for c in columns if c not in index]
            if unknown:
                raise InputError(f"{where}: unknown label column {unknown[0]!r}")
            all_hard = False
            for column in columns:
                vec[index[column]] = _to_float(row[column], where)
        total = sum(vec)
        if any(v < 0 for v in vec) or abs(total - 1.0) > 1e-4:
            raise InputError(f"{where}: probabilities must be non-negative and sum to 1")
        ids.append(item)
        out.append([v / total for v in vec])
    probs = np.array(out, dtype=np.float64)
    return Predictions(ids=ids, labels=names, probs=probs, hard=all_hard)


def align(
    human: HumanLabels, predictions: Predictions, *, allow_missing: bool = False
) -> tuple[FloatArray, FloatArray, list[str]]:
    """Return counts and probabilities for the same items in the order of the human file.

    Items in only one of the two files are an error unless ``allow_missing`` is set, in which
    case only the shared items are kept.
    """
    position = {item: i for i, item in enumerate(predictions.ids)}
    shared = [item for item in human.ids if item in position]
    missing = len(human.ids) - len(shared)
    extra = len(predictions.ids) - len(shared)
    if (missing or extra) and not allow_missing:
        raise InputError(
            f"{missing} annotated items have no prediction and {extra} predictions have no "
            "annotations; pass allow_missing to score the shared items only"
        )
    if not shared:
        raise InputError("no item ids are shared between the two files")
    human_pos = {item: i for i, item in enumerate(human.ids)}
    counts = human.counts[[human_pos[item] for item in shared]]
    probs = predictions.probs[[position[item] for item in shared]]
    return counts, probs, shared


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as JSON lines."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

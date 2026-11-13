import json

import numpy as np
import pytest

from softeval import io


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_counts_as_objects(tmp_path):
    path = _write(
        tmp_path / "h.jsonl",
        '{"id": "a", "counts": {"yes": 3, "no": 1}}\n\n{"id": "b", "counts": {"maybe": 2}}\n',
    )
    human = io.read_human(path)
    assert human.ids == ["a", "b"]
    assert human.labels == ["maybe", "no", "yes"]
    assert human.counts.tolist() == [[0, 1, 3], [2, 0, 0]]
    assert len(human) == 2


def test_counts_as_lists_need_matching_labels(tmp_path):
    path = _write(
        tmp_path / "h.jsonl", '{"id": 1, "counts": [3, 1]}\n{"id": 2, "counts": [0, 0]}\n'
    )
    human = io.read_human(path, labels="yes,no")
    assert human.labels == ["yes", "no"]
    assert human.counts.tolist() == [[3, 1], [0, 0]]
    assert io.read_human(path).labels == ["0", "1"]
    with pytest.raises(io.InputError, match="3 labels given"):
        io.read_human(path, labels=["a", "b", "c"])


def test_annotation_lists_and_long_rows_agree(tmp_path):
    lists = _write(
        tmp_path / "lists.jsonl",
        '{"id": "a", "annotations": ["y", "n", "y"]}\n{"id": "b", "annotations": []}\n',
    )
    long_csv = _write(tmp_path / "long.csv", "id,annotator,label\na,w1,y\na,w2,n\na,w3,y\nb,w1,\n")
    first = io.read_human(lists, labels=["y", "n"])
    second = io.read_human(long_csv, labels=["y", "n"])
    assert first.counts.tolist() == second.counts.tolist() == [[2, 1], [0, 0]]
    assert first.ids == second.ids == ["a", "b"]


def test_wide_csv_counts(tmp_path):
    path = _write(tmp_path / "h.csv", "id,e,n,c\nx,60,30,10\ny,5,,95\n")
    human = io.read_human(path, labels="e,n,c")
    assert human.counts.tolist() == [[60, 30, 10], [5, 0, 95]]


def test_human_errors(tmp_path):
    with pytest.raises(io.InputError, match="no rows"):
        io.read_human(_write(tmp_path / "empty.jsonl", "\n"))
    with pytest.raises(io.InputError, match="not valid JSON"):
        io.read_human(_write(tmp_path / "bad.jsonl", "{oops\n"))
    with pytest.raises(io.InputError, match="JSON object"):
        io.read_human(_write(tmp_path / "list.jsonl", "[1, 2]\n"))
    with pytest.raises(io.InputError, match="no id column"):
        io.read_human(_write(tmp_path / "noid.jsonl", '{"label": "y"}\n'))
    with pytest.raises(io.InputError, match="not in the given label set"):
        io.read_human(_write(tmp_path / "h.jsonl", '{"id": 1, "label": "z"}\n'), labels="y,n")
    with pytest.raises(io.InputError, match="expected a number"):
        io.read_human(_write(tmp_path / "h2.csv", "id,y,n\na,two,1\n"))
    with pytest.raises(io.InputError, match="unknown extension"):
        io.read_human(_write(tmp_path / "h.txt", "x"))
    with pytest.raises(io.InputError, match="fewer than two labels"):
        io.read_human(_write(tmp_path / "one.jsonl", '{"id": 1, "label": "y"}\n'))
    with pytest.raises(io.InputError, match="distinct"):
        io.read_human(_write(tmp_path / "h3.jsonl", '{"id": 1, "label": "y"}\n'), labels="y,y")


def test_predictions_as_labels_and_probabilities(tmp_path):
    labels = ["e", "n", "c"]
    hard = io.read_predictions(_write(tmp_path / "p.csv", "id,label\na,n\nb,e\n"), labels)
    assert hard.hard
    assert hard.probs.tolist() == [[0, 1, 0], [1, 0, 0]]
    soft = io.read_predictions(
        _write(
            tmp_path / "p.jsonl",
            '{"id": "a", "probs": {"e": 0.5, "c": 0.5}}\n{"id": "b", "probs": [0.2, 0.3, 0.5]}\n',
        ),
        labels,
    )
    assert not soft.hard
    assert soft.probs == pytest.approx(np.array([[0.5, 0, 0.5], [0.2, 0.3, 0.5]]))
    wide = io.read_predictions(_write(tmp_path / "w.csv", "id,e,n,c\na,0.1,0.2,0.7\n"), labels)
    assert wide.probs == pytest.approx(np.array([[0.1, 0.2, 0.7]]))
    assert len(wide) == 1


def test_prediction_errors(tmp_path):
    labels = ["e", "n"]
    with pytest.raises(io.InputError, match="sum to 1"):
        io.read_predictions(
            _write(tmp_path / "a.jsonl", '{"id": 1, "probs": [0.5, 0.2]}\n'), labels
        )
    with pytest.raises(io.InputError, match="unknown label"):
        io.read_predictions(_write(tmp_path / "b.jsonl", '{"id": 1, "label": "z"}\n'), labels)
    with pytest.raises(io.InputError, match="appears twice"):
        io.read_predictions(
            _write(tmp_path / "c.jsonl", '{"id": 1, "label": "e"}\n{"id": 1, "label": "n"}\n'),
            labels,
        )
    with pytest.raises(io.InputError, match="3 probabilities"):
        io.read_predictions(_write(tmp_path / "d.jsonl", '{"id": 1, "probs": [1, 0, 0]}\n'), labels)
    with pytest.raises(io.InputError, match="unknown label column"):
        io.read_predictions(_write(tmp_path / "e.csv", "id,e,q\na,0.5,0.5\n"), labels)


def test_align_orders_by_human_file_and_reports_missing(tmp_path):
    human = io.HumanLabels(["a", "b", "c"], ["y", "n"], np.array([[1, 0], [0, 1], [1, 1.0]]))
    preds = io.Predictions(
        ["c", "a", "z"], ["y", "n"], np.array([[0.5, 0.5], [1, 0], [0, 1.0]]), False
    )
    with pytest.raises(io.InputError, match="1 annotated items have no prediction"):
        io.align(human, preds)
    counts, probs, ids = io.align(human, preds, allow_missing=True)
    assert ids == ["a", "c"]
    assert counts.tolist() == [[1, 0], [1, 1]]
    assert probs.tolist() == [[1, 0], [0.5, 0.5]]
    other = io.Predictions(["q"], ["y", "n"], np.array([[1.0, 0.0]]), True)
    with pytest.raises(io.InputError, match="no item ids are shared"):
        io.align(human, other, allow_missing=True)


def test_write_jsonl_round_trip(tmp_path):
    path = tmp_path / "out.jsonl"
    io.write_jsonl(path, [{"id": "a", "counts": {"y": 2, "n": 1}}])
    assert json.loads(path.read_text()) == {"counts": {"n": 1, "y": 2}, "id": "a"}
    assert io.read_human(path).counts.tolist() == [[1, 2]]


def test_more_human_errors(tmp_path):
    def bad(name, text, match, **kwargs):
        with pytest.raises(io.InputError, match=match):
            io.read_human(_write(tmp_path / name, text), **kwargs)

    bad("a.jsonl", '{"id": 1, "counts": {"y": true, "n": 1}}\n', "expected a number")
    bad("b.jsonl", '{"id": 1, "counts": {"y": "inf", "n": 1}}\n', "finite number")
    bad("c.jsonl", '{"id": 1, "label": "y"}\n', "at least two labels", labels="y")
    bad(
        "d.jsonl", '{"id": 1, "counts": [1, 2]}\n{"id": 1, "counts": [1, 2, 3]}\n', "changes length"
    )
    bad("e.jsonl", '{"id": 1, "counts": 3}\n', "object or a list")
    bad("f.jsonl", '{"id": 1, "annotations": "y"}\n', "list of labels")
    bad("g.jsonl", '{"id": 1, "counts": [1, 2]}\n{"id": 2, "label": "y"}\n', "both as lists")
    bad("h.jsonl", '{"id": 1, "counts": [1, 2]}\n{"id": 2, "counts": [1, 2, 3]}\n', "different")
    bad("i.jsonl", '{"id": 1, "counts": {"y": -1, "n": 2}}\n', "negative count")


def test_repeated_rows_add_up_and_blank_annotations_are_skipped(tmp_path):
    path = _write(
        tmp_path / "h.jsonl",
        '{"id": "a", "counts": [1, 2]}\n{"id": "a", "counts": [2, 0]}\n',
    )
    assert io.read_human(path).counts.tolist() == [[3, 2]]
    path = _write(tmp_path / "l.jsonl", '{"id": "a", "annotations": ["y", null, "", "n", "y"]}\n')
    assert io.read_human(path).counts.tolist() == [[1, 2]]


def test_more_prediction_errors(tmp_path):
    labels = ["e", "n"]
    with pytest.raises(io.InputError, match="no rows"):
        io.read_predictions(_write(tmp_path / "a.csv", "id,label\n"), labels)
    with pytest.raises(io.InputError, match="unknown label"):
        io.read_predictions(_write(tmp_path / "b.jsonl", '{"id": 1, "probs": {"z": 1}}\n'), labels)
    with pytest.raises(io.InputError, match="object or a list"):
        io.read_predictions(_write(tmp_path / "c.jsonl", '{"id": 1, "probs": 0.5}\n'), labels)

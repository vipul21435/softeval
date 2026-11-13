import json
import math

import numpy as np

from softeval import report


def test_fmt_handles_every_cell_type():
    assert report.fmt(None) == "n/a"
    assert report.fmt(float("nan")) == "n/a"
    assert report.fmt(float("inf")) == "inf"
    assert report.fmt(float("-inf")) == "-inf"
    assert report.fmt(True) == "yes" and report.fmt(False) == "no"
    assert report.fmt(7) == "7"
    assert report.fmt(0.123456) == "0.1235"
    assert report.fmt(0.5, digits=2) == "0.50"
    assert report.fmt("text") == "text"


def test_interval_and_table():
    assert report.interval(0.1, 0.25) == "[0.1000, 0.2500]"
    assert report.interval(None, 0.2) == "n/a"
    assert report.interval(float("nan"), float("nan")) == "n/a"
    table = report.markdown_table(["a", "b"], [[1, 0.5], ["x", None]])
    assert table.splitlines() == ["| a | b |", "| --- | --- |", "| 1 | 0.5000 |", "| x | n/a |"]


def test_to_json_replaces_non_finite_numbers_and_numpy_values():
    payload = {
        "nan": float("nan"),
        "inf": math.inf,
        "array": np.array([1.0, np.nan]),
        "scalar": np.float64(2.5),
        "nested": [{"x": (1, 2)}],
    }
    assert json.loads(report.to_json(payload)) == {
        "nan": None,
        "inf": None,
        "array": [1.0, None],
        "scalar": 2.5,
        "nested": [{"x": [1, 2]}],
    }

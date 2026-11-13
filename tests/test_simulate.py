import pytest

from softeval import simulate as S


@pytest.fixture(scope="module")
def result():
    return S.simulate(n_items=1500, replicates=6, seed=11)


def _row(result, metric, estimator, n):
    for row in result.rows:
        if (row.metric, row.estimator, row.annotations) == (metric, estimator, n):
            return row
    raise AssertionError((metric, estimator, n))


def test_linear_metrics_are_unbiased_at_every_number_of_annotations(result):
    for n in (3, 5, 10, 100):
        for metric in ("cross_entropy", "annotator_match"):
            row = _row(result, metric, "plug-in", n)
            # stated tolerance: 2 % of the true value
            assert abs(row.bias) < 0.02 * abs(row.truth)


def test_plugin_divergences_overstate_and_the_bias_falls_with_annotations(result):
    for metric in ("kl", "js_divergence", "js_distance", "total_variation"):
        biases = [_row(result, metric, "plug-in", n).bias for n in (3, 5, 10, 100)]
        assert all(b > 0 for b in biases)
        assert biases == sorted(biases, reverse=True)
        # at 5 annotations the plug-in value is more than 50 % above the truth
        assert _row(result, metric, "plug-in", 5).relative_bias > 0.5


def test_corrections_beat_the_plugin_value(result):
    for metric in ("kl", "js_divergence", "js_distance", "total_variation"):
        for n in (5, 10, 100):
            plug = abs(_row(result, metric, "plug-in", n).bias)
            assert abs(_row(result, metric, "jackknife", n).bias) < 0.5 * plug
            assert abs(_row(result, metric, "split-half", n).bias) < 0.6 * plug
    # at 100 annotations the jackknife is within 2 % of the truth for the smooth divergences
    for metric in ("kl", "js_divergence"):
        assert abs(_row(result, metric, "jackknife", 100).relative_bias) < 0.02


def test_majority_accuracy_and_entropy_correlation_need_many_annotations(result):
    assert abs(_row(result, "majority_accuracy", "plug-in", 100).relative_bias) < 0.03
    assert _row(result, "majority_accuracy", "plug-in", 3).bias < -0.03
    low = _row(result, "entropy_spearman", "plug-in", 3)
    high = _row(result, "entropy_spearman", "plug-in", 100)
    assert low.bias < high.bias < 0.01
    assert abs(high.bias) < abs(low.bias)


def test_trusted_from_reads_the_bias_table(result):
    assert result.trusted_from("cross_entropy", "plug-in", 0.05) == 3
    assert result.trusted_from("total_variation", "plug-in", 0.05) is None
    assert result.trusted_from("kl", "jackknife", 0.05) in (10, 100)
    assert result.trusted_from("kl", "plug-in", 0.5) in (10, 100)
    payload = result.to_dict()
    assert payload["settings"]["seed"] == 11
    assert len(payload["rows"]) == len(result.rows)
    assert {"metric", "estimator", "annotations", "truth", "mean", "bias", "rmse"} <= set(
        payload["rows"][0]
    )


def test_same_seed_same_numbers_and_argument_checks():
    a = S.simulate(n_items=200, replicates=2, annotations=(3, 10), seed=3)
    b = S.simulate(n_items=200, replicates=2, annotations=(3, 10), seed=3)
    assert [r.mean for r in a.rows] == pytest.approx([r.mean for r in b.rows])
    with pytest.raises(ValueError, match="replicates"):
        S.simulate(replicates=0)
    with pytest.raises(ValueError, match="at least 2"):
        S.simulate(annotations=(1, 5))
    with pytest.raises(ValueError, match="alpha"):
        S.simulate(alpha=0)
    with pytest.raises(ValueError, match="noise"):
        S.simulate(noise=2)
    with pytest.raises(ValueError, match="two items"):
        S.simulate(n_items=1)

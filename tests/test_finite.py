import math

import numpy as np
import pytest

from softeval import finite
from softeval import metrics as M
from softeval.sampling import sample_annotators


def test_jackknife_by_hand():
    # counts (3, 1) against p = (0.7, 0.3): the plug-in TV is 0.05. Leaving out one of the three
    # label-0 annotators gives (2, 1) and TV 1/30; leaving out the label-1 annotator gives (3, 0)
    # and TV 0.3. Their mean over annotators is 0.1, so the jackknife is 4 * 0.05 - 3 * 0.1.
    out = finite.jackknife_items([[3, 1]], [[0.7, 0.3]])
    assert out["total_variation"][0] == pytest.approx(-0.1)


def test_single_annotation_and_empty_rows():
    counts = [[1, 0], [0, 0], [4, 4]]
    probs = [[0.6, 0.4]] * 3
    plain = M.item_metrics(counts, probs)
    jack = finite.jackknife_items(counts, probs)
    split = finite.split_half_items(counts, probs, n_splits=3, seed=0)
    for name in finite.CORRECTABLE:
        assert jack[name][0] == pytest.approx(plain[name][0])
        assert split[name][0] == pytest.approx(plain[name][0])
        assert math.isnan(jack[name][1]) and math.isnan(split[name][1])
        assert math.isfinite(jack[name][2]) and math.isfinite(split[name][2])
    with pytest.raises(ValueError, match="n_splits"):
        finite.split_half_items(counts, probs, n_splits=0)


def test_split_half_by_hand_when_every_half_is_the_same():
    # two annotators who agree: each half is one annotator with the same label, so the half
    # value equals the full value and the extrapolation (2 * D - 1 * D) / 1 returns it
    out = finite.split_half_items([[2, 0]], [[0.6, 0.4]], n_splits=4, seed=1)
    assert out["total_variation"][0] == pytest.approx(0.4)


def test_miller_madow_kl_by_hand():
    counts = [[2, 2, 0]]
    probs = [[0.5, 0.25, 0.25]]
    plain = M.kl_divergence(M.distributions(counts), probs)[0]
    assert finite.miller_madow_kl(counts, probs)[0] == pytest.approx(plain - 1 / 8)


def test_compositions_and_pmf():
    outcomes = finite.compositions(4, 3)
    assert len(outcomes) == math.comb(6, 2)
    assert np.all(outcomes.sum(axis=1) == 4)
    assert len({tuple(row) for row in outcomes.tolist()}) == len(outcomes)
    pmf = finite.multinomial_pmf(outcomes, [0.5, 0.3, 0.2])
    assert pmf.sum() == pytest.approx(1.0)
    # P(4, 0, 0) = 0.5 ** 4
    assert pmf[outcomes.tolist().index([4, 0, 0])] == pytest.approx(0.0625)
    edge = finite.multinomial_pmf(finite.compositions(3, 2), [1.0, 0.0])
    assert edge.tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="too many outcomes"):
        finite.compositions(400, 6)
    with pytest.raises(ValueError, match="n >= 0"):
        finite.compositions(3, 0)


def test_linear_metrics_are_exactly_unbiased():
    h = np.array([0.6, 0.3, 0.1])
    p = np.array([0.5, 0.2, 0.3])
    truth = M.item_metrics(h[None, :], p[None, :])
    for n in (1, 3, 10):
        expected = finite.expected_plugin(h, p, n)
        assert expected["cross_entropy"] == pytest.approx(truth["cross_entropy"][0])
        assert expected["annotator_match"] == pytest.approx(truth["annotator_match"][0])


def test_plugin_divergences_are_biased_upwards_and_shrink_with_n():
    h = np.array([0.6, 0.3, 0.1])
    p = np.array([0.5, 0.2, 0.3])
    truth = M.item_metrics(h[None, :], p[None, :])
    previous = None
    for n in (3, 5, 10, 100):
        expected = finite.expected_plugin(h, p, n)
        bias = {name: expected[name] - truth[name][0] for name in finite.CORRECTABLE}
        assert all(value > 0 for value in bias.values())
        if previous is not None:
            assert all(bias[name] < previous[name] for name in finite.CORRECTABLE)
        previous = bias
    # first-order theory: the KL bias is (K - 1) / (2 n); at n = 100 it holds within 10 %
    assert previous["kl"] == pytest.approx((3 - 1) / 200, rel=0.10)


def test_expected_plugin_agrees_with_seeded_monte_carlo():
    h = np.array([0.7, 0.2, 0.1])
    p = np.array([0.4, 0.4, 0.2])
    exact = finite.expected_plugin(h, p, 5)
    rng = np.random.default_rng(5)
    counts = sample_annotators(np.tile(h, (40000, 1)), 5, rng)
    simulated = M.item_metrics(counts, np.tile(p, (40000, 1)))
    # 40000 draws put the Monte Carlo standard error of each mean below 0.003
    for name in ("kl", "js_divergence", "total_variation", "majority_accuracy"):
        assert simulated[name].mean() == pytest.approx(exact[name], abs=0.01)


def test_expected_plugin_rejects_bad_input():
    with pytest.raises(ValueError, match="same labels"):
        finite.expected_plugin([0.5, 0.5], [0.2, 0.3, 0.5], 3)
    with pytest.raises(ValueError, match="at least 1"):
        finite.expected_plugin([0.5, 0.5], [0.5, 0.5], 0)


def test_corrections_remove_most_of_the_bias_in_a_seeded_population():
    rng = np.random.default_rng(6)
    human = rng.dirichlet(np.full(3, 0.5), size=3000)
    human = np.maximum(human, 1e-12)
    human /= human.sum(axis=1, keepdims=True)
    model = 0.7 * human + 0.3 * rng.dirichlet(np.ones(3), size=3000)
    truth = M.aggregate(M.item_metrics(human, model))
    counts = sample_annotators(human, 10, rng)
    out = finite.corrections(counts, model, n_splits=10, seed=0)
    for name in finite.CORRECTABLE:
        plug_bias = out[name]["plug_in"] - truth[name]
        jack_bias = out[name]["jackknife"] - truth[name]
        split_bias = out[name]["split_half"] - truth[name]
        # stated tolerance: at 10 annotations the corrected estimates keep under 40 % of the
        # plug-in bias (measured here: under 30 % for every metric)
        assert plug_bias > 0
        assert abs(jack_bias) < 0.4 * plug_bias
        assert abs(split_bias) < 0.4 * plug_bias
        assert out[name]["estimated_bias"] == pytest.approx(
            out[name]["plug_in"] - out[name]["jackknife"]
        )
    assert "miller_madow" in out["kl"] and "miller_madow" not in out["total_variation"]


def test_annotation_stats():
    assert finite.annotation_stats([[3, 0], [0, 0], [1, 4], [1, 1]]) == {
        "min": 2.0,
        "median": 3.0,
        "max": 5.0,
    }
    assert finite.annotation_stats([[0, 0]]) == {"min": 0.0, "median": 0.0, "max": 0.0}


def test_jackknife_matches_brute_force_over_annotators():
    rng = np.random.default_rng(8)
    counts = rng.integers(0, 4, size=(30, 3))
    counts[counts.sum(axis=1) < 2] = [1, 1, 0]
    probs = rng.dirichlet(np.ones(3), size=30)
    fast = finite.jackknife_items(counts, probs)
    for i in range(30):
        annotations = np.repeat(np.arange(3), counts[i])
        n = len(annotations)
        full = M.item_metrics(counts[i : i + 1], probs[i : i + 1])
        left_out = {name: 0.0 for name in finite.CORRECTABLE}
        for j in range(n):
            rest = np.bincount(np.delete(annotations, j), minlength=3)[None, :]
            values = M.item_metrics(rest, probs[i : i + 1])
            for name in finite.CORRECTABLE:
                left_out[name] += values[name][0] / n
        for name in finite.CORRECTABLE:
            expected = n * full[name][0] - (n - 1) * left_out[name]
            assert fast[name][i] == pytest.approx(expected, abs=1e-9)

import math

import numpy as np
import pytest
from scipy import stats
from scipy.spatial import distance

from softeval import metrics as M

LN2 = math.log(2.0)


def test_distributions_and_empty_row():
    h = M.distributions([[3, 1, 0], [0, 0, 0]])
    assert h[0] == pytest.approx([0.75, 0.25, 0.0])
    assert np.all(np.isnan(h[1]))


def test_entropy_by_hand():
    out = M.entropy([[0.5, 0.5], [1.0, 0.0], [0.25, 0.75]])
    assert out[0] == pytest.approx(LN2)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(-(0.25 * math.log(0.25) + 0.75 * math.log(0.75)))


def test_cross_entropy_and_kl_by_hand():
    h = np.array([[0.5, 0.5, 0.0]])
    p = np.array([[0.25, 0.25, 0.5]])
    assert M.cross_entropy(h, p)[0] == pytest.approx(-math.log(0.25))
    assert M.kl_divergence(h, p)[0] == pytest.approx(math.log(2.0))
    assert M.kl_divergence(h, h)[0] == pytest.approx(0.0, abs=1e-12)


def test_cross_entropy_clips_zero_probability():
    h = np.array([[0.5, 0.5]])
    p = np.array([[1.0, 0.0]])
    assert M.cross_entropy(h, p, eps=1e-3)[0] == pytest.approx(-0.5 * math.log(1e-3))


def test_js_and_tv_by_hand():
    h = np.array([[1.0, 0.0], [0.5, 0.5]])
    p = np.array([[0.0, 1.0], [0.5, 0.5]])
    assert M.js_divergence(h, p) == pytest.approx([LN2, 0.0], abs=1e-12)
    assert M.total_variation(h, p) == pytest.approx([1.0, 0.0])


def test_divergences_match_scipy():
    rng = np.random.default_rng(0)
    h = rng.dirichlet(np.ones(4), size=50)
    p = rng.dirichlet(np.ones(4), size=50)
    for i in range(50):
        assert M.kl_divergence(h[i : i + 1], p[i : i + 1])[0] == pytest.approx(
            stats.entropy(h[i], p[i]), rel=1e-9
        )
        assert M.js_distance(h[i : i + 1], p[i : i + 1])[0] == pytest.approx(
            distance.jensenshannon(h[i], p[i]), rel=1e-9
        )
        assert M.entropy(h[i : i + 1])[0] == pytest.approx(stats.entropy(h[i]), rel=1e-9)
        assert M.total_variation(h[i : i + 1], p[i : i + 1])[0] == pytest.approx(
            0.5 * distance.cityblock(h[i], p[i]), rel=1e-9
        )


def test_annotator_match_is_human_mass_on_model_answer():
    h = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    p = np.array([[0.1, 0.8, 0.1], [0.4, 0.4, 0.2]])
    out = M.annotator_match(h, p)
    assert out[0] == pytest.approx(0.3)
    # the model ties between labels 0 and 1, so its answer is drawn evenly between them
    assert out[1] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("ties", "expected"),
    [("split", [1.0, 0.5, 0.0, 0.5]), ("any", [1.0, 1.0, 0.0, 1.0])],
)
def test_majority_accuracy_tie_policies(ties, expected):
    counts = [[3, 1, 0], [2, 2, 0], [2, 2, 0], [2, 2, 0]]
    probs = [[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.0, 0.1, 0.9], [0.0, 1.0, 0.0]]
    assert M.majority_accuracy(counts, probs, ties=ties) == pytest.approx(expected)


def test_majority_accuracy_exclude_and_empty():
    counts = [[3, 1, 0], [2, 2, 0], [0, 0, 0]]
    probs = [[0.9, 0.1, 0.0]] * 3
    out = M.majority_accuracy(counts, probs, ties="exclude")
    assert out[0] == pytest.approx(1.0)
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert M.majority_ties(counts).tolist() == [False, True, False]
    with pytest.raises(ValueError, match="tie policy"):
        M.majority_accuracy(counts, probs, ties="first")


def test_model_tie_against_tied_majority():
    # both sides tie over the same two labels: independent uniform draws agree half the time
    out = M.majority_accuracy([[2, 2, 0]], [[0.5, 0.5, 0.0]], ties="split")
    assert out[0] == pytest.approx(0.5)


def test_rank_and_correlations_match_scipy():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, size=40).astype(float)
    y = x + rng.normal(size=40)
    assert M.rankdata(x) == pytest.approx(stats.rankdata(x))
    assert M.pearson(x, y) == pytest.approx(stats.pearsonr(x, y).statistic, rel=1e-9)
    assert M.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, rel=1e-9)
    assert math.isnan(M.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert math.isnan(M.spearman([1.0], [2.0]))


def test_miller_madow_adds_bins_minus_one_over_2n():
    counts = [[2, 2, 0], [5, 0, 0]]
    plain = M.entropy(M.distributions(counts))
    out = M.miller_madow_entropy(counts)
    assert out[0] == pytest.approx(plain[0] + 1 / 8)
    assert out[1] == pytest.approx(0.0)


def test_summarise_skips_empty_rows_and_counts_ties():
    counts = [[3, 1, 0], [0, 0, 0], [2, 2, 0]]
    probs = [[0.75, 0.25, 0.0], [0.2, 0.3, 0.5], [0.5, 0.5, 0.0]]
    s = M.summarise(counts, probs)
    assert (s.n_items, s.n_empty, s.n_tied) == (2, 1, 1)
    assert s.metrics["total_variation"] == pytest.approx(0.0, abs=1e-12)
    assert s.metrics["kl"] == pytest.approx(0.0, abs=1e-12)
    assert s.metrics["annotator_match"] == pytest.approx((0.75 + 0.5) / 2)
    assert s.metrics["majority_accuracy"] == pytest.approx((1.0 + 0.5) / 2)
    assert s.to_dict()["metrics"]["js_divergence"] == pytest.approx(0.0, abs=1e-12)


def test_hard_predictions_blank_the_undefined_metrics():
    counts = [[3, 1, 0], [1, 3, 0]]
    s = M.summarise(counts, M.one_hot([0, 0], 3))
    assert s.hard_predictions
    assert math.isnan(s.metrics["cross_entropy"])
    assert math.isnan(s.metrics["entropy_pearson"])
    assert s.metrics["annotator_match"] == pytest.approx(0.5)
    assert s.metrics["total_variation"] == pytest.approx((0.25 + 0.75) / 2)
    assert s.to_dict()["metrics"]["cross_entropy"] is None


def test_entropy_correlation_tracks_uncertainty():
    counts = np.array([[10, 0, 0], [7, 3, 0], [4, 3, 3], [5, 5, 0]])
    h = M.distributions(counts)
    s = M.summarise(counts, np.clip(h, 1e-6, None) / np.clip(h, 1e-6, None).sum(1, keepdims=True))
    assert s.metrics["entropy_spearman"] == pytest.approx(1.0)
    assert s.metrics["entropy_pearson"] > 0.99


def test_softmax_and_validation():
    assert M.softmax([[0.0, 0.0], [1000.0, 0.0]]) == pytest.approx(
        np.array([[0.5, 0.5], [1.0, 0.0]])
    )
    with pytest.raises(ValueError, match="sum to 1"):
        M.as_probs([[0.5, 0.2]])
    with pytest.raises(ValueError, match="negative"):
        M.as_counts([[1, -1]])
    with pytest.raises(ValueError, match="2-D"):
        M.as_counts([1, 2])
    with pytest.raises(ValueError, match="same shape"):
        M.item_metrics([[1, 2, 3]], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="out of range"):
        M.one_hot([3], 3)

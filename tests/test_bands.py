import math

import numpy as np
import pytest

from softeval import bands
from softeval import metrics as M


def test_agreement_by_share_and_entropy():
    counts = [[4, 0], [2, 2], [3, 1], [0, 0]]
    share = bands.agreement(counts, "share")
    assert share[:3] == pytest.approx([1.0, 0.5, 0.75])
    assert math.isnan(share[3])
    ent = bands.agreement(counts, "entropy")
    assert ent[:2] == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError, match="unknown banding"):
        bands.agreement(counts, "kappa")


def test_band_edges_are_half_open_with_a_closed_last_band():
    values = np.array([0.5, 0.7, 0.9, 1.0, 0.34, np.nan])
    masks = bands.band_masks(values, bands.SHARE_EDGES)
    members = [mask.tolist() for _, _, _, mask in masks]
    assert members[0] == [False, False, False, False, True, False]
    assert members[1] == [True, False, False, False, False, False]
    assert members[2] == [False, True, False, False, False, False]
    assert members[3] == [False, False, True, True, False, False]
    with pytest.raises(ValueError, match="increasing"):
        bands.band_masks(values, [0.5, 0.2])
    with pytest.raises(ValueError, match="increasing"):
        bands.band_masks(values, [0.5])


def test_bands_partition_the_annotated_items_and_match_direct_scoring():
    rng = np.random.default_rng(0)
    human = rng.dirichlet(np.full(3, 0.6), size=300)
    counts = rng.multinomial(10, human).astype(float)
    counts[0] = 0
    model = 0.8 * human + 0.2 / 3
    out = bands.score_bands(counts, model)
    assert sum(band.n_items for band in out) == 299
    assert [band.label for band in out] == ["[0, 0.5)", "[0.5, 0.7)", "[0.7, 0.9)", "[0.9, 1]"]
    share = bands.agreement(counts)
    top = (share >= 0.9) & ~np.isnan(share)
    direct = M.summarise(counts[top], model[top])
    assert out[-1].metrics["total_variation"] == pytest.approx(direct.metrics["total_variation"])
    assert out[-1].n_items == direct.n_items
    assert out[0].to_dict()["band"] == "[0, 0.5)"


def test_custom_edges_entropy_bands_and_empty_band():
    counts = [[10, 0], [9, 1], [5, 5]]
    probs = [[0.9, 0.1], [0.9, 0.1], [0.5, 0.5]]
    out = bands.score_bands(counts, probs, by="entropy", edges=[0, 0.2, 0.6, 1])
    assert [band.n_items for band in out] == [1, 1, 1]
    assert out[2].metrics["total_variation"] == pytest.approx(0.0, abs=1e-12)
    sparse = bands.score_bands(counts, probs, edges=[0, 0.2, 1])
    assert sparse[0].n_items == 0
    assert math.isnan(sparse[0].metrics["total_variation"])

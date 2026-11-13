# softeval

Score a model's predictions against the distribution of labels that several annotators gave, instead of against
one majority label, and report what annotators themselves score on the same metrics.

## Why this is hard

On many items people disagree for good reasons, and a majority label throws that away. Accuracy against the
majority then rewards a confident answer on an item that has no clear answer, and it cannot tell a model that
knows an item is contested from one that does not. Comparing the model's probabilities with the annotators'
label distribution fixes that, but brings its own problem: with three or five annotations per item the observed
distribution is a noisy estimate, so distances to it are too large on average, and by a different amount for
each model. A number such as "total variation 0.33" means little until one knows how much of it is annotation
noise and what a perfect predictor of the human distribution would score.

## Install

```
pip install git+https://github.com/vipul21435/softeval
```

Python 3.10 or newer. The only runtime dependency is numpy.

## Quickstart

`examples/toy` holds a small made-up file: eight items, three labels, up to five annotators each, one item with
no annotations, and two sets of predictions.

```
$ softeval score examples/toy/human.jsonl examples/toy/model_a.jsonl --labels entailment,neutral,contradiction
Items scored: 7 (no annotations: 1; tied majority: 2, scored as 'split')
Annotations per item: min 4, median 5, max 5

| metric | value | 95% interval |
| --- | --- | --- |
| cross-entropy (nats) | 0.6109 | [0.3333, 0.8535] |
| KL divergence, human to model (nats) | 0.0577 | [0.0320, 0.0841] |
| Jensen-Shannon divergence (nats) | 0.0193 | [0.0101, 0.0286] |
| Jensen-Shannon distance | 0.1287 | [0.0881, 0.1658] |
| total variation distance | 0.0786 | [0.0643, 0.0929] |
| match with a random annotator | 0.7000 | [0.5429, 0.8714] |
| accuracy against the majority | 0.8571 | [0.7143, 1.0000] |
| entropy correlation (Pearson) | 0.9967 | [0.9903, 1.0000] |
| entropy correlation (Spearman) | 0.9910 | [0.8847, 1.0000] |

Divergences corrected for the number of annotations (the plug-in value is too large when annotations are few):

| metric | plug-in | jackknife | split-half | Miller-Madow | estimated bias |
| --- | --- | --- | --- | --- | --- |
| KL divergence, human to model (nats) | 0.0577 | -0.1201 | -0.1073 | -0.0458 | 0.1778 |
| Jensen-Shannon divergence (nats) | 0.0193 | -0.0354 | -0.0316 |  | 0.0547 |
| Jensen-Shannon distance | 0.1287 | -0.0165 | 0.0270 |  | 0.1451 |
| total variation distance | 0.0786 | -0.0964 | -0.0315 |  | 0.1750 |
```

The second table is why the first one should not be read alone. With five annotators, the distances in the
first table are smaller than the noise in the annotations: the corrected estimates come out below zero, which
says that these seven items cannot distinguish model A from a model that matches the human distribution
exactly. `softeval ceiling examples/toy/human.jsonl --labels entailment,neutral,contradiction` says the same
from the other side: the human distribution itself, scored against as many fresh annotators as each item has,
shows a total variation of 0.1346.

Two models on the same items, with a paired bootstrap:

```
$ softeval compare examples/toy/human.jsonl examples/toy/model_a.jsonl examples/toy/model_b.jsonl --labels entailment,neutral,contradiction
Items scored: 7; paired bootstrap over items, 1000 resamples
A = examples/toy/model_a.jsonl
B = examples/toy/model_b.jsonl

| metric | A | B | A - B | 95% interval | share of resamples A better |
| --- | --- | --- | --- | --- | --- |
| cross-entropy (nats) | 0.6109 | 1.3064 | -0.6956 | [-1.0928, -0.2541] | 1.0000 |
| KL divergence, human to model (nats) | 0.0577 | 0.7533 | -0.6956 | [-1.0928, -0.2541] | 1.0000 |
| Jensen-Shannon divergence (nats) | 0.0193 | 0.1062 | -0.0869 | [-0.1493, -0.0195] | 0.9960 |
| Jensen-Shannon distance | 0.1287 | 0.2886 | -0.1599 | [-0.2963, -0.0033] | 0.9780 |
| total variation distance | 0.0786 | 0.2914 | -0.2129 | [-0.3700, -0.0428] | 0.9950 |
| match with a random annotator | 0.7000 | 0.7000 | 0.0000 | [0.0000, 0.0000] | 0.0000 |
| accuracy against the majority | 0.8571 | 0.8571 | 0.0000 | [0.0000, 0.0000] | 0.0000 |
| entropy correlation (Pearson) | 0.9967 | 0.6928 | 0.3038 | [0.0347, 0.8878] | 0.9939 |
| entropy correlation (Spearman) | 0.9910 | 0.4723 | 0.5187 | [0.0287, 1.4260] | 0.9830 |
```

Model B gives the same top answer as model A on every item, so accuracy and match cannot separate them; B is
overconfident, and every distribution metric shows it.

The other commands:

```
softeval ceiling  HUMAN                    # what annotators score against each other
softeval bands    HUMAN PREDICTIONS        # every metric by human-agreement band
softeval simulate                          # bias of each estimate at 3, 5, 10 and 100 annotations
```

Every command takes `--format json`. `score`, `ceiling`, `compare` and `bands` take `--annotations M` to keep
only M randomly chosen annotators per item, `--ties split|any|exclude` and `--seed`. `bands` groups items by
the share of annotators on the most common label, or by entropy with `--by entropy`; its Markdown table shows
six of the metrics and its JSON output all of them. `examples/quickstart.py` does the same through the Python
API.

### Input

Human labels, JSONL or CSV, one of:

- counts per item: `{"id": "a", "counts": {"yes": 3, "no": 2}}`, or a CSV with `id` and one column per label;
- a list per item: `{"id": "a", "annotations": ["yes", "no", "yes"]}`;
- one row per annotation: `{"id": "a", "label": "yes"}`, or a CSV with `id` and `label` columns.

Predictions, JSONL or CSV: a label (`{"id": "a", "label": "yes"}`) or probabilities
(`{"id": "a", "probs": {"yes": 0.7, "no": 0.3}}`, or a CSV with `id` and one column per label).

## How it works

For one item let *c_k* be the number of annotators who chose label *k*, *n* their total, *h_k = c_k / n* the
observed distribution and *p_k* the model's probability. Logarithms are natural. Each metric is computed per
item and averaged over the items that have at least one annotation; the others are counted and skipped.

| metric | per item |
| --- | --- |
| cross-entropy | `-sum_k h_k log p_k`, with `p_k` raised to 1e-12 where smaller |
| KL divergence | cross-entropy minus the entropy of `h` |
| Jensen-Shannon divergence | `KL(h, m)/2 + KL(p, m)/2` with `m = (h + p)/2`; at most `log 2` |
| Jensen-Shannon distance | square root of the divergence (what `scipy.spatial.distance.jensenshannon` returns) |
| total variation | `sum_k abs(h_k - p_k) / 2` |
| match with a random annotator | `h` at the model's top label: the chance that a randomly drawn annotator of the item gave the model's answer |
| accuracy against the majority | 1 if the model's top label is the most common one, with ties as below |
| entropy correlation | Pearson and Spearman correlation, across items, between the entropy of `p` and the entropy of `h` |

Ties. When several labels share the highest count, `--ties split` (the default) treats the gold label as drawn
uniformly among them, so a prediction inside a two-way tie earns 0.5; `any` accepts any of the tied labels;
`exclude` drops tied items from the accuracy. The number of tied items is always printed. Ties in the model's
own probabilities are broken uniformly. Predictions given as a label are scored as a one-hot distribution;
cross-entropy, KL and the entropy correlations are then undefined and print as `n/a`.

Few annotations. Treat an item's annotators as *n* draws from a population distribution. The observed
distribution is then right on average, so the metrics that are linear in it (cross-entropy, match) are unbiased
at any *n*. The divergences are convex in it, so their plug-in values are too large on average; for KL the
leading term is `(K - 1) / (2n)` for *K* labels with non-zero probability. softeval reports, next to the
plug-in value:

- the leave-one-annotator-out jackknife, `n D_n - (n - 1) mean_j D_(-j)`, which removes a bias proportional to
  `1/n` ([Jackknife resampling](https://en.wikipedia.org/wiki/Jackknife_resampling)). Removing an annotator who
  chose label *k* happens for *c_k* of the *n* annotators, so the mean is a weighted sum over labels and needs
  no sampling;
- a split-half extrapolation, `(n D_n - r D_r) / (n - r)` with `D_r` the metric on a random half of
  `r = n // 2` annotators, averaged over random halves, under the same `1/n` assumption;
- for KL, the plug-in value minus `(m - 1) / (2n)` with *m* labels observed, the first-order correction of the
  plug-in entropy usually called Miller-Madow (background:
  [Entropy estimation](https://en.wikipedia.org/wiki/Entropy_estimation)).

`softeval.finite.expected_plugin` gives the exact expectation of every plug-in metric for one item by
enumerating all outcomes of *n* annotators, for a handful of labels; the tests use it to check the sampling
code and the ceilings.

Ceilings, from the annotations alone:

- one annotator against the others: each annotator's label is scored as a prediction against the remaining
  annotators of the item. Its match is `sum_k c_k (c_k - 1) / (n (n - 1))`, the chance that two different
  annotators of the item agree;
- the full distribution against a fresh sample: the observed distribution is scored against `m` new annotators
  drawn from it. This is what a model that predicted the human distribution exactly would get with `m`
  annotations. For cross-entropy, match and accuracy no prediction does better in expectation. For KL,
  Jensen-Shannon and total variation it is the reference point rather than a strict bound: against very few
  annotators a sharper prediction can score lower than the exact one;
- half against half: the distribution of a random half of the annotators (with 0.5 added to each count) scored
  against the other half.

Intervals are percentile bootstrap intervals from resampling items with replacement; `compare` resamples the
same items for both models and reports the difference.

Background on evaluating against label distributions:
Pavlick and Kwiatkowski, [Inherent Disagreements in Human Textual Inferences](https://aclanthology.org/Q19-1043/)
(TACL 2019);
Peterson, Battleday, Griffiths and Russakovsky,
[Human uncertainty makes classification more robust](https://arxiv.org/abs/1908.07086) (2019, CIFAR10H);
Nie, Zhou and Bansal,
[What Can We Learn from Collective Human Opinions on Natural Language Inference Data?](https://aclanthology.org/2020.emnlp-main.734/)
(EMNLP 2020, ChaosNLI);
Uma, Fornaciari, Hovy, Paun, Plank and Poesio,
[Learning from Disagreement: A Survey](https://jair.org/index.php/jair/article/view/12752) (JAIR 2021);
Plank, [The "Problem" of Human Label Variation](https://aclanthology.org/2022.emnlp-main.731/) (EMNLP 2022);
and the shared tasks
[SemEval-2021 Task&nbsp;12: Learning with Disagreements](https://aclanthology.org/2021.semeval-1.41/)
(Uma, Fornaciari, Dumitrache, Miller, Chamberlain, Plank, Simpson and Poesio) and
[SemEval-2023 Task&nbsp;11: Learning with Disagreements (LeWiDi)](https://aclanthology.org/2023.semeval-1.314/)
(Leonardelli, Abercrombie, Almanea, Basile, Fornaciari, Plank, Rieser, Uma and Poesio).

## Which numbers can be trusted at 3, 5, 10 and 100 annotations

`softeval simulate` draws items with a known population distribution (symmetric Dirichlet) and a model that is
that distribution mixed with noise, so every metric has a true value; it then draws annotators and compares
each estimate with the truth.

```
$ softeval simulate
Simulation: 2000 items, 3 labels, Dirichlet alpha 0.5, model noise 0.3, 20 replicates, seed 0
Bias = mean estimate minus the value against the population distribution.

| metric | estimator | truth | bias at 3 | bias at 5 | bias at 10 | bias at 100 | within 5% from |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cross-entropy (nats) | plug-in | 0.7516 | -0.0007 | -0.0007 | 0.0001 | -0.0001 | 3 |
| KL divergence, human to model (nats) | plug-in | 0.0870 | 0.2757 | 0.1765 | 0.0921 | 0.0098 | not in this run |
| KL divergence, human to model (nats) | jackknife | 0.0870 | 0.0511 | 0.0209 | 0.0054 | 0.0001 | 100 |
| KL divergence, human to model (nats) | split-half | 0.0870 | 0.0822 | 0.0348 | 0.0087 | 0.0002 | 100 |
| KL divergence, human to model (nats) | Miller-Madow | 0.0870 | 0.1715 | 0.0878 | 0.0328 | 0.0011 | 100 |
| Jensen-Shannon divergence (nats) | plug-in | 0.0260 | 0.0843 | 0.0546 | 0.0288 | 0.0031 | not in this run |
| Jensen-Shannon divergence (nats) | jackknife | 0.0260 | 0.0191 | 0.0076 | 0.0020 | 0.0000 | 100 |
| Jensen-Shannon divergence (nats) | split-half | 0.0260 | 0.0303 | 0.0128 | 0.0033 | 0.0001 | 100 |
| Jensen-Shannon distance | plug-in | 0.1436 | 0.1706 | 0.1215 | 0.0717 | 0.0094 | not in this run |
| Jensen-Shannon distance | jackknife | 0.1436 | 0.0772 | 0.0408 | 0.0157 | 0.0004 | 100 |
| Jensen-Shannon distance | split-half | 0.1436 | 0.1023 | 0.0574 | 0.0224 | 0.0008 | 100 |
| total variation distance | plug-in | 0.1377 | 0.1560 | 0.1065 | 0.0602 | 0.0075 | not in this run |
| total variation distance | jackknife | 0.1377 | 0.0466 | 0.0285 | 0.0108 | 0.0001 | 100 |
| total variation distance | split-half | 0.1377 | 0.0633 | 0.0368 | 0.0148 | 0.0005 | 100 |
| match with a random annotator | plug-in | 0.6928 | 0.0019 | 0.0010 | -0.0001 | 0.0001 | 3 |
| accuracy against the majority | plug-in | 0.9150 | -0.1525 | -0.1213 | -0.0792 | -0.0124 | 100 |
| entropy correlation (Pearson) | plug-in | 0.7816 | -0.4010 | -0.2871 | -0.1663 | -0.0199 | 100 |
| entropy correlation (Pearson) | Miller-Madow | 0.7816 | -0.4016 | -0.2888 | -0.1692 | -0.0203 | 100 |
| entropy correlation (Spearman) | plug-in | 0.7945 | -0.4098 | -0.2950 | -0.1694 | -0.0199 | 100 |
| entropy correlation (Spearman) | Miller-Madow | 0.7945 | -0.4098 | -0.2950 | -0.1737 | -0.0203 | 100 |
```

Reading it, for this setting:

- Cross-entropy and match with a random annotator are unbiased from 3 annotations on. Their intervals are wider
  with fewer annotations, but the values can be compared across datasets with different numbers of annotators.
- Plug-in KL, Jensen-Shannon and total variation cannot be read as they stand at 3, 5 or 10 annotations: at 5
  the plug-in KL is three times the true value. At 100 they are still 5 to 12 % too large.
- The jackknife removes most of that: at 10 annotations it is 6 to 11 % above the truth, at 100 it is within
  0.3 %. At 3 and 5 annotations it is still 20 to 75 % too large, so corrected divergences from so few
  annotations are closer than the plug-in values but are not accurate. The split-half estimate is a little
  worse than the jackknife everywhere; Miller-Madow is worse than both at every size tried.
- Accuracy against the majority is too low when the majority itself is uncertain: 0.12 too low at 5
  annotations, 0.012 at 100.
- The entropy correlation is attenuated by the noise in the human entropies: 0.29 too low at 5 annotations,
  0.02 at 100. Correcting the entropies with Miller-Madow does not help.

The relative size of these biases depends on how close the model is to the human distribution (`--noise`) and
on how peaked the human distributions are (`--alpha`): the closer the model, the larger the share of its
measured distance that is annotation noise. Run the simulation with settings that resemble the data at hand.

## Results on ChaosNLI

[ChaosNLI](https://github.com/easonnie/ChaosNLI) (Nie, Zhou and Bansal, 2020) has 100 new annotations per item
for 1,514 SNLI, 1,599 MNLI and 1,532 alphaNLI development items, next to the label each item had in the
original dataset. Its repository states the licence as Creative Commons Attribution-NonCommercial 4.0, so no
ChaosNLI data is committed here; the scripts below download it.

What could not be run: the authors also published saved predictions of several models. On
2026-11-13 the two download links in their repository (the dataset archive and the prediction archive) served
a "file deleted" page. The dataset files are available from a
[preservation mirror](https://huggingface.co/datasets/earino/chaosnli) (same licence; `fetch.py` checks the
files against the SHA-256 sums the mirror publishes), but the mirror does not carry the model predictions. The
comparison of those models is therefore not here. The tables use predictors that the public files themselves
contain; `prepare.py` also converts the authors' prediction format, which is tested on a made-up file only.

```
python examples/chaosnli/fetch.py data/chaosnli
python examples/chaosnli/prepare.py data/chaosnli data/chaosnli/prepared
python examples/chaosnli/tables.py data/chaosnli/prepared
```

The last command takes about 5 seconds (measured on a loaded laptop) and prints the tables below, and the same
three for alphaNLI. The predictors:

- `old-label`: all probability on the label the item had in the original dataset;
- `old-five`: the distribution of the five original annotations, with 0.5 added to each count;
- `old-prior`: one distribution for every item, the pooled original labels of the subset;
- `random-baseline`: the random-baseline prediction file in the ChaosNLI repository;
- `uniform`: equal probability on every label.

### Two views of the same predictors

"acc. vs old label" is accuracy against the original single label; everything else is against the 100 new
annotations. Rank 1 is best.

SNLI subset (1,514 items):

| predictor | acc. vs old label | acc. vs majority of 100 | match | TV | JS div. | cross-entropy | rank by acc. | rank by JS div. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old-five | 1.0000 | 0.7503 | 0.6570 | 0.2778 | 0.0711 | 0.8032 | 1 | 1 |
| old-label | 1.0000 | 0.7503 | 0.6570 | 0.3430 | 0.1569 | n/a | 1 | 4 |
| old-prior | 0.4472 | 0.5363 | 0.5025 | 0.4010 | 0.1346 | 1.0346 | 3 | 2 |
| random-baseline | 0.3278 | 0.3334 | 0.3365 | 0.5327 | 0.2447 | 1.8479 | 4 | 5 |
| uniform | 0.3333 | 0.3333 | 0.3333 | 0.4400 | 0.1542 | 1.0986 | 5 | 3 |

MNLI subset (1,599 items):

| predictor | acc. vs old label | acc. vs majority of 100 | match | TV | JS div. | cross-entropy | rank by acc. | rank by JS div. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old-five | 1.0000 | 0.6823 | 0.5610 | 0.2106 | 0.0404 | 0.8958 | 1 | 1 |
| old-label | 1.0000 | 0.6823 | 0.5610 | 0.4390 | 0.1972 | n/a | 1 | 5 |
| old-prior | 0.4509 | 0.3652 | 0.3928 | 0.3267 | 0.0892 | 1.0728 | 3 | 2 |
| random-baseline | 0.3635 | 0.3530 | 0.3460 | 0.4739 | 0.1942 | 1.7889 | 4 | 4 |
| uniform | 0.3333 | 0.3333 | 0.3333 | 0.3443 | 0.0997 | 1.0986 | 5 | 3 |

The original label and the five original annotations have the same top answer on every item, so no accuracy
can separate them; the distribution metrics do (Jensen-Shannon 0.0711 against 0.1569 on SNLI). The original
label is first by accuracy and fourth of five by Jensen-Shannon divergence on SNLI, behind a constant prior and
the uniform distribution; on MNLI it is last, behind the random baseline. The ranking also depends on
which distance is used: by total variation the original label is second on SNLI and fourth on MNLI.

### Human ceilings

SNLI subset:

| human reference | acc. vs majority | match | TV | JS div. | cross-entropy |
| --- | --- | --- | --- | --- | --- |
| one annotator vs the other 99 | 0.7489 | 0.6580 | 0.3420 | 0.1573 | n/a |
| 50 annotators vs the other 50 | 0.9322 | 0.7487 | 0.0729 | 0.0098 | 0.5794 |
| full distribution vs 100 fresh annotators | 0.9655 | 0.7546 | 0.0343 | 0.0023 | 0.5533 |
| full distribution vs 5 fresh annotators | 0.8583 | 0.7543 | 0.1545 | 0.0400 | 0.5535 |

MNLI subset:

| human reference | acc. vs majority | match | TV | JS div. | cross-entropy |
| --- | --- | --- | --- | --- | --- |
| one annotator vs the other 99 | 0.6425 | 0.5397 | 0.4603 | 0.2135 | n/a |
| 50 annotators vs the other 50 | 0.8731 | 0.6397 | 0.0881 | 0.0102 | 0.7713 |
| full distribution vs 100 fresh annotators | 0.9370 | 0.6508 | 0.0431 | 0.0025 | 0.7431 |
| full distribution vs 5 fresh annotators | 0.7663 | 0.6508 | 0.1963 | 0.0529 | 0.7443 |

On SNLI the original label agrees with the majority of 100 on 75.0 % of the items, which is what one of the 100
annotators scores against the other 99 (74.9 %). A predictor that knew the human distribution exactly would
reach a match of 0.75, not 1, and a total variation of 0.034 against 100 annotators; `old-five`, the best
predictor above, is at 0.278. Against 5 annotators the same exact predictor would show a total variation of
0.155, which is the size of the annotation noise at that number of annotations.

### With only 5 of the 100 annotations

Each item keeps 5 randomly chosen annotations; values are means over 20 such draws (seed 0).

SNLI subset:

| predictor | acc. (100) | acc. (5) | match (100) | match (5) | TV (100) | TV (5) plug-in | TV (5) jackknife | JS (100) | JS (5) plug-in | JS (5) jackknife |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old-five | 0.7503 | 0.7194 | 0.6570 | 0.6588 | 0.2778 | 0.3311 | 0.2756 | 0.0711 | 0.1178 | 0.0780 |
| old-label | 0.7503 | 0.7194 | 0.6570 | 0.6588 | 0.3430 | 0.3412 | 0.3412 | 0.1569 | 0.1696 | 0.1537 |
| old-prior | 0.5363 | 0.5270 | 0.5025 | 0.5039 | 0.4010 | 0.4432 | 0.4051 | 0.1346 | 0.1823 | 0.1423 |
| random-baseline | 0.3334 | 0.3368 | 0.3365 | 0.3373 | 0.5327 | 0.5615 | 0.5375 | 0.2447 | 0.2834 | 0.2516 |
| uniform | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.4400 | 0.4814 | 0.4386 | 0.1542 | 0.2023 | 0.1624 |

MNLI subset:

| predictor | acc. (100) | acc. (5) | match (100) | match (5) | TV (100) | TV (5) plug-in | TV (5) jackknife | JS (100) | JS (5) plug-in | JS (5) jackknife |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old-five | 0.6823 | 0.6191 | 0.5610 | 0.5588 | 0.2106 | 0.2970 | 0.2206 | 0.0404 | 0.0969 | 0.0435 |
| old-label | 0.6823 | 0.6191 | 0.5610 | 0.5588 | 0.4390 | 0.4412 | 0.4412 | 0.1972 | 0.2168 | 0.1940 |
| old-prior | 0.3652 | 0.3822 | 0.3928 | 0.3925 | 0.3267 | 0.3905 | 0.3334 | 0.0892 | 0.1488 | 0.0937 |
| random-baseline | 0.3530 | 0.3507 | 0.3460 | 0.3457 | 0.4739 | 0.5154 | 0.4805 | 0.1942 | 0.2425 | 0.1987 |
| uniform | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3443 | 0.4104 | 0.3449 | 0.0997 | 0.1593 | 0.1043 |

- Match does not move (0.6570 with 100 annotations, 0.6588 with 5 on SNLI), as expected for a metric that is
  linear in the human distribution.
- Accuracy against the majority drops (0.7503 to 0.7194 on SNLI, 0.6823 to 0.6191 on MNLI) although the
  predictor is the same, because the majority of 5 is a noisier target.
- The plug-in distances grow, and not by the same amount for every predictor: on MNLI the
  Jensen-Shannon divergence of `old-five` more than doubles (0.0404 to 0.0969) while that of the one-hot
  `old-label` moves from 0.1972 to 0.2168. On SNLI this changes the order: with 5 annotations the plug-in
  Jensen-Shannon divergence puts `old-label` (0.1696) ahead of `old-prior` (0.1823), the opposite of what 100
  annotations show (0.1569 against 0.1346).
- The jackknife brings the 5-annotation values close to the 100-annotation ones (`old-five` on SNLI: total
  variation 0.2756 against 0.2778, Jensen-Shannon 0.0780 against 0.0711) and restores that order (0.1423 for
  `old-prior`, 0.1537 for `old-label`).
- It does not always work for total variation. In the alphaNLI table printed by the same command, the uniform
  predictor has total variation 0.3812 with 100 annotations and 0.3979 with 5, plug-in and jackknife alike:
  with two labels and a prediction of one half, leaving one annotator out never crosses the kink of the
  absolute value, so the jackknife sees nothing to correct.

The single commands give intervals as well. `old-prior` (A) against `old-label` (B) on SNLI, rows that are
`n/a` for a one-hot prediction left out:

```
softeval compare data/chaosnli/prepared/snli/human.jsonl data/chaosnli/prepared/snli/pred_old-prior.jsonl data/chaosnli/prepared/snli/pred_old-label.jsonl --labels e,n,c
```

| metric | A | B | A - B | 95% interval | share of resamples A better |
| --- | --- | --- | --- | --- | --- |
| Jensen-Shannon divergence (nats) | 0.1346 | 0.1569 | -0.0223 | [-0.0319, -0.0135] | 1.0000 |
| Jensen-Shannon distance | 0.3523 | 0.3538 | -0.0014 | [-0.0141, 0.0106] | 0.5890 |
| total variation distance | 0.4010 | 0.3430 | 0.0581 | [0.0394, 0.0758] | 0.0000 |
| match with a random annotator | 0.5025 | 0.6570 | -0.1545 | [-0.1767, -0.1327] | 0.0000 |
| accuracy against the majority | 0.5363 | 0.7503 | -0.2140 | [-0.2503, -0.1780] | 0.0000 |

The Jensen-Shannon divergence and the total variation distance disagree about these two predictors, each with
an interval that excludes zero, and the square root of the Jensen-Shannon divergence cannot separate them; a
report should say which distance it uses.

By agreement band (the share of the 100 annotators on the most common label):

```
softeval bands data/chaosnli/prepared/snli/human.jsonl data/chaosnli/prepared/snli/pred_old-five.jsonl --labels e,n,c
```

| band | items | cross-entropy | KL | JS div. | TV | match | majority acc. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [0, 0.5) | 52 | 1.1114 | 0.1393 | 0.0309 | 0.1763 | 0.4142 | 0.4712 |
| [0.5, 0.7) | 499 | 0.8710 | 0.1225 | 0.0345 | 0.1642 | 0.5131 | 0.6122 |
| [0.7, 0.9) | 637 | 0.7880 | 0.2492 | 0.0689 | 0.3031 | 0.6771 | 0.7928 |
| [0.9, 1] | 326 | 0.6799 | 0.4647 | 0.1376 | 0.4184 | 0.8769 | 0.9233 |
| all | 1514 | 0.8032 | 0.2501 | 0.0711 | 0.2778 | 0.6570 | 0.7503 |

Accuracy and match of `old-five` rise with agreement while its distances get worse: five annotations smoothed
by 0.5 put at most 5.5 / 6.5 = 0.85 on one label, less than 90 or more annotators out of 100 agreeing.

## Limitations

- The saved model predictions that ChaosNLI was published with were not available, so there is no comparison
  of trained models here; the predictors above are simple ones built from the public files.
- Annotators are treated as exchangeable draws from one population per item. Annotator identities are not
  used: an `annotator` column is read and ignored, and there is no model of individual annotators or of their
  reliability.
- Bootstrap intervals resample items only. They do not include the noise from having few annotators per item,
  which the corrections and ceilings address separately.
- The jackknife and split-half corrections assume a bias proportional to `1/n`. Total variation and the
  Jensen-Shannon distance are not smooth and are corrected only in part (see the alphaNLI case above). At 3 to
  5 annotations no correction gives accurate values. Corrected values can be negative, for single items and,
  when the model is within the annotation noise, for the mean.
- The fresh-sample ceiling draws new annotators from the observed distribution, treating it as the population
  distribution. With few annotations that distribution is itself noisy, so the ceiling is approximate there.
- Agreement bands are formed from the same annotations that are scored. With few annotations the extreme bands
  collect items that look unanimous or split by chance.
- Single-label classification only: no multi-label items, ordinal scales or continuous ratings, and no
  weighting of annotators.
- Cross-entropy and KL depend on the floor of 1e-12 applied to zero probabilities; a model that outputs exact
  zeros is penalised by about 27.6 nats times the human mass on that label.
- `--annotations M` chooses annotators independently per item; it does not simulate a smaller shared pool of
  annotators.

## Related tools

- [labelfuse](https://github.com/vipul21435/labelfuse) aggregates several annotators' labels into one; softeval
  is for when the labels should not be reduced to one.
- [calibcheck](https://github.com/vipul21435/calibcheck) measures a model's confidence against correctness on
  single-label data.
- [rubricops](https://github.com/vipul21435/rubricops) covers inter-rater agreement coefficients.
- [evalstats](https://github.com/vipul21435/evalstats) has more on error bars and paired comparisons for
  evaluations in general.
- The [ChaosNLI repository](https://github.com/easonnie/ChaosNLI) has its own scoreboard with JSD, KL and
  accuracy against the old and the new majority label.

## Development

```
uv run --extra dev ruff check .
uv run --extra dev pytest -q --cov=softeval
```

99 tests, offline, about 13 seconds on Python 3.12 (measured on a loaded laptop); line and branch coverage
99 %. The same suite passes on Python 3.10 and 3.13.

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io

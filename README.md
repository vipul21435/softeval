# softeval

Score a model's predictions against the distribution of labels that several annotators gave, instead of against
one majority label.

Status: early. What works so far is listed below; the rest of the plan is under "Not implemented yet".

## Why

On many items annotators disagree for good reasons. Reducing their labels to one majority label discards that,
and accuracy against the majority rewards a confident answer on an item that has no clear answer. softeval keeps
the label counts and compares the model's probabilities with them.

## Install

```
pip install git+https://github.com/vipul21435/softeval
```

Python 3.10 or newer; the only runtime dependency is numpy.

## What works so far

`softeval score HUMAN PREDICTIONS` reads annotations and predictions from JSONL or CSV and prints, as Markdown or
JSON (`--format json`):

- cross-entropy and KL divergence of the model against the human distribution (nats);
- Jensen-Shannon divergence and its square root, and total variation distance;
- the probability that the model's top answer equals the label of a randomly drawn annotator;
- accuracy against the majority label, with ties among annotators handled by a stated policy
  (`--ties split|any|exclude`) and counted in the output;
- the correlation between the model's entropy and the annotators' entropy across items (Pearson and Spearman).

Items with no annotations are skipped and counted. Predictions given as a label without probabilities are scored
too; cross-entropy, KL and the entropy correlations are not defined for them and print as `n/a`.

Example on the small made-up file in `examples/toy` (seven annotated items, one item with no annotations):

```
$ softeval score examples/toy/human.jsonl examples/toy/model_a.jsonl --labels entailment,neutral,contradiction
Items scored: 7 (no annotations: 1; tied majority: 2, scored as 'split')

| metric | value |
| --- | --- |
| cross-entropy (nats) | 0.6109 |
| KL divergence, human to model (nats) | 0.0577 |
| Jensen-Shannon divergence (nats) | 0.0193 |
| Jensen-Shannon distance | 0.1287 |
| total variation distance | 0.0786 |
| match with a random annotator | 0.7000 |
| accuracy against the majority | 0.8571 |
| entropy correlation (Pearson) | 0.9967 |
| entropy correlation (Spearman) | 0.9910 |
```

Input formats are described at the top of `src/softeval/io.py`.

## Not implemented yet

- corrections for the bias of these metrics when each item has only a few annotations;
- human ceilings (one annotator against the others, the full distribution against a fresh sample);
- bootstrap intervals and the paired comparison of two models;
- scores by human-agreement band, and the `ceiling`, `compare`, `bands` and `simulate` commands;
- a run on public data.

## Development

```
uv run --extra dev ruff check .
uv run --extra dev pytest -q
```

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io

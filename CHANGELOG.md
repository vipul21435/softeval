# Changelog

## 0.1.0 (2026-11-13)

First release.

- Metrics of predicted probabilities against per-item label counts: cross-entropy, KL divergence,
  Jensen-Shannon divergence and distance, total variation distance, match with a random annotator,
  accuracy against the majority with an explicit tie policy, and the correlation between model
  entropy and human entropy.
- Corrections for few annotations per item: leave-one-annotator-out jackknife, split-half
  extrapolation, Miller-Madow for KL, and an exact expectation of the plug-in metrics by
  enumeration for a handful of labels.
- Human ceilings: one annotator against the others, the full distribution against a fresh sample
  of annotators, and half against half.
- Percentile bootstrap intervals over items and a paired comparison of two models.
- Scores by human-agreement band (majority share or normalised entropy).
- A simulation that reports the bias of every estimate at a chosen number of annotations.
- Command line: `softeval score`, `ceiling`, `compare`, `bands`, `simulate`, each with Markdown
  or JSON output.
- Readers for JSONL and CSV: label counts, annotation lists or one row per annotation; predictions
  as a label or as probabilities.
- Example scripts that download ChaosNLI, convert it and print the tables in the README.

"""softeval: score predictions against the distribution of annotator labels."""

from softeval.bands import score_bands
from softeval.bootstrap import bootstrap_metrics, paired_bootstrap
from softeval.io import align, read_human, read_predictions
from softeval.metrics import item_metrics, summarise

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "align",
    "bootstrap_metrics",
    "item_metrics",
    "paired_bootstrap",
    "read_human",
    "read_predictions",
    "score_bands",
    "summarise",
]

"""Markdown and JSON rendering shared by the command line and the examples."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

METRIC_LABELS = {
    "cross_entropy": "cross-entropy (nats)",
    "kl": "KL divergence, human to model (nats)",
    "js_divergence": "Jensen-Shannon divergence (nats)",
    "js_distance": "Jensen-Shannon distance",
    "total_variation": "total variation distance",
    "annotator_match": "match with a random annotator",
    "majority_accuracy": "accuracy against the majority",
    "entropy_pearson": "entropy correlation (Pearson)",
    "entropy_spearman": "entropy correlation (Spearman)",
}


def fmt(value: Any, digits: int = 4) -> str:
    """Format a number for a table; missing or undefined values print as ``n/a``."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], digits: int = 4) -> str:
    """Render a GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell, digits) for cell in row) + " |")
    return "\n".join(lines)


def to_json(payload: Any) -> str:
    """Serialise a result with ``nan`` and infinities written as ``null``."""
    return json.dumps(_clean(payload), indent=2, sort_keys=False)


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and callable(value.item) and getattr(value, "ndim", 1) == 0:
        return _clean(value.item())
    if hasattr(value, "tolist") and callable(value.tolist):
        return _clean(value.tolist())
    return value


def interval(low: float | None, high: float | None, digits: int = 4) -> str:
    """Format a confidence interval as ``[low, high]``."""
    if low is None or high is None or math.isnan(low) or math.isnan(high):
        return "n/a"
    return f"[{fmt(float(low), digits)}, {fmt(float(high), digits)}]"

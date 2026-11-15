"""Deterministic review allocation and transparent confusion-count metrics."""

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


def review_count(n, capacity_units):
    """One unit = 0.1%. Integer ceiling avoids an accidental extra review."""
    if not isinstance(capacity_units, (int, np.integer)) or not 0 <= capacity_units <= 1000:
        raise ValueError("Capacity must be an integer from 0 through 1000 tenths of a percent.")
    if n < 0:
        raise ValueError("Batch size cannot be negative.")
    return min(n, (n * int(capacity_units) + 999) // 1000)


def rank_order(scores, ids):
    scores, ids = np.asarray(scores), np.asarray(ids)
    if scores.ndim != 1 or ids.ndim != 1:
        raise ValueError("Scores and transaction IDs must be one-dimensional vectors.")
    if len(scores) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Scores require an equal-length vector of unique transaction IDs.")
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be finite.")
    return np.lexsort((ids, -scores))


def selection(scores, ids, capacity_units):
    chosen = np.zeros(len(scores), dtype=bool)
    chosen[rank_order(scores, ids)[:review_count(len(scores), capacity_units)]] = True
    return chosen


def counts(labels, selected):
    labels, selected = np.asarray(labels), np.asarray(selected, dtype=bool)
    if labels.ndim != 1 or selected.ndim != 1:
        raise ValueError("Labels and selections must be one-dimensional vectors.")
    if len(labels) != len(selected) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Metrics require aligned binary labels and selections.")
    fraud = labels == 1
    tp, fp = int((fraud & selected).sum()), int((~fraud & selected).sum())
    fn, tn = int((fraud & ~selected).sum()), int((~fraud & ~selected).sum())
    return {
        "n": len(labels), "k": tp + fp, "fraud": tp + fn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def ranking_metrics(labels, scores):
    if not np.any(np.asarray(labels) == 1):
        return {"average_precision": None, "pr_auc_trapezoidal": None}
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "pr_auc_trapezoidal": float(auc(recall, precision)),
    }


def capacity_results(labels, scores, ids, capacities=range(1, 51)):
    return [
        {"capacity_units": int(c), **counts(labels, selection(scores, ids, c))}
        for c in capacities
    ]


def temporal_results(labels, times, selected, edges):
    """Attribute the original global selections; never reallocate per window."""
    times, labels, selected = np.asarray(times), np.asarray(labels), np.asarray(selected)
    if len(edges) != 4 or np.any(np.diff(edges) <= 0):
        raise ValueError("Temporal analysis needs three positive-duration windows.")
    if len(times) != len(labels) or len(times) != len(selected):
        raise ValueError("Temporal arrays must align.")
    if len(times) and (times.min() < edges[0] or times.max() > edges[-1]):
        raise ValueError("Temporal windows do not cover the batch.")
    windows = np.searchsorted(np.asarray(edges)[1:-1], times, side="right")
    return [
        {"window": i + 1, "start": float(edges[i]), "end": float(edges[i + 1]),
         **counts(labels[windows == i], selected[windows == i])}
        for i in range(3)
    ]


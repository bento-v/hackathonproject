"""Evaluate a frozen run; never train or select a model using test outcomes."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reviewwise.artifacts import load_manifest, write_json
from reviewwise.config import FEATURES, MODEL_NAMES, POLICY, ROOT
from reviewwise.data import describe_splits, fingerprint, load_data, split_data
from reviewwise.ranking import (
    capacity_results, counts, rank_order, ranking_metrics, review_count,
    selection, temporal_results,
)


def failure_examples(predictions, labels, selected_model):
    ids = predictions["transaction_id"].to_numpy()
    order = rank_order(predictions[selected_model], ids)
    amount_order = rank_order(predictions["amount"], ids)
    ranks = np.empty(len(ids), dtype=int)
    amount_ranks = np.empty(len(ids), dtype=int)
    ranks[order] = np.arange(1, len(ids) + 1)
    amount_ranks[amount_order] = np.arange(1, len(ids) + 1)
    chosen = selection(predictions[selected_model], ids, 10)
    examples = {}
    for category, eligible in (
        ("false_positive", (labels == 0) & chosen),
        ("missed_fraud", (labels == 1) & ~chosen),
    ):
        candidates = order[np.asarray(eligible)[order]]
        if not len(candidates):
            examples[category] = None
            continue
        position = candidates[0]
        row = predictions.iloc[position]
        examples[category] = {
            "transaction_id": int(ids[position]), "amount": float(row["Amount"]),
            "elapsed_seconds": float(row["Time"]), "model_score": float(row[selected_model]),
            "model_rank": int(ranks[position]), "amount_rank": int(amount_ranks[position]),
            "minimum_reviews": int(ranks[position]),
            "included_at_2_percent": bool(ranks[position] <= review_count(len(ids), 20)),
            "class": int(labels[position]),
        }
    return examples


def evaluate(data_path, directory, results_dir):
    directory, results_dir = Path(directory), Path(results_dir)
    manifest = load_manifest(directory, data_path)
    splits = split_data(load_data(data_path))
    if describe_splits(splits) != manifest["audit"]["splits"]:
        raise ValueError("Split membership or data summary differs from the frozen training run.")
    test = splits["test"].reset_index(drop=True)
    predictions = test[["transaction_id", "Time", "Amount"]].copy()
    for name in MODEL_NAMES:
        # Files are generated locally and verified against the training manifest.
        model = joblib.load(directory / f"{name}.joblib")
        if list(model.feature_names_in_) != FEATURES:
            raise ValueError(f"{name} has an incompatible feature list.")
        predictions[name] = model.predict_proba(test[FEATURES])[:, 1]
    predictions["amount"] = predictions["Amount"]
    labels = test["Class"].to_numpy()
    ids = predictions["transaction_id"].to_numpy()
    curves, temporal, metrics = [], [], []
    baseline = {r["capacity_units"]: r for r in capacity_results(labels, predictions["amount"], ids)}
    for name in (*MODEL_NAMES, "amount"):
        rows = capacity_results(labels, predictions[name], ids)
        overall = ranking_metrics(labels, predictions[name])
        for row in rows:
            row.update({
                "method": name,
                "additional_fraud_vs_amount": row["tp"] - baseline[row["capacity_units"]]["tp"],
                "recall_difference_vs_amount": row["recall"] - baseline[row["capacity_units"]]["recall"],
            })
            curves.append(row)
            if row["capacity_units"] in POLICY["demo_capacity_units"]:
                metrics.append({**row, **overall})
        chosen = selection(predictions[name], ids, 10)
        temporal.extend({"method": name, **row} for row in temporal_results(
            labels, test["Time"], chosen, manifest["temporal_edges"]))
    result = {
        "schema_version": 1, "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": manifest["input_sha256"],
        "manifest_sha256": fingerprint(directory / "manifest.json"),
        "selected_model": manifest["selected_model"], "batch": manifest["audit"]["splits"]["test"],
        "metrics": metrics, "flag_nothing": counts(labels, np.zeros(len(labels), dtype=bool)),
        "temporal_edges": manifest["temporal_edges"],
        "temporal_policy": POLICY["temporal_policy"],
    }
    if fingerprint(data_path) != manifest["input_sha256"]:
        raise ValueError("Dataset changed during evaluation; refusing to save results.")
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    test[["transaction_id", "Class"]].to_csv(directory / "labels.csv", index=False)
    write_json(directory / "failure_examples.json", failure_examples(predictions, labels, manifest["selected_model"]))
    write_json(results_dir / "metrics.json", result)
    pd.DataFrame(curves).to_csv(results_dir / "capacity.csv", index=False)
    pd.DataFrame(temporal).to_csv(results_dir / "temporal.csv", index=False)
    write_json(directory / "evaluation_index.json", {
        "manifest_sha256": result["manifest_sha256"],
        "local_files": {name: fingerprint(directory / name) for name in
                        ("predictions.csv", "labels.csv", "failure_examples.json")},
        "result_files": {name: fingerprint(results_dir / name) for name in
                         ("metrics.json", "capacity.csv", "temporal.csv")},
    })
    print(f"Frozen selected model remains: {manifest['selected_model']}")
    print(pd.DataFrame([row for row in metrics if row["capacity_units"] == 10])[
        ["method", "k", "tp", "fp", "fn", "precision", "recall", "average_precision", "pr_auc_trapezoidal"]
    ].to_string(index=False))
    print("\nGlobal 1% queue, attributed by time window:")
    print(pd.DataFrame(temporal).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/creditcard.csv")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    try:
        evaluate(args.data, args.artifacts, args.results)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"Evaluation stopped: {error}\n")


if __name__ == "__main__":
    main()

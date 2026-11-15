"""Validate the source and form chronological, timestamp-preserving splits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reviewwise.config import FEATURES, REQUIRED, ROOT


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(frame):
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The CSV contains no transactions.")
    data = frame[REQUIRED].copy()
    for column in REQUIRED:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Required columns contain missing or infinite values; fix the input.")
    if not data["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 (legitimate) and 1 (fraud).")
    if (data[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Time and Amount must be nonnegative.")
    data["Class"] = data["Class"].astype(int)
    data.insert(0, "transaction_id", np.arange(1, len(data) + 1))
    return data.sort_values(["Time", "transaction_id"], kind="stable").reset_index(drop=True)


def load_data(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Dataset not found: {path}. Download creditcard.csv from the cited Kaggle source.")
    return validate(pd.read_csv(path))


def split_data(data):
    """Choose the nearest interior timestamp-group boundary; ties go earlier."""
    times = data["Time"].to_numpy()
    boundaries = np.flatnonzero(times[:-1] != times[1:]) + 1
    if len(boundaries) < 2:
        raise ValueError("Need at least three distinct timestamps to create three splits.")
    # Integer distances avoid rounding ambiguity at the 60% and 80% targets.
    cuts = [int(boundaries[np.argmin(abs(boundaries * 5 - len(data) * numerator))])
            for numerator in (3, 4)]
    if cuts[0] >= cuts[1]:
        raise ValueError("Nearest timestamp boundaries do not form three nonempty splits.")
    splits = {
        "train": data.iloc[:cuts[0]].copy(),
        "validation": data.iloc[cuts[0]:cuts[1]].copy(),
        "test": data.iloc[cuts[1]:].copy(),
    }
    for name, rows in splits.items():
        if set(rows["Class"]) != {0, 1}:
            raise ValueError(f"The fixed {name} split does not contain both classes; cannot evaluate this design.")
    return splits


def describe_splits(splits):
    return {
        name: {
            "rows": len(rows), "fraud": int(rows["Class"].sum()),
            "legitimate": int((rows["Class"] == 0).sum()),
            "fraud_prevalence": float(rows["Class"].mean()),
            "time_min": float(rows["Time"].min()),
            "time_max": float(rows["Time"].max()),
            "membership_sha256": hashlib.sha256(
                rows["transaction_id"].to_numpy(dtype="<i8").tobytes()).hexdigest(),
        }
        for name, rows in splits.items()
    }


def temporal_edges(test):
    if test["Time"].nunique() < 2:
        raise ValueError("Test split needs at least two distinct timestamps for temporal analysis.")
    return np.linspace(test["Time"].min(), test["Time"].max(), 4).tolist()


def audit_data(data, splits):
    # Exact merges on distinct vectors, not approximate hash matches.
    unique = {name: rows[FEATURES].drop_duplicates() for name, rows in splits.items()}
    overlap = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap[f"{left}__{right}"] = len(unique[left].merge(unique[right], on=FEATURES))
    return {
        "rows": len(data), "fraud": int(data["Class"].sum()),
        "exact_duplicate_rows_beyond_first": int(data.duplicated(REQUIRED).sum()),
        "identical_feature_vectors_across_splits": overlap,
        "splits": describe_splits(splits),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/creditcard.csv")
    args = parser.parse_args()
    data = load_data(args.data)
    print(json.dumps({"input_sha256": fingerprint(args.data), **audit_data(data, split_data(data))}, indent=2))


if __name__ == "__main__":
    main()


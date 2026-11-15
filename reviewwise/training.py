"""Train on early transactions and select using validation only."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import warnings

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reviewwise.artifacts import read_json, versions, write_json
from reviewwise.config import FEATURES, MODEL_PARAMETERS, POLICY, ROOT, validate_model_parameters
from reviewwise.data import audit_data, fingerprint, load_data, split_data, temporal_edges
from reviewwise.ranking import counts, selection


def build_models(parameters=None):
    parameters = validate_model_parameters(MODEL_PARAMETERS if parameters is None else parameters)
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(**parameters["logistic_regression"])),
        "random_forest": RandomForestClassifier(**parameters["random_forest"]),
    }


def choose_model(validation):
    # Deliberately use only the predeclared metric and tie-breaker.
    return ("random_forest" if validation["random_forest"]["recall"] >
            validation["logistic_regression"]["recall"] else "logistic_regression")


def train(data_path, directory, results_dir, parameters=None):
    parameters = validate_model_parameters(MODEL_PARAMETERS if parameters is None else parameters)
    directory, results_dir = Path(directory), Path(results_dir)
    if (directory / "manifest.json").exists():
        raise ValueError("A frozen training manifest already exists. Use a new --artifacts directory for a documented new run.")
    directory.mkdir(parents=True, exist_ok=True)
    print("Validating data and chronological splits...", flush=True)
    before_hash = fingerprint(data_path)
    data = load_data(data_path)
    splits = split_data(data)
    edges = temporal_edges(splits["test"])
    audit = audit_data(data, splits)
    training, validation = splits["train"], splits["validation"]
    models, validation_results, hashes = build_models(parameters), {}, {}
    for name, model in models.items():
        print(f"Fitting {name} on {len(training):,} training rows...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(training[FEATURES], training["Class"])
        scores = model.predict_proba(validation[FEATURES])[:, 1]
        validation_results[name] = counts(validation["Class"], selection(scores, validation["transaction_id"], 10))
        path = directory / f"{name}.joblib"
        joblib.dump(model, path)
        hashes[name] = fingerprint(path)
        print(f"Validation at 1%: {validation_results[name]}", flush=True)
    winner = choose_model(validation_results)
    if fingerprint(data_path) != before_hash:
        raise ValueError("Dataset changed during training; no manifest was frozen.")
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_kind": "baseline" if parameters == MODEL_PARAMETERS else "parameter_experiment",
        "input_sha256": before_hash, "features": FEATURES, "versions": versions(),
        "model_parameters": parameters, "policy": POLICY,
        "selected_model": winner, "validation": validation_results,
        "audit": audit, "temporal_edges": edges,
        "model_files": hashes,
    }
    # Membership is recorded locally, not published as a row-level export.
    import pandas as pd
    pd.concat([rows[["transaction_id"]].assign(split=name) for name, rows in splits.items()]).to_csv(
        directory / "split_membership.csv", index=False)
    write_json(directory / "manifest.json", manifest)
    write_json(results_dir / "data_audit.json", audit)
    write_json(results_dir / "validation.json", {
        "selected_model": winner, "rule": POLICY["selection"], "results": validation_results,
        "input_sha256": before_hash,
    })
    print(f"Frozen selected model: {winner}. No test performance has been evaluated.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/creditcard.csv")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--tuning-report", type=Path,
                        help="Use model settings from a validation-only search report.")
    args = parser.parse_args()
    try:
        parameters = None
        if args.tuning_report:
            report = read_json(args.tuning_report)
            if report.get("input_sha256") != fingerprint(args.data):
                raise ValueError("Tuning report belongs to a different dataset.")
            parameters = validate_model_parameters(report.get("model_parameters"))
        train(args.data, args.artifacts, args.results, parameters=parameters)
    except (ValueError, FileNotFoundError, ConvergenceWarning) as error:
        parser.exit(1, f"Training stopped: {error}\n")


if __name__ == "__main__":
    main()


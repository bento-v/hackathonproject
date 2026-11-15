"""Search a small declared grid using validation only; never evaluate test scores."""

import argparse
from pathlib import Path
import warnings

from sklearn.exceptions import ConvergenceWarning

from reviewwise.artifacts import write_json
from reviewwise.config import FEATURES, MODEL_NAMES, ROOT, experiment_parameters
from reviewwise.data import fingerprint, load_data, split_data
from reviewwise.ranking import counts, ranking_metrics, selection
from reviewwise.training import build_models, choose_model


def tune(data_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Search output already exists; choose a new output path.")
    before = fingerprint(data_path)
    splits = split_data(load_data(data_path))
    training, validation = splits["train"], splits["validation"]
    # Order fixes ties in advance: baseline first, then the declared alternatives.
    grid = {
        "logistic_regression": [experiment_parameters(c) for c in (0, -2, -1, 1, 2)],
        "random_forest": [experiment_parameters(),
                          experiment_parameters(0, 200, 8, 5),
                          experiment_parameters(0, 200, 16, 5),
                          experiment_parameters(0, 200, 12, 1)],
    }
    rows, best = [], {}
    for name in MODEL_NAMES:
        for parameters in grid[name]:
            model = build_models(parameters)[name]
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(training[FEATURES], training["Class"])
            scores = model.predict_proba(validation[FEATURES])[:, 1]
            metrics = counts(validation["Class"], selection(
                scores, validation["transaction_id"], 10))
            row = {"model": name, "parameters": parameters[name],
                   **metrics, **ranking_metrics(validation["Class"], scores)}
            rows.append(row)
            if name not in best or row["recall"] > best[name]["recall"]:
                best[name] = row
            print(f"{name}: validation caught {row['tp']}/{row['fraud']} in {row['k']} reviews", flush=True)
    if fingerprint(data_path) != before:
        raise ValueError("Dataset changed during search; refusing to save results.")
    report = {
        "input_sha256": before,
        "selection_rule": "validation recall at 1%; within-family ties keep first candidate; family ties favor logistic regression",
        "selected_model": choose_model(best),
        "model_parameters": {name: best[name]["parameters"] for name in MODEL_NAMES},
        "candidates": rows,
        "test_evaluated": False,
        "limitation": "Exploratory search on a previously used validation set; confirm on new untouched data.",
    }
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/creditcard.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/tuning.json")
    args = parser.parse_args()
    try:
        tune(args.data, args.output)
    except (ValueError, OSError, ConvergenceWarning) as error:
        parser.exit(1, f"Tuning stopped: {error}\n")


if __name__ == "__main__":
    main()

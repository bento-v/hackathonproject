"""Choices fixed before test evaluation. Capacity units are tenths of a percent."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
REQUIRED = ["Time", *FEATURES, "Class"]
SEED = 42
MODEL_NAMES = ("logistic_regression", "random_forest")
DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "amount": "Largest payments",
}
MODEL_PARAMETERS = {
    "logistic_regression": {
        "class_weight": "balanced", "C": 1.0, "solver": "lbfgs",
        "max_iter": 2000, "random_state": SEED,
    },
    "random_forest": {
        "n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5,
        "class_weight": "balanced_subsample", "n_jobs": -1,
        "random_state": SEED,
    },
}
POLICY = {
    "primary_capacity_units": 10,
    "demo_capacity_units": [5, 10, 20],
    "curve_capacity_units": list(range(1, 51)),
    "capacity_denominator": 1000,
    "tie_break": "transaction_id ascending",
    "selection": "validation recall at 1%; tie: logistic_regression",
    "temporal_windows": 3,
    "temporal_policy": "break down global top 1%; do not rerank within windows",
}

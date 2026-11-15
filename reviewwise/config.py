"""Choices fixed before test evaluation. Capacity units are tenths of a percent."""

from copy import deepcopy
import math
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


def experiment_parameters(log10_c=0.0, trees=200, depth=12, leaf=5):
    """Bounded UI controls; preserve all other baseline settings."""
    if (isinstance(log10_c, bool) or not isinstance(log10_c, (int, float))
            or not math.isfinite(log10_c) or not -3 <= log10_c <= 2):
        raise ValueError("Log10 C must be between -3 and 2.")
    bounds = (("trees", trees, 50, 400), ("depth", depth, 3, 20), ("leaf", leaf, 1, 20))
    for name, value, lower, upper in bounds:
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer from {lower} through {upper}.")
    parameters = deepcopy(MODEL_PARAMETERS)
    parameters["logistic_regression"]["C"] = 10.0 ** log10_c
    parameters["random_forest"].update(n_estimators=trees, max_depth=depth, min_samples_leaf=leaf)
    return parameters


def validate_model_parameters(parameters):
    """Accept baseline artifacts and only the documented experiment controls."""
    try:
        logistic, forest = parameters["logistic_regression"], parameters["random_forest"]
        c = logistic["C"]
        if (isinstance(c, bool) or not isinstance(c, (int, float))
                or not math.isfinite(c) or not 0.001 <= c <= 100):
            raise ValueError("Invalid C.")
        expected = experiment_parameters(0, forest["n_estimators"], forest["max_depth"],
                                         forest["min_samples_leaf"])
        expected["logistic_regression"]["C"] = c
        if parameters != expected:
            raise ValueError("Unsupported model settings.")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Configuration contains unsupported model parameters.") from error
    return deepcopy(parameters)

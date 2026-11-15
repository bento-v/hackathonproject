"""Isolated local training runs for the interactive experiment controls."""

from pathlib import Path
from uuid import uuid4

from reviewwise.config import validate_model_parameters
from reviewwise.training import train


def run_experiment(data_path, runs_directory, parameters):
    parameters = validate_model_parameters(parameters)
    run_directory = Path(runs_directory) / uuid4().hex
    artifacts, results = run_directory / "artifacts", run_directory / "results"
    train(data_path, artifacts, results, parameters=parameters)
    return artifacts, results


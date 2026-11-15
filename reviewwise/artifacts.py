"""Small, explicit artifact contracts; only load models created locally."""

import importlib.metadata
import json
from pathlib import Path

from reviewwise.config import FEATURES, MODEL_NAMES, POLICY, validate_model_parameters
from reviewwise.data import fingerprint


def versions():
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "streamlit", "plotly", "joblib", "pyarrow")
    return {name: importlib.metadata.version(name) for name in packages}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing artifact: {path}. Run training, then evaluation as described in README.md.")
    return json.loads(path.read_text(encoding="utf-8"))


def check_file(path, expected_hash):
    if not Path(path).is_file() or fingerprint(path) != expected_hash:
        raise ValueError(f"Missing or changed artifact: {path}. Reproduce the matching training/evaluation run.")


def load_manifest(directory, data_path):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("features") != FEATURES:
        raise ValueError("Artifact schema or feature list is incompatible with this code.")
    validate_model_parameters(manifest.get("model_parameters"))
    if manifest.get("policy") != POLICY:
        raise ValueError("Configuration changed since training. Do not reuse stale evaluation artifacts.")
    if manifest.get("selected_model") not in MODEL_NAMES:
        raise ValueError("Manifest has an invalid selected model.")
    if not Path(data_path).is_file() or fingerprint(data_path) != manifest.get("input_sha256"):
        raise ValueError("Dataset fingerprint differs from the frozen training run (or the CSV is missing).")
    current = versions()
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib"):
        if current[name] != manifest["versions"][name]:
            raise ValueError(f"{name} version differs from training; use the recorded requirements.")
    if set(manifest.get("model_files", {})) != set(MODEL_NAMES):
        raise ValueError("Manifest does not contain both frozen model files.")
    for name, digest in manifest["model_files"].items():
        check_file(directory / f"{name}.joblib", digest)
    return manifest


def load_evaluation(directory, results_dir, data_path):
    directory, results_dir = Path(directory), Path(results_dir)
    manifest = load_manifest(directory, data_path)
    index = read_json(directory / "evaluation_index.json")
    if index.get("manifest_sha256") != fingerprint(directory / "manifest.json"):
        raise ValueError("Evaluation belongs to another training run. Evaluate the frozen models again.")
    for filename in ("predictions.csv", "labels.csv", "failure_examples.json"):
        check_file(directory / filename, index["local_files"][filename])
    for filename in ("metrics.json", "capacity.csv", "temporal.csv"):
        check_file(results_dir / filename, index["result_files"][filename])
    return manifest, read_json(results_dir / "metrics.json")

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from reviewwise.artifacts import load_evaluation
from reviewwise.config import FEATURES, MODEL_PARAMETERS, ROOT, experiment_parameters, validate_model_parameters
from reviewwise.data import fingerprint
from reviewwise.evaluation import evaluate
from reviewwise.experiments import run_experiment
from reviewwise.training import train


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data/creditcard.csv"
        self.data.parent.mkdir()
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(500, len(FEATURES))), columns=FEATURES)
        frame["Time"] = np.arange(500)
        frame["Amount"] = np.abs(frame.Amount) * 100
        frame["Class"] = (np.arange(500) % 10 == 0).astype(int)
        frame["V1"] += frame.Class * 3
        frame.to_csv(self.data, index=False)

    def test_bounded_controls_preserve_defaults_and_reject_other_settings(self):
        self.assertEqual(experiment_parameters(), MODEL_PARAMETERS)
        custom = experiment_parameters(-1, 50, 4, 2)
        self.assertEqual(custom["logistic_regression"]["C"], 0.1)
        self.assertEqual(MODEL_PARAMETERS["random_forest"]["n_estimators"], 200)
        for settings in ((-4, 200, 12, 5), (0, 0, 12, 5), (0, 50, True, 5), (0, 50, 4, 0)):
            with self.assertRaises(ValueError):
                experiment_parameters(*settings)
        custom["logistic_regression"]["class_weight"] = None
        with self.assertRaises(ValueError):
            validate_model_parameters(custom)

    def test_custom_run_round_trips_without_overwriting_other_runs(self):
        parameters = experiment_parameters(-1, 50, 4, 2)
        with contextlib.redirect_stdout(io.StringIO()):
            first = run_experiment(self.data, self.root / "runs", parameters)
            before = fingerprint(first[0] / "manifest.json")
            second = run_experiment(self.data, self.root / "runs", parameters)
        self.assertNotEqual(first, second)
        self.assertEqual(fingerprint(first[0] / "manifest.json"), before)
        manifest, evidence = load_evaluation(*second, self.data)
        self.assertEqual(manifest["model_parameters"], parameters)
        model = joblib.load(second[0] / "logistic_regression.joblib")
        self.assertEqual(model[-1].C, 0.1)
        self.assertEqual(evidence["selected_model"], manifest["selected_model"])
        self.assertNotIn("Class", pd.read_csv(second[0] / "predictions.csv"))

    def test_web_retraining_activation_failure_and_restore(self):
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.data, self.root / "artifacts", self.root / "results")
            evaluate(self.data, self.root / "artifacts", self.root / "results")
        original_hash = fingerprint(self.root / "artifacts/manifest.json")

        def button(app, label):
            return next(item for item in app.button if item.label == label)

        with patch("reviewwise.config.ROOT", self.root), patch(
            "reviewwise.experiments.run_experiment", wraps=run_experiment
        ) as runner, contextlib.redirect_stdout(io.StringIO()):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            self.assertEqual(len(app.exception), 0)
            button(app, "Reveal historical outcomes").click().run()
            self.assertTrue(app.session_state["revealed"])
            app.slider(key="experiment_log10_c").set_value(-1.0)
            app.slider(key="experiment_trees").set_value(50)
            self.assertEqual(runner.call_count, 0)
            button(app, "Retrain models").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            self.assertEqual(runner.call_count, 1)
            self.assertFalse(app.session_state["revealed"])
            active = app.session_state["experiment_paths"]
            manifest, _ = load_evaluation(*active, self.data)
            self.assertEqual(manifest["model_parameters"]["logistic_regression"]["C"], 0.1)
            self.assertEqual(manifest["model_parameters"]["random_forest"]["n_estimators"], 50)
            app.radio[0].set_value("2%").run()
            self.assertEqual(runner.call_count, 1)
            runner.side_effect = ValueError("Synthetic training failure")
            button(app, "Retrain models").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("Synthetic training failure", app.error[0].value)
            self.assertEqual(app.session_state["experiment_paths"], active)
            button(app, "Return to original saved run").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertNotIn("experiment_paths", app.session_state)
        self.assertEqual(fingerprint(self.root / "artifacts/manifest.json"), original_hash)


if __name__ == "__main__":
    unittest.main()

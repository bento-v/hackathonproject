"""Checks against local frozen artifacts; skipped in a clean, untrained checkout."""

import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from reviewwise.artifacts import load_evaluation, read_json
from reviewwise.config import ROOT
from reviewwise.evaluation import failure_examples
from reviewwise.ranking import counts, review_count, selection


@unittest.skipUnless((ROOT / "artifacts/evaluation_index.json").exists(), "Train/evaluate to run local integration checks")
class FrozenRunTests(unittest.TestCase):
    def test_saved_metrics_reconcile_with_predictions(self):
        manifest, evidence = load_evaluation(ROOT / "artifacts", ROOT / "results", ROOT / "data/creditcard.csv")
        predictions = pd.read_csv(ROOT / "artifacts/predictions.csv")
        labels = pd.read_csv(ROOT / "artifacts/labels.csv")
        self.assertNotIn("Class", predictions)
        np.testing.assert_array_equal(predictions.transaction_id, labels.transaction_id)
        for metric in evidence["metrics"]:
            actual = counts(labels.Class, selection(predictions[metric["method"]], predictions.transaction_id,
                                                    metric["capacity_units"]))
            for key, value in actual.items():
                self.assertEqual(metric[key], value)
            self.assertEqual(metric["n"], metric["tp"] + metric["fp"] + metric["fn"] + metric["tn"])
        temporal = pd.read_csv(ROOT / "results/temporal.csv")
        for method, windows in temporal.groupby("method"):
            aggregate = next(row for row in evidence["metrics"] if row["method"] == method and row["capacity_units"] == 10)
            for key in ("n", "k", "fraud", "tp", "fp", "fn", "tn"):
                self.assertEqual(int(windows[key].sum()), aggregate[key])
        self.assertEqual(failure_examples(predictions, labels.Class.to_numpy(), manifest["selected_model"]),
                         read_json(ROOT / "artifacts/failure_examples.json"))

    def test_reveal_reset_and_budget_counts(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        evidence = read_json(ROOT / "results/metrics.json")
        for label, capacity in (("1%", 10), ("0.5%", 5), ("2%", 20)):
            app.radio[0].set_value(label).run()
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(app.session_state["revealed"])
            self.assertEqual(len(app.dataframe), 2)
            for frame in app.dataframe:
                self.assertNotIn("Historical outcome", frame.value)
                self.assertEqual(len(frame.value), review_count(evidence["batch"]["rows"], capacity))
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.session_state["revealed"])
            self.assertIn("Historical outcome", app.dataframe[0].value)
            model_metric = next(row for row in evidence["metrics"] if row["method"] == evidence["selected_model"]
                                and row["capacity_units"] == capacity)
            self.assertEqual(int((app.dataframe[0].value["Historical outcome"] == "Fraud").sum()), model_metric["tp"])


if __name__ == "__main__":
    unittest.main()

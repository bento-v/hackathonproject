import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from reviewwise.config import FEATURES, validate_model_parameters
from reviewwise.data import temporal_edges
from reviewwise.ranking import counts, rank_order
from reviewwise.tuning import tune


class ReviewFixTests(unittest.TestCase):
    def test_column_vectors_cannot_broadcast_confusion_counts(self):
        with self.assertRaises(ValueError):
            counts([[0], [1]], [True, False])
        with self.assertRaises(ValueError):
            rank_order([[0.2], [0.8]], [1, 2])

    def test_single_timestamp_test_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, "two distinct timestamps"):
            temporal_edges(pd.DataFrame({"Time": [4, 4]}))

    def test_search_is_invariant_to_test_features_and_labels(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(500, len(FEATURES))), columns=FEATURES)
        frame["Amount"] = frame.Amount.abs()
        frame["Time"] = np.arange(500)
        frame["Class"] = (np.arange(500) % 10 == 0).astype(int)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "data.csv"
            frame.to_csv(source, index=False)
            # Test rows must never reach fit or predict_proba during search.
            from reviewwise.training import build_models
            def guarded_models(parameters):
                models = build_models(parameters)
                for model in models.values():
                    original = model.predict_proba
                    def predict(rows, original=original):
                        self.assertEqual(len(rows), 100)
                        self.assertLess(rows.index.max(), 400)
                        return original(rows)
                    model.predict_proba = predict
                return models
            with patch("reviewwise.tuning.build_models", side_effect=guarded_models):
                first = tune(source, Path(tmp) / "first.json")
                frame.loc[400:, FEATURES] = 1000
                frame.loc[400:, "Class"] = 1 - frame.loc[400:, "Class"]
                frame.to_csv(source, index=False)
                second = tune(source, Path(tmp) / "second.json")
            self.assertEqual(first["candidates"], second["candidates"])
            self.assertEqual(first["model_parameters"], second["model_parameters"])
            validate_model_parameters(first["model_parameters"])
            self.assertFalse(first["test_evaluated"])


if __name__ == "__main__":
    unittest.main()

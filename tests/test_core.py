import unittest

import numpy as np
import pandas as pd

from reviewwise.config import FEATURES, REQUIRED
from reviewwise.data import split_data, validate
from reviewwise.ranking import counts, rank_order, review_count, selection, temporal_results
from reviewwise.training import build_models, choose_model


def small_data():
    frame = pd.DataFrame({column: np.zeros(20) for column in REQUIRED})
    frame["Time"] = np.repeat(np.arange(10), 2)
    frame["Class"] = np.tile([0, 1], 10)
    frame["Amount"] = np.arange(20)
    return validate(frame)


class DataTests(unittest.TestCase):
    def test_timestamp_groups_and_disjoint_membership(self):
        splits = split_data(small_data())
        self.assertEqual([len(x) for x in splits.values()], [12, 4, 4])
        self.assertLess(splits["train"].Time.max(), splits["validation"].Time.min())
        self.assertLess(splits["validation"].Time.max(), splits["test"].Time.min())
        ids = [set(rows.transaction_id) for rows in splits.values()]
        self.assertFalse(ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])

    def test_equal_distance_uses_earlier_boundary(self):
        frame = small_data().iloc[:10].copy()
        frame["Time"] = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
        # 60% falls between boundaries 5 and 7; earlier boundary wins.
        self.assertEqual(len(split_data(frame)["train"]), 5)

    def test_invalid_data_rejected(self):
        for column, value in (("Amount", -1), ("Class", 2), ("V1", np.nan)):
            with self.subTest(column=column):
                frame = small_data()
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    validate(frame)

    def test_features_and_train_only_scaling(self):
        splits = split_data(small_data())
        model = build_models()["logistic_regression"]
        model.fit(splits["train"][FEATURES], splits["train"].Class)
        self.assertFalse({"Class", "Time", "transaction_id"} & set(model.feature_names_in_))
        np.testing.assert_allclose(model[0].mean_, splits["train"][FEATURES].mean())
        self.assertNotEqual(model[0].mean_[-1], small_data().Amount.mean())


class RankingTests(unittest.TestCase):
    def test_budget_and_deterministic_ties(self):
        self.assertEqual(review_count(100, 10), 1)
        self.assertEqual(review_count(101, 10), 2)
        self.assertEqual(review_count(10000, 7), 70)
        np.testing.assert_array_equal(rank_order([0.5, 0.5, 0.9], [7, 2, 9]), [2, 1, 0])
        np.testing.assert_array_equal(selection([0.5, 0.5, 0.9], [7, 2, 9], 500), [False, True, True])

    def test_hand_calculated_counts(self):
        result = counts([1, 0, 1, 0, 0], [True, True, False, False, False])
        self.assertEqual(result, {"n": 5, "k": 2, "fraud": 2, "tp": 1, "fp": 1,
                                  "fn": 1, "tn": 2, "precision": 0.5, "recall": 0.5})
        self.assertIsNone(counts([1, 0], [False, False])["precision"])
        self.assertIsNone(counts([0, 0], [True, False])["recall"])

    def test_windows_preserve_global_queue_and_endpoints(self):
        rows = temporal_results([1, 0, 0, 1], [0, 1, 2, 3], [True, True, False, False], [0, 1, 2, 3])
        self.assertEqual([r["n"] for r in rows], [1, 1, 2])
        self.assertEqual([r["k"] for r in rows], [1, 1, 0])
        self.assertIsNone(rows[1]["recall"])
        self.assertEqual(sum(r["tp"] for r in rows), 1)

    def test_nested_budgets(self):
        ids = np.arange(1, 1001)
        scores = np.random.default_rng(42).random(1000)
        previous = set()
        for capacity in range(1, 51):
            current = set(ids[selection(scores, ids, capacity)])
            self.assertTrue(previous <= current)
            self.assertEqual(len(current), review_count(1000, capacity))
            previous = current

    def test_selection_rule_has_no_test_input(self):
        validation = {"logistic_regression": {"recall": 0.8}, "random_forest": {"recall": 0.8}}
        self.assertEqual(choose_model(validation), "logistic_regression")
        validation["random_forest"]["recall"] = 0.9
        self.assertEqual(choose_model(validation), "random_forest")


if __name__ == "__main__":
    unittest.main()

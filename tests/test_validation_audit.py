import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from reviewwise.config import FEATURES
from reviewwise.validation_audit import audit


class ValidationAuditTests(unittest.TestCase):
    def test_chronology_and_test_invariance(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(500, len(FEATURES))), columns=FEATURES)
        frame['Amount'] = frame.Amount.abs()
        frame['Time'] = np.arange(500) // 2
        frame['Class'] = (np.arange(500) % 10 == 0).astype(int)
        frame.loc[200, FEATURES] = frame.loc[0, FEATURES]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.csv'
            frame.to_csv(path, index=False)
            first = audit(path, Path(tmp) / 'first.json')
            frame.loc[400:, FEATURES] = 999
            frame.loc[400:, 'Class'] = 1 - frame.loc[400:, 'Class']
            frame.to_csv(path, index=False)
            second = audit(path, Path(tmp) / 'second.json')
        self.assertEqual(first['results'], second['results'])
        self.assertFalse(first['test_scored'])
        self.assertTrue(any(row['excluded_seen_vectors'] for row in first['results']))
        for row in first['results']:
            self.assertLess(row['train_time_max'], row['validation_time_min'])
            self.assertEqual(row['tp'] + row['fn'], row['fraud'])

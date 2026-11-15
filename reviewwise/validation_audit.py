"""Expanding chronological validation and unseen-vector sensitivity; excludes test."""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from reviewwise.config import FEATURES, ROOT
from reviewwise.data import load_data, split_data, fingerprint
from reviewwise.training import build_models
from reviewwise.ranking import counts, selection, ranking_metrics
from reviewwise.artifacts import write_json


def audit(data_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output path; existing audit is preserved.')
    source_hash = fingerprint(data_path)
    splits = split_data(load_data(data_path))
    development = pd.concat([splits['train'], splits['validation']], ignore_index=True)
    times = development.Time.to_numpy()
    boundaries = np.flatnonzero(times[:-1] != times[1:]) + 1
    cuts = [int(boundaries[np.argmin(abs(boundaries - len(development) * f))])
            for f in (.4, .6, .8)] + [len(development)]
    if len(set(cuts)) != 4:
        raise ValueError('Insufficient timestamp boundaries for expanding windows.')
    rows = []
    for fold, (start, end) in enumerate(zip(cuts, cuts[1:]), 1):
        train, validation = development.iloc[:start], development.iloc[start:end]
        if set(train.Class) != {0, 1}:
            raise ValueError('Each training window must contain both classes.')
        model = build_models()['logistic_regression']
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(train[FEATURES], train.Class)
        scores = model.predict_proba(validation[FEATURES])[:, 1]
        # Exact vector match, not approximate hash equality; labels excluded.
        seen = validation[FEATURES].merge(train[FEATURES].drop_duplicates().assign(_seen=True),
                                         on=FEATURES, how='left', sort=False)._seen.eq(True).to_numpy(bool)
        for cohort, mask in [('all', np.ones(len(validation), dtype=bool)), ('unseen_vectors', ~seen)]:
            subset = validation.iloc[np.flatnonzero(mask)]
            metrics = counts(subset.Class, selection(scores[mask], subset.transaction_id, 10))
            rows.append({'fold': fold, 'cohort': cohort, 'train_rows': len(train),
                         'train_time_max': float(train.Time.max()),
                         'validation_time_min': float(validation.Time.min()),
                         'validation_time_max': float(validation.Time.max()),
                         'excluded_seen_vectors': int(seen.sum()) if cohort == 'unseen_vectors' else 0,
                         **metrics, **ranking_metrics(subset.Class, scores[mask])})
    if fingerprint(data_path) != source_hash:
        raise ValueError('Dataset changed during audit.')
    report = {'input_sha256': source_hash, 'model': 'baseline logistic regression C=1',
              'test_scored': False, 'results': rows,
              'interpretation': 'Descriptive expanding-window audit, not independent confirmation or customer-level deduplication. Each cohort receives its own 1% budget; denominators differ.'}
    write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/creditcard.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/validation_audit.json')
    args = parser.parse_args()
    audit(args.data, args.output)

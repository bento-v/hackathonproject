"""Prepare isolated, clearly labeled synthetic artifacts for interface practice."""
import numpy as np
import pandas as pd
from reviewwise.config import FEATURES, ROOT, experiment_parameters
from reviewwise.training import train
from reviewwise.evaluation import evaluate


def prepare():
    root = ROOT / 'tmp/synthetic_demo'
    if root.exists():
        raise ValueError('Synthetic demo already exists; reuse it or choose a clean workspace.')
    (root / 'data').mkdir(parents=True)
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(3000, len(FEATURES))), columns=FEATURES)
    frame['Time'] = np.arange(len(frame))
    frame['Amount'] = np.abs(frame.Amount) * 100
    frame['Class'] = (np.arange(len(frame)) % 40 == 0).astype(int)
    frame['V1'] += frame.Class * 3
    data = root / 'data/creditcard.csv'
    frame.to_csv(data, index=False)
    train(data, root / 'artifacts', root / 'results', experiment_parameters(0, 50, 8, 5))
    evaluate(data, root / 'artifacts', root / 'results')
    print('SYNTHETIC ONLY. Launch with REVIEWWISE_SYNTHETIC_DEMO=1 streamlit run app.py')


if __name__ == '__main__':
    prepare()

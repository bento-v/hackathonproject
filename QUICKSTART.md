# Run ReviewWise

Use Python 3.13. From the repository directory:

```sh
python -m venv .venv
```

Activate on macOS/Linux with `source .venv/bin/activate`, or on Windows
PowerShell with `.venv\Scripts\Activate.ps1`. Then:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

## Practice without downloading real data

```sh
python -m reviewwise.demo
```

On macOS/Linux:

```sh
REVIEWWISE_SYNTHETIC_DEMO=1 streamlit run app.py
```

On Windows PowerShell:

```powershell
$env:REVIEWWISE_SYNTHETIC_DEMO="1"
streamlit run app.py
```

The visible warning labels this synthetic mode. Generated transactions and
artifacts stay in tmp/synthetic_demo. Their metrics are not real-world evidence.
Remove the environment variable or start a new terminal to return to real mode.

## Real historical demo

Download the credited public ULB/Worldline data to data/creditcard.csv. Then:

```sh
python -m reviewwise.training
python -m reviewwise.evaluation
streamlit run app.py
```

Training refuses to overwrite a frozen run. Use fresh --artifacts and --results
paths for experiments. In the sidebar, Retrain models now reports validation
results only. Evaluate frozen run on test batch is a separate explicit action;
it activates the experiment only after successful evaluation.

## Validation and sensitivity checks

```sh
python -m reviewwise.tuning --output tmp/my-search.json
python -m reviewwise.validation_audit --output tmp/my-audit.json
```

The audit fits baseline logistic regression on expanding portions of the first
80% of the dataset, evaluating three later, nonoverlapping time windows. Equal
timestamps stay together. Test rows are never scored. It also evaluates each
window after excluding exact input vectors seen in its training history.
Each cohort receives its own 1% capacity, so review counts and denominators vary.
These are descriptive checks on already explored development data, not new
independent confirmation. The audit is not customer-level deduplication.

Measured baseline recall across the three windows was 40/52 (76.9%), 91/102
(89.2%), and 45/52 (86.5%). Excluding seen vectors left these fraud counts
unchanged but reduced legitimate review counts. Full metrics, including AP,
PR-AUC and errors, are in results/validation_audit.json.

## Two-minute presentation

1. Explain the analyst's limited review budget and show both queues.
2. Reveal historical results and compare equal review counts.
3. Increase the budget and explain extra reviews versus extra frauds caught.
4. Show a false positive and a missed fraud. Explain that model scores do not
   establish fraud and investigation is not automatic blocking.
5. Close with the small historical sample, anonymous features, and varying
   temporal results. Use real-mode evidence only for performance claims.

This project is a prototype and is not financial advice.

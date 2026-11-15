# Reproduce ReviewWise

## Setup

Tested locally on Windows with Python 3.13.5 and the exact versions in
requirements.txt. For a fresh checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the original Credit Card Fraud Detection CSV from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it at
data/creditcard.csv. A Kaggle account may be required. Do not commit the CSV.
The pipeline prints actionable errors for missing columns, nonnumeric,
nonfinite, missing, negative Time/Amount, or invalid Class values.

## Run the experiment

Run from the project folder:

```powershell
# Optional read-only data audit
.\.venv\Scripts\python.exe -m reviewwise.data --data data/creditcard.csv

# Training/selection uses training and validation only
.\.venv\Scripts\python.exe -m reviewwise.training --data data/creditcard.csv

# Evaluate the frozen models; no test-driven model selection occurs
.\.venv\Scripts\python.exe -m reviewwise.evaluation --data data/creditcard.csv

# Check core invariants and the local end-to-end app flow
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Launch the local app
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://127.0.0.1:8501. Stop the foreground server with Ctrl+C.
The server binds to localhost. No login, API key, or remote inference is used.

Training refuses to overwrite an existing frozen manifest. On this workspace,
training and evaluation have already completed, so launch the app directly.
For an explicit reproduction run, use separate paths:

```powershell
.\.venv\Scripts\python.exe -m reviewwise.training --artifacts tmp/repro_artifacts --results tmp/repro_results
.\.venv\Scripts\python.exe -m reviewwise.evaluation --artifacts tmp/repro_artifacts --results tmp/repro_results
```

Those example reproduction paths have already been used for a verification
training run locally. Choose another empty directory for another run.
The app reads the primary artifacts/ and results/ directories only.
Do not adjust settings in response to test scores. A rerun to fix a correctness
bug must document the bug; a reproduction run must keep the original policy.

## What is recorded

- artifacts/manifest.json: input SHA-256, dependency versions, model parameters,
  feature names, split membership hashes and boundaries, validation results,
  selected model, temporal window edges, and model-file hashes.
- artifacts/: local fitted models, split IDs, predictions without labels,
  separate labels, deterministic failure examples, and evaluation-file hashes.
- results/: aggregate data audit, validation results, test metrics, capacity
  curves, and the temporal breakdown. These compact files can be shared.

The app checks fingerprints, versions, model files, and evaluation-file hashes
before loading the demo. Missing or stale artifacts produce an error instead
of silently training. Only load joblib files created locally by this project.

Never publish data/, artifacts/, .venv/, caches, or secrets. The .gitignore
excludes them. The repository contains source, aggregate evidence, and the PDF
guide. The app runs locally; repository publication does not deploy the app
or submit it to the competition.

## Evaluation details

Capacity uses integer units: 10 means 1%, and K = (N * units + 999) // 1000.
Rank descending by score, with original CSV row ID ascending to resolve ties.
Precision and recall come from the selected queue; average precision and
trapezoidal PR-AUC use the full ranking. Undefined metrics are represented as
JSON null / UI N/A, not zero. Three temporal windows use the saved global
top-1% selections and include the final timestamp in the final window.

The duplicate audit counts exact full-row duplicates beyond their first
occurrence and unique matching input vectors across each pair of splits.
It does not deduplicate, infer customer identity, or establish data independence.

## Verification performed

- Unit checks cover timestamp groups, chronological boundaries, feature
  exclusion, training-only scaling, exact budgets, ties, confusion counts,
  undefined metrics, nested queues, selection rules, and artifact mismatches.
- Integration checks reconcile saved counts with predictions and temporal
  totals, verify deterministic failure examples, and exercise Streamlit's
  hidden/revealed states at 0.5%, 1%, and 2% capacity.
- A second training run reproduced split summaries, validation counts, model
  selection, and primary review queues exactly. Both models' validation scores
  agreed within absolute tolerance 1e-14.
- Browser visual verification was attempted but could not complete because
  computer-use tooling could not confidently determine Chrome's URL.

The tests use Streamlit's official AppTest interface. Local artifact integration
tests skip in a clean checkout until training and evaluation are complete.
No user usability study or external deployment test has been performed.

## Mac web experiments

From the downloaded project folder with the dataset at `data/creditcard.csv`:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Open http://localhost:8501. Expand the left sidebar if it is collapsed. Change
the sliders and click **Retrain models**. Wait for completion; on the full CSV
this may take several minutes. Changing sliders or the review budget alone
does not train. The button works even when no original fitted models exist,
provided the CSV and dependencies are available. With no CSV it is disabled.

Controls: log10(C) -3 to 2 (C 0.001 to 100); forest trees 50 to 400, maximum
depth 3 to 20, minimum leaf size 1 to 20. Defaults match the original baseline.
Class balancing remains fixed. Training fits preprocessing on training rows
only, chooses between the two candidates on validation recall at 1%, freezes
the models, then calculates test evidence. Test scores never choose the winner.

Successful runs live in `tmp/web_runs/<unique-run-id>/artifacts` and `results`.
Actual fitted settings are recorded in the manifest and displayed under
**Full evaluation and model selection**. Original `artifacts/` and `results/`
are not overwritten. Failed runs do not replace the active demo. Use
**Return to original saved run** to switch back. The active selection is per
browser session; restarting the app returns to the original saved run. If
there are no original artifacts, that return shows the normal setup message.
The run files remain on disk until removed. Keep `tmp/` out of Git.

Changing sliders is a manual validation experiment, not an automatic search.
The already-inspected historical test set is exploratory evidence for these
runs; freeze a candidate and use new untouched data to confirm improvement.

New synthetic-data tests exercise real custom training/evaluation and web
controls without requiring the downloaded dataset or existing artifacts:

```bash
python -m unittest discover -s tests -v
```

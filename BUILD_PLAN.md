# ReviewWise build plan

Status: local pipeline, frozen evaluation, demo, and documentation implemented.
12 automated tests pass. A repeat training run reproduces selection and the
primary validation queues. Browser visual verification remains incomplete.
PROJECT.md is the implementation contract; COMPETITION.MD records the brief.

## Implemented layout

```text
app.py                    Streamlit presentation and reveal state
reviewwise/
  data.py                 validation, IDs, fingerprinting, chronological splits
  ranking.py              deterministic top-K selection and metrics
  training.py             model fitting and validation-only selection CLI
  evaluation.py           frozen test evaluation and temporal breakdown CLI
  artifacts.py            artifact contracts and compatibility checks
tests/                    focused correctness and leakage checks
data/creditcard.csv        existing local input, excluded from Git
artifacts/                local models and row-level predictions, excluded
results/                  compact aggregate results for the submission
requirements.txt          versions verified in the working environment
README.md                 reproduction, measured results, limitations, credits
DEMO.md                   two-minute script with measured findings
.gitignore                data, environment, artifacts, caches, and secrets
```

Keep modules small enough for the user to explain. Prefer standard-library
unittest over an extra test dependency if sufficient. No extra service is needed.

Implemented commands (training refuses to overwrite an existing frozen run):

```powershell
.\.venv\Scripts\python.exe -m reviewwise.training --data data/creditcard.csv
.\.venv\Scripts\python.exe -m reviewwise.evaluation --data data/creditcard.csv
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Phase 1: input and baseline

- [x] Add ignore rules before Git tracking or generated artifacts.
- [x] Record compatible installed dependency versions.
- [x] Implement input validation, fingerprinting, and stable row IDs.
- [x] Implement chronological splits without dividing equal timestamps.
- [x] Record counts, ranges, prevalence, and duplicate diagnostics.
- [x] Implement exact top-K ranking, Amount baseline, and metrics.

Acceptance: the CSV loads or produces a precise data error; split and ranking
invariants pass; a hand-calculated example agrees with precision, recall,
TP, FP, FN, TN, and review counts.

## Phase 2: train and select without test evaluation

- [x] Implement the two specified pipelines with fixed settings and seed.
- [x] Fit training data only and resolve convergence issues.
- [x] Select by validation recall at 1%, with the specified tie-breaker.
- [x] Save pipelines and a manifest containing selection, parameters,
      features, versions, data hash, and split membership.
- [x] Check implementation against PROJECT.md before opening test results.

Acceptance: the same input/configuration reproduces split, selection, and
rankings. Test labels cannot affect model selection. Scaling statistics
match the training data rather than the entire dataset.

## Phase 3: final evidence

- [x] Evaluate frozen candidates and Amount at equal capacities.
- [x] Save counts, precision/recall, AP, PR-AUC, and capacity curves.
- [x] Break down the global 1% queue into the three temporal windows.
- [x] Choose failure examples using the documented rule.
- [x] Keep the validation winner as the headline model.

Acceptance: confusion counts reconcile with totals; every ranking uses K
reviews; window counts sum to global counts; increasing capacity cannot reduce
fraud caught for a fixed ranking; undefined metrics are explicit.

Do not tune after seeing test results. Document correctness bugs and rerun
affected evaluation when fixed. Aesthetic changes do not require retraining.
Never quietly replace an unfavorable result.

## Phase 4: demo

- [x] Load artifacts without training in Streamlit.
- [x] Implement capacity selector and side-by-side queues.
- [x] Implement explicit outcome reveal and reset behavior.
- [x] Add measured counts, capacity chart, and temporal breakdown.
- [x] Add deterministic failure examples and concise limitations.
- [x] Handle missing/stale artifacts and absent error categories clearly.

Acceptance: exercise all three capacities locally; verify labels and
outcome-derived visuals are hidden before reveal; verify changing capacity
resets reveal; compare displayed counts with saved evaluation results.

## Phase 5: submission preparation

- [x] Write README with measured results and verified reproduction commands.
- [x] Explain the contribution and limitation in plain language.
- [x] Credit sources and AI assistance; include the finance disclaimer.
- [x] Write DEMO.md: problem, input/output, measured comparison, one limitation.
- [ ] Confirm the user can explain splitting, selection, ranking, precision,
      recall, and the distinction between scores and probabilities.
- [x] Add ignore rules for raw data, secrets, local artifacts, and environments.
- [ ] Audit actual tracked files when the Git repository is created.
- [ ] Publish the repository and prepare the submission when requested.
- [ ] Complete visual browser review; automated app interaction tests pass.
- [ ] Record the optional video and verify public submission links.

Target first submission: November 15, 2026, 6:59 PM Pacific.
Official deadline: 8:59 PM Pacific. Publication and submission are separate
actions from local implementation.

## Implemented test coverage

- Equal timestamps stay together at split boundaries; split IDs never overlap.
- Class, Time, and transaction_id never enter the model feature matrix.
- Scaling uses training rows only.
- Tied scores select exactly K rows deterministically, including exact integer
  capacity boundaries that floating-point arithmetic could mishandle.
- Metrics agree with a small manually calculated example.
- Zero selections and windows without fraud produce the specified N/A values.
- Temporal windows cover test rows once and preserve the global queue.
- Stored model selection is unaffected by changes to test outcomes.
- Artifact/data mismatches fail clearly instead of serving stale results.

## Scope control

If time gets tight, preserve correct evaluation, equal-budget comparison,
the temporal check, one working reveal flow, and the README. Reduce styling
and secondary charts first. Do not add models or integrations to compensate
for an unclear result. Keep every performance claim tied to the frozen evidence in results/.

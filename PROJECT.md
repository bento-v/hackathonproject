# ReviewWise

Implementation specification for the Veritas AI International High School
Hackathon 2026, AI + Finance track.

Status: implemented locally with frozen measured results; see README.md.
Automated checks pass. Visual browser review and publication remain pending.
Read BUILD_PLAN.md for build order and acceptance checks.
Read COMPETITION.MD for the competition summary and original brief link.

## User and contribution

User: a payment risk analyst prioritizing investigations after a historical
transaction batch closes, with capacity to review 1% of payments. The 1%
capacity is a prototype assumption, not a validated customer requirement.

Main question: at the same review budget, does a learned ranking catch more
fraud than reviewing the largest payments first?

Our contribution beyond the suggested classifier:
1. Measure fraud caught against review effort.
2. Show what extra capacity buys at 0.5%, 1%, and 2%.
3. Check whether aggregate results hide weak periods within the test batch.

The product prioritizes a queue. Anonymous data does not support a complete
investigation workflow or a decision to block a card. This is historical
batch review, not real-time payment authorization.

## Scope and stack

Deliver a reproducible local pipeline, Streamlit demo, focused correctness
checks, approximately one-page README, and a two-minute demo script.
Use Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, and joblib.
Use the existing virtual environment and record working dependency versions.

Exclude chatbots, invented transaction explanations, accounts, payment
integrations, automatic blocking, live ingestion, demographic fairness claims,
and additional model families from this build.

## Data contract

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Local input: data/creditcard.csv. Never commit raw data or .venv.

Required columns: Time, V1 through V28, Amount, Class.
- All required values must be numeric, finite, and nonmissing.
- Class must be 0 or 1; 1 denotes historical fraud.
- Time and Amount must be nonnegative.
- Reject invalid data clearly; do not silently drop or impute rows.
- transaction_id is the original one-based CSV data-row position. Use it
  for auditing and tie-breaking, never as a feature.
- Record input SHA-256, row count, class counts, time range, duplicate rows,
  and identical model-input vectors across splits. Do not silently deduplicate
  or claim these records are independent customers.

Model inputs: V1 through V28 and Amount only. Exclude Class, Time, IDs,
split names, and any derived label information. Use Time only for ordering,
splitting, display, and the stability check.

## Chronological split

Sort by Time ascending, then transaction_id ascending. Target approximately
60% training, 20% validation, and 20% testing. Keep equal timestamps together.

Choose valid group boundaries nearest 60% and 80% of the total row count.
For equally near boundaries, choose the earlier one. Require three nonempty,
chronologically ordered splits; stop if this is impossible. Record actual
counts, boundaries, time ranges, and fraud prevalence.

Require both classes in each split. Otherwise stop and describe the limitation;
do not silently choose more favorable boundaries. Inspecting split counts
is allowed, but model decisions must use validation performance only.

## Initial models and selection

Use random_state=42 where supported and exactly two initial candidates:

1. StandardScaler followed by LogisticRegression:
   class_weight="balanced", C=1.0, solver="lbfgs", max_iter=2000.
2. RandomForestClassifier on unscaled inputs:
   n_estimators=200, max_depth=12, min_samples_leaf=5,
   class_weight="balanced_subsample", n_jobs=-1.

Fit every preprocessing step and model using training rows only. Resolve
convergence warnings before accepting training. Rank by the positive-class
predict_proba output, displayed as a model score rather than an established
calibrated fraud probability.

Select the model with highest validation recall at 1% capacity. If tied,
select logistic regression for simplicity. Freeze the selected model,
features, parameters, preprocessing, and evaluation policy before evaluating
test performance. Keep the training-fitted models; do not refit on validation.

The original baseline uses the fixed settings above. The web interface now
supports explicit parameter experiments: logistic log10(C) from -3 to 2 in
steps of 0.5; forest trees from 50 to 400 in steps of 50, maximum depth from
3 to 20, and minimum leaf size from 1 to 20. All other parameters, including
balanced class weights and seed, remain fixed. Sliders do not start training;
the Retrain models form submission fits both candidates and selects on
validation recall at 1%, then evaluates the frozen winner and candidates.
Each run records its actual parameters in a separate tmp/web_runs directory.
Only a fully trained and evaluated run becomes active; failure preserves the
previous active run. Users can return to the original artifacts. Outcomes reset
to hidden on successful retraining and when returning to the original run.
These experiments happen after the original test results were inspected and
must be labeled exploratory. New untouched data is needed to confirm any
improvement. Never automatically optimize against test outcomes.

## Baselines and review policy

Operational baseline: rank by Amount descending.
Educational baseline: flag nothing, illustrating why accuracy is misleading.
The latter uses zero capacity and is not an equal-budget competitor.

For a batch of N rows and capacity c, select exactly
K = min(N, ceil(c * N)). Rank by score descending, breaking every tie by
transaction_id ascending. Apply the same rule to the Amount ranking.

Primary capacity: 1%. Comparison capacities: 0.5% and 2%.
Capacity curve: 0.1% through 5.0%, in increments of 0.1 percentage points.
Calculate K using integer capacity units so floating-point rounding cannot
accidentally add a row at an exact boundary.

This is batch-relative selection, not a universal score cutoff. Do not use a
0.5 probability threshold for the headline comparison.

## Evaluation

Evaluate both frozen candidates and Amount on the same test rows. Keep the
validation-selected model as the headline model even if another wins on test.
Report actual outcomes even if the simple baseline wins.

At each capacity report:
- N and K.
- Fraud caught (TP), fraud missed (FN), legitimate transactions selected (FP),
  and legitimate transactions not selected (TN).
- Precision = TP / (TP + FP); recall = TP / (TP + FN).
- Additional fraud caught and recall difference versus Amount at equal K.

For each ranking also report average_precision_score and trapezoidal PR-AUC
using precision_recall_curve and auc(recall, precision). Label them separately.
Reporting both implementations is our choice; the brief requests area under
the precision-recall curve without specifying these two implementations.

For flag-nothing, recall is zero and precision is N/A because no rows are
selected. Do not present a ranking curve for a policy without a ranking.
All zero-denominator metrics must be N/A with an explanation. Accuracy is
not the headline metric.

## Temporal stability check

Before final evaluation, fix three equal-duration windows spanning the test
Time range. Use left-inclusive/right-exclusive intervals, including the final
endpoint in the last interval. Equal timestamps always share a window.

Select the global top 1% of the complete test batch first. Break down those
same selections by window. Do not rerank and select a fresh 1% per window.
The policy remains one historical batch throughout.

Show transactions, fraud count, reviews allocated, TP, FP, FN, precision, and
recall per window. Total review budgets are equal across rankings, but their
allocation to individual windows may differ; show that difference explicitly.
Keep windows without fraud visible and label recall N/A. Show denominators
so a single error is not disguised as a broad stability conclusion.

This is a descriptive check on a short historical sample, not a statistical
claim about drift, long-term robustness, or a reason to retune using test data.
Do not manufacture a weak period if the evidence does not show one.

## Demo behavior

One main screen with supporting evidence below it:
1. Load the held-out historical test batch and display its size and time range.
2. Choose 0.5%, 1% (default), or 2% review capacity.
3. Display the selected model queue beside the largest-payment queue.
4. Reveal historical outcomes using an explicit button.
5. Show equal-budget counts, capacity curve, and temporal breakdown.
6. Inspect a false positive and a missed fraud at the primary 1% capacity.

Queue columns: rank, transaction ID, elapsed time, amount, and model score
where applicable. Explain that the baseline ranks by amount. Never infer a
merchant, geography, customer identity, or calendar date.

Before reveal, hide labels, label-derived colors, performance summaries, and
failure examples. Reset reveal when capacity or batch changes. This is a
presentation interaction, not a privacy or access-control boundary.

After reveal, choose the highest-ranked false positive and highest-ranked
missed fraud deterministically. Show rank, amount, the other method's rank,
and minimum review count needed to include the transaction. State whether
2% capacity catches the missed fraud. Explain that these examples concern
the primary 1% policy. If an error category has no examples, say so.

Explain observations rather than invented causes. Test labels never affect
ranking order. A table of anonymous feature values is not the main demo.

## Artifacts and reproducibility

Provide separate commands to train/select, evaluate frozen models, and launch
the app. The app loads saved artifacts; it never trains on rerun or needs
labels to produce scores.

Persist locally:
- Fitted model pipelines and configuration/selection manifest.
- Input fingerprint, package versions, split boundaries, validation results.
- Test predictions, IDs, and metadata; keep labels as a separate evaluation
  input, joined only for evaluation/reveal.
- Aggregate metrics, capacity curves, temporal results, failure-example IDs.

Fail clearly on absent/stale artifacts, mismatched input fingerprints, or
incompatible features. Never load untrusted joblib files.

Commit code, configuration, verified dependencies, aggregate results, and
documentation. Exclude raw data, row-level exports, model binaries, secrets,
caches, and .venv. Document how to reproduce excluded artifacts.
Publication and competition submission are separate from local preparation.

## Claims and submission

The main takeaway must use actual measurements:
"With K reviews, the selected model caught X frauds versus Y for reviewing
the largest payments. [Measured limitation or stability finding.]"

Disclose old data spanning two days, anonymous features, the unvalidated
capacity assumption, limited investigation context, and lack of deployment
or calibration evidence. Fraud transaction amounts are not money saved.
These data do not support demographic fairness conclusions.

Prepare an accessible GitHub repository, approximately one-page README,
and optional recommended demo video of at most two minutes. Cite the dataset,
libraries, tutorials, and AI tools including Codex. The team must understand
and be able to explain every implemented component.

Deadline: November 15, 2026, 8:59 PM Pacific. Target a first complete submission
by 6:59 PM Pacific; check link access in a private window.

Required disclaimer:
"This project is a prototype and is not financial advice."

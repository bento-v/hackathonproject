# Review and validation-only tuning

Update: the public dataset has now been downloaded and the search executed.
See [TUNING_RESULTS.md](TUNING_RESULTS.md). No improvement in validation recall
at 1% was found. Dependencies are now available: 16 tests passed and two
full-artifact integration checks were skipped. The limitations below describe
the initial review before this follow-up experiment.

Reviewed bento-v/hackathonproject main on November 15, 2026. This document records the review findings and follow-up experiments.

## Findings

- Fixed: counts() accepted column-shaped labels and broadcast them against a
  selection vector, silently inflating confusion counts. Ranking and confusion
  metrics now require one-dimensional inputs. The existing pipeline supplies
  vectors, so this finding does not invalidate its published measurements.
- Fixed: a valid chronological split can leave the test batch with a single
  distinct timestamp. Training previously saved models that evaluation could
  not analyze in positive-duration windows. Training now rejects this before fit.
- Experiment design: every sidebar experiment evaluates the same test set.
  The README already calls this exploratory. Use the new validation-only CLI
  for parameter selection, and confirm any improvement on new untouched data.
- Duplicate feature vectors across splits remain a generalization limitation,
  already disclosed in README. No automatic deduplication was applied because
  that would change the declared experiment and require new baseline results.

## Tune locally

Use your existing environment with requirements.txt installed and the cited
public dataset at data/creditcard.csv. From the repository root:

```bash
python -m reviewwise.tuning --output tmp/tuning.json
python -m reviewwise.training --tuning-report tmp/tuning.json --artifacts tmp/tuned/artifacts --results tmp/tuned/results
```

The search fits five logistic C values (1, .01, .1, 10, 100) and four forest
settings (baseline; depth 8; depth 16; leaf size 1). All other settings remain
fixed. Each candidate trains on the original training rows only. Selection
uses validation recall at 1%; ties keep the first candidate within a family
and favor logistic regression between families. AP and PR-AUC are recorded
for context and do not break ties. Test features are never scored in search.

The report records the dataset hash, settings, candidate validation metrics,
and winner. Training verifies the dataset hash before consuming the report,
fits both chosen family configurations, and freezes the normal manifest.
It does not evaluate test performance. The original artifacts are preserved.

For a descriptive comparison on the already-seen test batch, explicitly run:

```bash
python -m reviewwise.evaluation --artifacts tmp/tuned/artifacts --results tmp/tuned/results
```

This comparison cannot establish an unbiased improvement. Repeated validation
search also risks overfitting, especially with only 57 validation frauds.
Freeze the policy before obtaining an independent confirmation dataset.

## Verification and limitations

- Existing core tests: 9 passed.
- New regression/search tests: 3 passed on synthetic data, including invariance
  to changes in test features and labels and a guard against scoring test rows.
- Python compilation passed.
- The actual fraud CSV and model artifacts are absent. No new fraud-detection
  performance was measured and no model improvement is claimed.
- Full artifact, integration, and Streamlit tests were not run: Streamlit is
  unavailable and dependency installation was blocked by network restrictions.
- Checks used available Python 3.12, NumPy 2.3.5, pandas 2.2.3 and sklearn 1.8.0,
  not the repository's pinned reproduction environment. Pins were preserved.

OpenAI Codex assisted with this review, implementation, and synthetic checks.

# Measured tuning results

Nine configurations were fitted on the public fraud dataset on November 15, 2026.
Selection used validation recall at 1% capacity only. Test rows were not scored.

**No improvement in the primary metric was found. Keep baseline logistic regression (C=1).**

| Candidate | Caught / 57 | False positives | Missed | Precision | Recall | AP | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic C=1 | 49 | 521 | 8 | 8.60% | 85.96% | 0.772324 | 0.771922 |
| Logistic C=0.01 | 49 | 521 | 8 | 8.60% | 85.96% | 0.746019 | 0.745349 |
| Logistic C=0.1 | 49 | 521 | 8 | 8.60% | 85.96% | 0.769115 | 0.768676 |
| Logistic C=10 | 49 | 521 | 8 | 8.60% | 85.96% | 0.772492 | 0.772092 |
| Logistic C=100 | 49 | 521 | 8 | 8.60% | 85.96% | 0.772484 | 0.772084 |
| Forest depth=12, leaf=5 | 48 | 522 | 9 | 8.42% | 84.21% | 0.766606 | 0.766028 |
| Forest depth=8, leaf=5 | 48 | 522 | 9 | 8.42% | 84.21% | 0.770588 | 0.770164 |
| Forest depth=16, leaf=5 | 48 | 522 | 9 | 8.42% | 84.21% | 0.759432 | 0.758896 |
| Forest depth=12, leaf=1 | 47 | 523 | 10 | 8.25% | 82.46% | 0.758914 | 0.758362 |

Each queue contains exactly 570 reviews from 56,964 validation transactions.
Logistic C=10 slightly increased AP (0.772324 to 0.772492), but caught no additional fraud at the chosen budget. This tiny secondary-metric change does not justify replacing the baseline.

The dataset audit matches the repository: 284,807 rows, 492 frauds, and split fraud counts 360/57/75. Downloaded from the TensorFlow public mirror of the credited ULB/Worldline dataset. Raw data stays excluded from Git.

Full parameters, dataset SHA-256, runtime versions, and audit are in [results/tuning_validation.json](results/tuning_validation.json). This environment differs from the pinned baseline environment; both baseline candidates were rerun here for comparison.

Tests: 16 passed, 2 integration checks skipped because frozen full-dataset evaluation artifacts are absent. Synthetic app retraining and artifact checks passed after dependencies were installed. No test performance was used for this search, and no trained replacement is recommended.

Next experiment: predeclare a small additional search or preprocessing change and assess its consistency across chronological validation windows before touching new confirmation data. Avoid broad adaptive searches against just 57 validation frauds.

OpenAI Codex assisted with the implementation, experiment execution, and reporting.

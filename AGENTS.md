# Project instructions

Read PROJECT.md before making changes.
Read BUILD_PLAN.md for implementation order and acceptance checks.
PROJECT.md defines our implementation choices; COMPETITION.MD distinguishes
the competition requirements from our design.

## Workflow
- Build a working data pipeline and baseline before the UI.
- Keep code simple enough for a high school team to explain.
- Run relevant checks after changes.
- Never invent results or claim unmeasured performance.

## Data and evaluation
- Use public data only.
- Keep raw data, credentials, and .venv out of Git.
- Split chronologically; keep equal timestamps together.
- Never include Class in model inputs.
- Fit preprocessing on training data only.
- Use validation data for model and threshold selection.
- Do not tune against test results.
- Compare ranking methods at identical review capacity.
- Report precision, recall, PR-AUC, and average precision.
- Include false-positive and false-negative counts.
- Use class_weight="balanced" for logistic regression.

## Product
- Present the app as a historical batch-review prototype.
- Label outputs as model scores, not fraud probabilities.
- Do not invent explanations for anonymous features.
- Include: “This project is a prototype and is not financial advice.”

## Documentation
- Record actual results and limitations in README.md.
- Credit datasets, tutorials, libraries, and AI tools.

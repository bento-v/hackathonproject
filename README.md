# ReviewWise

**Make every review count.** AI + Finance · Veritas AI Hackathon 2026

ReviewWise helps a payment risk analyst prioritize a closed transaction batch
when only a small fraction can be investigated. Choose a review budget, compare
the model queue with the largest payments, and reveal historical outcomes.
Selection means investigation, not automatic blocking.

## What we added

Beyond the brief's card-fraud classifier, we ask how much fraud a fixed review
budget catches, what extra review capacity buys, and whether the aggregate
result hides weaker periods. Every ranking receives exactly the same number
of reviews. Three equal-duration windows describe the original global 1%
queue; they do not select new queues in each window.

## How it works

The public dataset has 284,807 transactions and 492 fraud labels. We sort by
Time and keep equal timestamps together: 170,882 training rows (360 frauds),
56,964 validation rows (57), and 56,961 test rows (75). Models use V1–V28 and
Amount; labels, Time, and IDs are excluded. Scaling is fitted on training only.

Balanced logistic regression beat the specified random forest on validation
recall at 1%: 49/57 versus 48/57. We froze logistic regression before evaluating
the test set. All settings were specified in advance; there was no tuning
against test results. See [the specification](PROJECT.md).

## Measured results

On the held-out test batch, **570 reviews caught 65 of 75 frauds with the selected
model, versus 3 with the largest-payment rule.** There were still 505 legitimate
payments in the model queue and 10 missed frauds.

| Method at 1% capacity | Caught | Legitimate reviews | Missed | Precision | Recall | AP | Trapezoidal PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression (selected) | 65 | 505 | 10 | 11.40% | 86.67% | 0.7451 | 0.7432 |
| Random forest | 65 | 505 | 10 | 11.40% | 86.67% | 0.8122 | 0.8120 |
| Largest payments | 3 | 567 | 72 | 0.53% | 4.00% | 0.0014 | 0.0013 |

AP and PR-AUC summarize each complete ranking, not just the selected queue.
The forest had better test AP, but the validation-selected model remains the
headline model. Flagging nothing catches zero fraud with zero reviews;
precision is undefined and it is not an equal-budget competitor.

**More work did not buy much more recall:** 285, 570, and 1,140 model reviews
caught 64, 65, and 66 frauds respectively. The original 1% queue caught 35/38,
23/26, and 7/11 frauds across the three windows. The last window was weaker,
but 11 frauds are too few to establish a general drift claim.

A concrete failure: a historically legitimate payment ranked second with a
score rounded to 1.0. A missed fraud ranked 790th, outside the 570-slot queue;
2% capacity would include it. These show score error and capacity constraints,
not an established explanation of customer behavior.

## Limits and next steps

This is old data spanning only two days, with anonymous features and no usable
demographic groups. The audit found 1,081 exact duplicate rows beyond the first
and 312 distinct feature vectors shared between training and test. They were
retained as specified; this is not evidence of generalization to new customers.
The 1% capacity is an assumption. Scores are not calibrated probabilities,
and fraud amounts are not money saved. Next: validate capacity with an analyst
and evaluate a frozen policy on longer, independently sourced data.

**This project is a prototype and is not financial advice.**

## Run and reproduce

With the existing local environment and generated artifacts:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [the local demo](http://127.0.0.1:8501). See [REPRODUCING.md](REPRODUCING.md)
for fresh setup, training, evaluation, tests, and artifact rules, and
[DEMO.md](DEMO.md) for the two-minute presentation. Aggregate evidence is in
[results/metrics.json](results/metrics.json).

For a visual walkthrough of the data, models, results, and code, read
[ReviewWise Explained (PDF)](output/pdf/ReviewWise_Explained.pdf). The guide
records the project's status before repository publication.

## Credits

[ULB/Worldline dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud);
[competition brief](https://drive.google.com/file/d/15tYD48XRmCgWbk1_lLDtc8lebi6qm-4-/view);
[Python](https://www.python.org/), [NumPy](https://numpy.org/),
[pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/),
[scikit-learn](https://scikit-learn.org/), [Streamlit](https://streamlit.io/),
[Plotly](https://plotly.com/python/), [joblib](https://joblib.readthedocs.io/),
[Apache Arrow](https://arrow.apache.org/).
OpenAI [Codex](https://openai.com/codex/) assisted with planning, implementation,
testing, and documentation; official library documentation informed API use.
The student must review and understand the implementation before submission.
Team details are supplied in the submission form; they have not been entered here.

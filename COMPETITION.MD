# Veritas AI International High School Hackathon 2026
## Rules and Credit Card Fraud Project Guidelines

Source: Veritas AI International High School Hackathon 2026
Project Briefs, particularly pages 3–4, 8–9, and 18–21.

Original PDF:
https://drive.google.com/file/d/15tYD48XRmCgWbk1_lLDtc8lebi6qm-4-/view

This document summarizes the relevant competition instructions.
Consult the original PDF if clarification is needed.

## 1. Competition dates

- Hacking starts: Saturday, November 14, 2026, at 12:00 AM Eastern.
- Submissions close: Sunday, November 15, 2026, at 11:59 PM Eastern.
- Pacific deadline: Sunday, November 15, at 8:59 PM.
- Submit an initial complete version early.
- Resubmissions are allowed before the deadline; judges use the latest.
- Only work submitted before the deadline is judged.

## 2. Competition rules

### Team eligibility
- Teams consist of one to three high school students.
- Students may attend different schools or live in different countries.
- Everyone listed on the submission must have contributed.

### Build during the competition
- All project work must happen within the competition build window.
- Learning tools, running starter notebooks, and browsing datasets
  beforehand are allowed.
- Extending a project built before the hackathon is not allowed.

### AI must be central
- AI or machine learning must perform the project's main function.
- A regular website built with AI coding tools does not qualify by itself.

### Allowed resources
Teams may use:
- Public datasets.
- Pretrained models.
- AI APIs.
- Open-source libraries.
- Tutorial code.
- Coding assistants and chatbots.

Cite these resources in the write-up. The team must understand the
project and be able to explain every part to judges.

### Originality and privacy
- Do not submit someone else's project as your own.
- Do not collect personal information without permission.
- Use public or synthetic data.

### Submission scope
- Submit one project per team.
- Choose one competition track.

### Honest reporting
- Report the results actually obtained, including weak results.
- Do not invent scores or hide failures.

### Required finance disclaimer
Include this exact sentence in the write-up:

"This project is a prototype and is not financial advice."

## 3. What every project must demonstrate

### A specific user
Name one specific user and explain when they would use the tool.

### Real inputs and outputs
Show an actual input producing a result.

### Comparison with a simple baseline
Report the baseline score alongside the model score.

### Failure analysis
Show at least one example the model gets wrong and explain a possible
reason.

### Unseen test data
All reported evaluation results must come from data the model did not
see during training.

## 4. AI + Finance track requirements

Challenge:

Build an AI tool that helps a person or small business make a safer
or fairer money decision, and show that it beats a simple rule.

Requirements:
- Compare the model with at least one simple baseline.
- Report results for both the baseline and model.
- Choose metrics appropriate to the problem.
- For rare events such as fraud, report precision and recall rather
  than relying on accuracy.
- If the data has dates or temporal ordering, train on earlier data
  and test on later data.
- Include the required financial-advice disclaimer.

Finance judges pay particular attention to:
- Appropriate measures of success.
- Fair comparison against a simple baseline.
- Whether a real person could trust the tool enough to act on it.
- Checks for unfair treatment of groups, where the data supports them.

Do not invent demographic information when the dataset lacks it.

## 5. Suggested project: Card Fraud Detection

Source: Section 4.2.2, page 9.

Level: Intermediate.
Data type: Tabular.
Suggested tools: Google Colab, pandas, scikit-learn.

### Goal

Flag card transactions likely to be fraudulent so a bank can review
them before approving payment.

### Dataset

Use the Credit Card Fraud Detection dataset on Kaggle.

Dataset page:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The brief describes:
- 284,807 transactions.
- European cardholders.
- Transactions recorded over two days in 2013.
- 492 fraudulent transactions.

A free Kaggle account may be needed to download it.

### Columns

- V1–V28: features anonymized for privacy.
- Time: transaction timing.
- Amount: transaction amount.
- Class: target label.
  - 1 means fraud.
  - 0 means not fraud.

### Baseline warning

Predicting "not fraud" for every transaction achieves approximately
99.8% accuracy while catching no fraud.

Therefore, accuracy must not be the main measure of success.

### Suggested modeling steps

1. Download and inspect the dataset.
2. Establish the baseline that predicts no fraud.
3. Train logistic regression with class_weight="balanced".
4. Train a tree-based model, such as random forest.
5. Evaluate on held-out test data.

Apply the finance track's temporal split requirement:
train on earlier transactions and test on later transactions.

### Required reported metrics for this suggested project

Report:
- Precision.
- Recall.
- Area under the precision-recall curve.

Implementation note:
Average precision and trapezoidal PR-AUC are related but different.
Name the metric implementation used. Our project can report both.

## 6. Make the project your own

Add at least one contribution beyond the suggested steps.

The brief suggests:
- Choose a decision threshold based on the cost of missed fraud
  versus wrongly blocked cards, and explain the choice.
- Test whether fraud patterns change over the two days and discuss
  deployment implications.
- Make flags understandable to a bank employee deciding whether
  to block a card.

Other general ways to add originality:
- Design for a specific user or setting.
- Use another appropriate public dataset.
- Ask an additional research question.
- Investigate when and why the model fails.

Teams following only the provided steps still have a complete
submission, but additions beyond the baseline help distinguish it.

## 7. Our project's planned extension: ReviewWise

This section records our design choices.
These are not additional competition requirements.
PROJECT.md is the authoritative, detailed implementation specification;
BUILD_PLAN.md records the implementation sequence and acceptance checks.

### User
A payment risk analyst with limited capacity to review transactions.

### Main research question
At a fixed review capacity of 1% of transactions, does a learned ranking
catch more fraud than reviewing the largest payments first?

### Models and baselines
- Never flag any transaction.
- Rank transactions by Amount.
- Logistic regression with class_weight="balanced".
- Random forest.

### Evaluation design
- Sort by Time.
- Use approximately the earliest 60% for training,
  next 20% for validation, and final 20% for testing.
- Keep equal timestamps together.
- Fit preprocessing only on training data.
- Use validation data for model and threshold selection.
- For ReviewWise, select by validation recall at 1% capacity, breaking ties
  in favor of logistic regression. Use top-K batch selection rather than
  tuning a fixed score threshold; see PROJECT.md for exact settings.
- Freeze the configuration before final test evaluation.
- Never include Class in model inputs.
- Compare rankings at identical review capacity.

### Additional metrics
- Precision and recall at 1% review capacity.
- Fraud caught and missed.
- Legitimate transactions flagged.
- Recall across several review capacities.
- A descriptive breakdown of the global 1% queue across three consecutive,
  equal-duration test windows, including fraud counts and reviews per window.
  Do not select a separate top 1% inside each window.

### Demo scope
The app simulates historical batch review.
Selecting the top 1% requires observing the complete batch.
Do not describe this as real-time payment authorization.

### Interpretation limits
- The dataset is old and covers only two days.
- Anonymous features limit meaningful business explanations.
- Model scores are not automatically calibrated probabilities.
- Detected fraud amounts do not equal money saved.
- Offline results do not establish real-world deployment performance.

## 8. Submission requirements

### Code — required
Provide either:
- A Google Colab notebook, or
- A GitHub repository.

Anyone with the link must be able to view it.
If submitting a notebook, save it with outputs visible.

### Write-up — required
Approximately one page.

Acceptable formats include:
- A Google Doc with public link access.
- The GitHub README.
- A PDF or Word document uploaded through the submission form.

### Team details — required
Provide:
- Selected track.
- Every contributing member's name.
- Email.
- School.

### Credits — required
Link every dataset, pretrained model, tutorial, and AI tool used.
Credit the libraries and other resources used in the project.

### Demo video — optional but recommended
- Maximum length: two minutes.
- Use an unlisted YouTube link or accessible Google Drive link.

### Sharing checks
- Open every submission link in a private/incognito window.
- Confirm it works without signing in.
- Check teammates' names and email addresses.
- Submit early and resubmit if needed.

## 9. Suggested write-up structure

# Project name
Track: AI + Finance
Team: names

## The problem
Who has the problem, and why does it matter?

## What we built
What goes in, what comes out, and how the user uses it.

## What we added
Explain the contribution beyond the suggested card fraud project.

## How it works
Describe the dataset, number of examples, cleaning, split, and models.

## Results
Report actual baseline and model results on held-out test data.
Show a failure and explain a possible reason.

## What we'd do next
Give one or two concrete next steps.

## Sources
Link datasets, models, tutorials, libraries, and AI tools.

Include:
"This project is a prototype and is not financial advice."

## 10. Suggested two-minute demo

- About 15 seconds: the user and problem.
- About 60 seconds: a real input producing a result.
- About 30 seconds: data, model, and baseline comparison.
- About 15 seconds: one limitation and next step.

A screen recording with narration is sufficient.

## 11. Judging criteria

### Technical quality
The project works, uses a sound approach, and is tested fairly.

### Impact
It addresses a real problem for identifiable users.

### Creativity
It adds a fresh question, setting, dataset, or approach.

### Clarity
Judges can understand the project in two minutes.

The brief emphasizes that honest, modest results are better than
unexplained, suspiciously strong results.

## 12. Final checklist

- [ ] Team and project comply with the competition rules.
- [ ] AI performs the main task.
- [ ] Specific user and decision are clear.
- [ ] Public or synthetic data is used appropriately.
- [ ] Earlier transactions train the model; later ones test it.
- [ ] Test labels never enter the model inputs.
- [ ] Preprocessing is fitted only on training data.
- [ ] Simple baseline results are reported.
- [ ] Precision, recall, and PR-AUC are reported.
- [ ] At least one original contribution is explained.
- [ ] At least one model failure is shown.
- [ ] Results are real and reproducible.
- [ ] Financial disclaimer is included.
- [ ] Sources and AI tools are credited.
- [ ] Code and write-up links work without signing in.
- [ ] Submission is completed before the deadline.

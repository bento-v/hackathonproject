# ReviewWise: two-minute demo

Start the local app at 1% capacity with outcomes hidden. Rehearse aloud;
shorten narration if it exceeds two minutes. All numbers below are from the
frozen run in results/metrics.json, not estimates or invented targets.

## 0:00–0:15 — the problem

"A payment risk analyst cannot investigate every transaction. ReviewWise asks:
if you can review just 1%, can a model catch more fraud than picking the
largest payments? This is a historical batch-review prototype."

## 0:15–0:45 — input and output

Show the batch size, 1% selector, and two queues.

"Here are 56,961 held-out transactions. Both strategies get exactly 570
reviews. The left queue uses logistic regression; the right picks the largest
payments. The historical labels are hidden. These scores rank transactions;
they are not calibrated fraud probabilities."

Click **Reveal historical outcomes**.

"The model catches 65 of 75 frauds. The simple rule catches 3. But the model
also sends 505 legitimate payments for review and misses 10 frauds."

## 0:45–1:10 — our contribution

Scroll to the review-capacity chart and table.

"Our contribution is measuring what review effort actually buys. At half a
percent, 285 reviews catch 64 frauds. Doubling to 570 reviews catches only one
more. Doubling again catches just one more. More review work is not automatically
a large improvement. We did not change our primary 1% policy after seeing this."

## 1:10–1:35 — challenge the average

Show the temporal table and missed-fraud example.

"The same queue catches 35 of 38 frauds in the first window, but only 7 of 11
in the last. Eleven is a small sample, so we cannot call this proof of drift.
This missed fraud ranked 790th, outside our 570 slots; a 2% budget includes it.
Anonymous features do not tell us the customer's story."

## 1:35–2:00 — method and limits

"We trained on earlier transactions, selected logistic regression on validation
recall, and froze it before testing. We also report the random forest, including
its better test average precision. The data covers just two days in 2013 and
contains repeated feature vectors across splits. Next we would validate the
review budget with an analyst and test on longer, independent data."

## Questions to be able to answer

- Why not accuracy? Fraud is rare; flagging nothing looks accurate but catches none.
- Why earlier data first? It prevents training on later transactions when evaluating earlier ones.
- Why logistic regression? It caught 49/57 validation frauds at 1%, versus 48/57 for the forest.
- Why not switch to the forest now? Test outcomes cannot choose the headline model.
- What does 86.7% recall mean? It caught 65 of the test set's 75 labeled frauds.
- What does 11.4% precision mean? 65 of 570 selected transactions were labeled fraud.
- Why two PR metrics? Average precision and trapezoidal integration summarize the curve differently.
- Is the model preventing fraud? No. This evaluation measures prioritization, not intervention outcomes.
- Can it explain a merchant or a demographic disparity? No. Those facts are not available in the inputs.

The video is optional and has not been recorded. The write-up must include:
"This project is a prototype and is not financial advice."

"""A historical review-budget demo. Train/evaluate separately before launching."""

import pandas as pd
import plotly.express as px
import streamlit as st

from reviewwise.artifacts import load_evaluation, read_json
from reviewwise.config import DISPLAY_NAMES, ROOT
from reviewwise.ranking import rank_order, review_count

st.set_page_config(page_title="ReviewWise | Make every review count", page_icon="◈", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1280px; padding-top: 2.5rem; padding-bottom: 3rem;}
  h1 {letter-spacing: -.055em; font-weight: 650 !important;}
  h2, h3 {letter-spacing: -.025em;}
  .eyebrow {font-size: .76rem; letter-spacing: .16em; font-weight: 700; color: #136B50;}
  .lede {font-size: 1.12rem; color: #52645D; max-width: 730px; line-height: 1.6;}
  .status {display: inline-block; border: 1px solid #CCD9D0; border-radius: 999px;
           padding: .3rem .7rem; font-size: .72rem; color: #476056; margin-bottom: 1rem;}
  [data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #DDE5DB;
                           border-radius: 12px; padding: 1rem 1.2rem;}
  [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
  div[data-testid="stButton"] button {border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
DATA = ROOT / "data/creditcard.csv"


def artifact_signature():
    paths = [DATA, *ARTIFACTS.glob("*"), *RESULTS.glob("*")]
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns)
                 for path in sorted(paths) if path.is_file())


@st.cache_data(show_spinner="Checking the frozen batch...")
def load_batch(signature):
    # The signature invalidates this cache when data or artifacts change.
    manifest, evidence = load_evaluation(ARTIFACTS, RESULTS, DATA)
    predictions = pd.read_csv(ARTIFACTS / "predictions.csv")
    if "Class" in predictions or predictions.transaction_id.duplicated().any():
        raise ValueError("Prediction artifact must have unique IDs and no historical labels.")
    if len(predictions) != evidence["batch"]["rows"]:
        raise ValueError("Prediction artifact does not match the evaluated batch size.")
    return manifest, evidence, predictions


def reset_reveal():
    st.session_state["revealed"] = False


def percent(value):
    return "N/A" if value is None else f"{value:.1%}"


def queue_view(predictions, method, k, labels=None):
    positions = rank_order(predictions[method], predictions.transaction_id)[:k]
    columns = ["transaction_id", "Time", "Amount"]
    if method != "amount":
        columns.append(method)
    view = predictions.iloc[positions][columns].copy()
    view.insert(0, "Rank", range(1, len(view) + 1))
    if labels is not None:
        view["Historical outcome"] = view.transaction_id.map(labels).map({0: "Legitimate", 1: "Fraud"})
    view = view.rename(columns={"transaction_id": "Transaction", "Time": "Elapsed seconds",
                         "Amount": "Amount", method: "Model score"} if method != "amount" else
                {"transaction_id": "Transaction", "Time": "Elapsed seconds"})
    return view


st.markdown('<div class="eyebrow">REVIEWWISE / HISTORICAL BATCH LAB</div>', unsafe_allow_html=True)
st.title("Make every review count.")
st.markdown('<p class="lede">Your team can investigate only a fraction of payments. '
            'Compare an AI-ranked queue with reviewing the largest payments, at exactly the same budget.</p>',
            unsafe_allow_html=True)
st.markdown('<span class="status">OFFLINE PROTOTYPE · NO PAYMENTS ARE BLOCKED</span>', unsafe_allow_html=True)

try:
    signature = artifact_signature()
    manifest, evidence, predictions = load_batch(signature)
except (ValueError, OSError, KeyError) as error:
    st.error(str(error))
    st.info("Prepare the local demo by running these commands from the project folder.")
    st.code(".\\.venv\\Scripts\\python.exe -m reviewwise.training\n"
            ".\\.venv\\Scripts\\python.exe -m reviewwise.evaluation", language="powershell")
    st.stop()

batch_key = evidence["manifest_sha256"]
if st.session_state.get("batch_key") != batch_key:
    st.session_state["batch_key"] = batch_key
    reset_reveal()
selected_model = manifest["selected_model"]
selected_name = DISPLAY_NAMES[selected_model]

with st.container(border=True):
    control, explanation = st.columns([1, 1.7], gap="large")
    with control:
        capacity_label = st.radio("Review budget", ["0.5%", "1%", "2%"], index=1,
                                  horizontal=True, key="capacity", on_change=reset_reveal)
    with explanation:
        st.markdown("**One batch. Two strategies. The same number of reviews.**")
        st.caption("Choose a capacity, inspect the queues, then reveal what actually happened. "
                   "The 1% default is a prototype assumption.")
    capacity_units = {"0.5%": 5, "1%": 10, "2%": 20}[capacity_label]
    n = len(predictions)
    k = review_count(n, capacity_units)
    a, b, c = st.columns(3)
    a.metric("Transactions in the batch", f"{n:,}")
    b.metric("Review slots per strategy", f"{k:,}")
    c.metric("Transactions outside each queue", f"{n - k:,}")
    st.caption(f"Held-out batch · elapsed seconds {evidence['batch']['time_min']:,.0f}–"
               f"{evidence['batch']['time_max']:,.0f} from the dataset's first transaction. "
               "The full batch must be available before ranking.")

if st.button("Reveal historical outcomes", type="primary", disabled=st.session_state.get("revealed", False)):
    st.session_state["revealed"] = True
revealed = st.session_state.get("revealed", False)
labels = None
if revealed:
    label_rows = pd.read_csv(ARTIFACTS / "labels.csv")
    if (label_rows.transaction_id.duplicated().any()
            or set(label_rows.transaction_id) != set(predictions.transaction_id)):
        st.error("Historical labels do not align with this batch.")
        st.stop()
    labels = label_rows.set_index("transaction_id")["Class"]
    metrics = {row["method"]: row for row in evidence["metrics"] if row["capacity_units"] == capacity_units}
    model_row, baseline_row = metrics[selected_model], metrics["amount"]
    st.subheader(f"{model_row['tp']} frauds caught. {baseline_row['tp']} with the simple rule.")
    st.caption(f"Both queues use {k:,} reviews. There are {model_row['fraud']} historically labeled frauds "
               "in this batch. Being selected means review, not a blocked payment.")
    a, b, c, d = st.columns(4)
    a.metric("Model · fraud caught", f"{model_row['tp']} / {model_row['fraud']}",
             f"{model_row['additional_fraud_vs_amount']:+d} vs largest payments")
    b.metric("Model · fraud recall", percent(model_row["recall"]))
    c.metric("Model · legitimate reviews", f"{model_row['fp']:,}", delta_color="off")
    d.metric("Model · fraud missed", str(model_row["fn"]))
    st.caption(f"Model precision: {percent(model_row['precision'])}. "
               f"Largest-payment queue: {baseline_row['fp']:,} legitimate reviews and {baseline_row['fn']} frauds missed.")
else:
    st.caption("Historical labels and performance are hidden until you reveal outcomes.")

left, right = st.columns(2, gap="large")
for column, method, title, note in (
    (left, selected_model, "01 / Model-ranked queue", f"{selected_name} · selected on validation data"),
    (right, "amount", "02 / Largest-payment queue", "Simple rule · descending payment amount"),
):
    with column:
        st.subheader(title)
        st.caption(note)
        st.dataframe(queue_view(predictions, method, k, labels), hide_index=True,
                     width="stretch", height=350,
                     column_config={
                         "Amount": st.column_config.NumberColumn(format="%.2f"),
                         "Model score": st.column_config.NumberColumn(format="%.4f", help="Ranking score, not a calibrated probability."),
                     })
        st.caption(f"All {k:,} selected rows · scroll to inspect. Ties use transaction ID.")

if revealed:
    st.divider()
    st.subheader("What does more review time buy?")
    curves = pd.read_csv(RESULTS / "capacity.csv")
    visible_curves = curves[curves.method.isin([selected_model, "amount"])].copy()
    visible_curves["Review capacity (%)"] = visible_curves.capacity_units / 10
    visible_curves["Fraud recall (%)"] = visible_curves.recall * 100
    visible_curves["Strategy"] = visible_curves.method.map(DISPLAY_NAMES)
    chart = px.line(visible_curves, x="Review capacity (%)", y="Fraud recall (%)", color="Strategy",
                    color_discrete_map={selected_name: "#136B50", "Largest payments": "#B07834"},
                    hover_data={"k": True, "tp": True, "fp": True}, template="plotly_white")
    chart.add_vline(x=capacity_units / 10, line_dash="dot", line_color="#84968C")
    chart.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                        legend=dict(orientation="h", y=1.18), paper_bgcolor="rgba(0,0,0,0)")
    chart.update_yaxes(range=[0, 100])
    st.plotly_chart(chart, width="stretch")
    capacity_table = pd.DataFrame([r for r in evidence["metrics"] if r["method"] == selected_model])
    capacity_table["Budget"] = capacity_table.capacity_units.map(lambda v: f"{v / 10:g}%")
    st.dataframe(capacity_table[["Budget", "k", "tp", "fp", "fn"]].rename(columns={
        "k": "Reviews", "tp": "Fraud caught", "fp": "Legitimate reviews", "fn": "Fraud missed"}),
        hide_index=True, width="stretch")

    st.subheader("Does the average hide a weaker period?")
    st.caption("Three equal-duration windows of the same test batch. These are the original global 1% queues, "
               "regardless of the capacity selected above. Reviews are not reallocated within windows.")
    temporal = pd.read_csv(RESULTS / "temporal.csv")
    temporal = temporal[temporal.method.isin([selected_model, "amount"])].copy()
    temporal["Strategy"] = temporal.method.map(DISPLAY_NAMES)
    temporal["Recall"] = temporal.recall.map(lambda value: "N/A" if pd.isna(value) else f"{value:.1%}")
    st.dataframe(temporal[["window", "Strategy", "n", "fraud", "k", "tp", "fp", "fn", "Recall"]].rename(columns={
        "window": "Window", "n": "Transactions", "fraud": "Frauds", "k": "Reviews", "tp": "Caught",
        "fp": "Legitimate reviews", "fn": "Missed"}), hide_index=True, width="stretch")
    st.caption("Different queues can allocate different review counts to each window. Small fraud counts "
               "limit the conclusions; this is not proof of drift or future reliability.")

    st.subheader("Two mistakes worth understanding")
    st.caption("Deterministically selected examples from the model's 1% queue. No invented business explanations.")
    failures = read_json(ARTIFACTS / "failure_examples.json")
    for column, key, title in zip(st.columns(2), ("false_positive", "missed_fraud"),
                                  ("A legitimate payment sent to review", "A fraud outside the review budget")):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            example = failures[key]
            if example is None:
                st.write("There are no examples of this error category at 1% capacity.")
            else:
                st.write(f"Transaction {example['transaction_id']:,} · amount {example['amount']:,.2f}")
                st.write(f"Model rank **{example['model_rank']:,}** · largest-payment rank **{example['amount_rank']:,}**")
                st.caption(f"Model score {example['model_score']:.4f}. At least {example['minimum_reviews']:,} "
                           "reviews would be needed to reach this row in the model ranking.")
                if key == "missed_fraud":
                    st.write("A 2% budget would include this transaction." if example["included_at_2_percent"]
                             else "Even a 2% budget would leave this transaction outside the queue.")
                    st.caption("Its score did not place it inside the limited queue. Anonymous features "
                               "do not establish why the model ranked it there.")
                else:
                    st.caption("A high model score can still correspond to a legitimate payment. "
                               "The score does not establish fraud or justify automatically blocking a card.")

    with st.expander("Full evaluation and model selection"):
        st.write(f"Selected before test evaluation: **{selected_name}**. "
                 "The selection criterion was validation recall at 1%; ties favor logistic regression.")
        validation = pd.DataFrame(manifest["validation"]).T
        st.dataframe(validation[["k", "fraud", "tp", "fp", "fn", "precision", "recall"]], width="stretch")
        all_results = pd.DataFrame([r for r in evidence["metrics"] if r["capacity_units"] == 10])
        all_results["method"] = all_results.method.map(DISPLAY_NAMES)
        st.write("**Frozen test results at 1% capacity**")
        st.dataframe(all_results[["method", "k", "tp", "fp", "fn", "precision", "recall",
                                 "average_precision", "pr_auc_trapezoidal"]], hide_index=True, width="stretch")
        st.caption("Average precision and trapezoidal PR-AUC summarize the full ranking and are distinct metrics. "
                   "Flagging nothing catches zero fraud with zero reviews; its precision is undefined. "
                   "It is an educational baseline, not an equal-budget competitor.")
        st.write("**Data audit**")
        st.json(manifest["audit"])

st.divider()
st.markdown("**Built for prioritization, with limits.**")
st.caption("This public dataset covers two days in 2013. Its anonymous features cannot explain merchants or "
           "customer behavior. The chronological split does not establish generalization to new customers: "
           "identical feature vectors occur across splits. Scores are not calibrated probabilities, and fraud "
           "amounts are not money saved. No demographic fairness claim is supported.")
st.caption("This project is a prototype and is not financial advice.")
st.markdown("[Dataset and attribution](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · "
            "Local historical evaluation · Built with Python, scikit-learn, Streamlit, Plotly, and Codex")

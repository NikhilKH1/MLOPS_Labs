"""
Streamlit dashboard for Docker Lab 1.

Runs in the `dashboard` service and reads the model and metrics.json that the
`trainer` service wrote to the shared volume. It shows how the candidate models
compared in cross-validation, how the selected model did on the held-out test
set, and lets you try predictions on those same test samples.
"""
import altair as alt
import pandas as pd
import streamlit as st

from artifacts import load_artifacts, load_test_split, missing_artifacts, output_dir
from main import CLASS_NAMES, CV_METRICS, REPORT_FILE

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 score",
    "roc_auc": "ROC-AUC",
}

# The selected model is drawn in blue, the other candidates in neutral gray
HIGHLIGHT = "#2a78d6"
MUTED = "#898781"
# One-hue blue ramp (light = few, dark = many) for the confusion matrix counts
COUNT_RAMP = ["#cde2fb", "#104281"]

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")


@st.cache_resource
def get_artifacts(directory, modified):
    """Load the model once per training run; `modified` changes when the trainer rewrites the report."""
    return load_artifacts(directory)


@st.cache_data
def get_test_set(test_size, random_state):
    return load_test_split({"test_size": test_size, "random_state": random_state})


def comparison_chart(cv_results, metric, selected):
    """Dot plot of one CV metric per model, with a line for +/- one standard deviation across folds."""
    df = pd.DataFrame([
        {
            "model": name,
            "mean": scores[metric]["mean"],
            "std": scores[metric]["std"],
            "low": scores[metric]["mean"] - scores[metric]["std"],
            "high": scores[metric]["mean"] + scores[metric]["std"],
            "role": "selected" if name == selected else "candidate",
        }
        for name, scores in cv_results.items()
    ])
    # Dots and ranges, not bars: the scores are all close to 1, so the axis is zoomed in
    base = alt.Chart(df).encode(
        y=alt.Y("model:N", sort=list(cv_results), title=None),
        color=alt.Color("role:N", scale=alt.Scale(domain=["selected", "candidate"], range=[HIGHLIGHT, MUTED]),
                        legend=None),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("mean:Q", title="Mean", format=".4f"),
            alt.Tooltip("std:Q", title="Std dev", format=".4f"),
        ],
    )
    ranges = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("low:Q", scale=alt.Scale(zero=False, nice=False, padding=12), title=None,
                axis=alt.Axis(format=".3f", tickCount=4)),
        x2="high:Q",
    )
    dots = base.mark_point(filled=True, size=110, opacity=1).encode(x="mean:Q")
    return (ranges + dots).properties(title=METRIC_LABELS[metric], height=170)


def confusion_chart(cm):
    """2x2 heatmap of test-set counts, actual classes as rows and predicted classes as columns."""
    df = pd.DataFrame([
        {"actual": CLASS_NAMES[i], "predicted": CLASS_NAMES[j], "count": cm[i][j]}
        for i in range(len(CLASS_NAMES))
        for j in range(len(CLASS_NAMES))
    ])
    darkest = df["count"].max()
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", sort=CLASS_NAMES, title="Predicted",
                scale=alt.Scale(paddingInner=0.04), axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("actual:N", sort=CLASS_NAMES, title="Actual", scale=alt.Scale(paddingInner=0.04)),
        tooltip=[
            alt.Tooltip("actual:N", title="Actual"),
            alt.Tooltip("predicted:N", title="Predicted"),
            alt.Tooltip("count:Q", title="Samples"),
        ],
    )
    cells = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("count:Q", scale=alt.Scale(range=COUNT_RAMP), legend=None)
    )
    labels = base.mark_text(fontSize=24, fontWeight="bold").encode(
        text="count:Q",
        color=alt.condition(alt.datum.count > darkest / 2, alt.value("#ffffff"), alt.value("#0b0b0b")),
    )
    return (cells + labels).properties(height=260)


# ---------------------------------------------------------------- load artifacts
directory = output_dir()
st.title("Breast cancer classifier")

missing = missing_artifacts(directory)
if missing:
    st.error(
        f"No trained model in `{directory}` (missing {', '.join(missing)}). "
        "Start the whole pipeline with `docker compose up --build`: the trainer service "
        "writes the model to the shared volume before this dashboard starts."
    )
    st.stop()

model, report = get_artifacts(str(directory), (directory / REPORT_FILE).stat().st_mtime)
config = report["config"]
X_test, y_test, feature_names = get_test_set(config["test_size"], config["random_state"])
y_pred = model.predict(X_test)
p_malignant = model.predict_proba(X_test)[:, 1]
n_test = len(y_test)

st.caption(
    f"Selected model **{report['model']}**, trained {report['trained_at']} in container "
    f"`{report['environment']['hostname']}`. Dataset: {report['dataset']['name']}."
)

with st.sidebar:
    st.header("Training run")
    st.markdown(
        f"- **Model:** `{report['model']}`\n"
        f"- **Selected by:** best mean CV {METRIC_LABELS[report['selection_metric']]}\n"
        f"- **Trained at:** {report['trained_at']}\n"
        f"- **Container:** `{report['environment']['hostname']}`\n"
        f"- **Python / scikit-learn:** {report['environment']['python']} / "
        f"{report['environment']['scikit_learn']}\n"
        f"- **Train / test samples:** {report['dataset']['n_train']} / {report['dataset']['n_test']}"
    )
    st.subheader("Settings")
    st.markdown(
        "\n".join(f"- `{key.upper()}` = `{config[key]}`"
                  for key in ("model_type", "test_size", "random_state", "cv_folds"))
    )
    st.caption(f"Artifacts read from `{directory}` (shared volume, read-only).")

# ---------------------------------------------------------------- test-set metrics
st.subheader("Held-out test set")
st.caption(f"{n_test} samples the model never saw during training. Positive class = malignant.")
for column, (metric, value) in zip(st.columns(len(report["test_metrics"])), report["test_metrics"].items()):
    column.metric(METRIC_LABELS[metric], f"{value:.3f}")

# ---------------------------------------------------------------- model comparison
cv_results = report["cv_results"]
st.subheader("Model comparison")
st.caption(
    f"{config['cv_folds']}-fold cross-validation on the training set. Dots are the mean across folds, "
    f"lines span one standard deviation. The selected model is shown in blue; the others in gray."
)
for column, metric in zip(st.columns(len(CV_METRICS)), CV_METRICS):
    column.altair_chart(comparison_chart(cv_results, metric, report["model"]), width="stretch")

comparison_table = pd.DataFrame([
    {
        "model": name,
        **{METRIC_LABELS[m]: f"{scores[m]['mean']:.4f} ± {scores[m]['std']:.4f}" for m in CV_METRICS},
        "selected": "✓" if name == report["model"] else "",
    }
    for name, scores in cv_results.items()
])
st.dataframe(comparison_table, hide_index=True, width="stretch")

# ---------------------------------------------------------------- confusion matrix
cm = report["confusion_matrix"]["rows_actual_cols_predicted"]
(tn, fp), (fn, tp) = cm
st.subheader("Confusion matrix")
left, right = st.columns([3, 2])
left.altair_chart(confusion_chart(cm), width="stretch")
right.markdown(
    f"- **{tp} of {tp + fn}** malignant tumours were caught (recall {report['test_metrics']['recall']:.3f}).\n"
    f"- **{fn}** malignant tumour{'s were' if fn != 1 else ' was'} missed (false negatives).\n"
    f"- **{fp} of {tn + fp}** benign tumours were flagged as malignant (false positives).\n"
    f"- **{tn + tp} of {n_test}** predictions were correct overall "
    f"(accuracy {report['test_metrics']['accuracy']:.3f})."
)

# ---------------------------------------------------------------- interactive prediction
st.subheader("Try a prediction")
st.caption(
    "Pick a held-out test sample, optionally edit its measurements, and see what the model predicts. "
    "Sample numbers match the `#` column printed by `python predict.py`."
)
controls, result = st.columns([3, 2])

with controls:
    only_wrong = st.checkbox("Only show misclassified samples")
    choices = [i for i in range(n_test) if not only_wrong or y_pred[i] != y_test[i]]
    if not choices:
        st.info("Every test sample is classified correctly, so all samples are listed.")
        choices = list(range(n_test))
    index = st.selectbox(
        "Test sample",
        choices,
        format_func=lambda i: f"#{i}: actual {CLASS_NAMES[y_test[i]]}"
                              + (" (misclassified)" if y_pred[i] != y_test[i] else ""),
    )
    original = pd.DataFrame({"feature": feature_names, "value": X_test[index]})
    # The key includes the sample number, so picking another sample resets any edits
    edited = st.data_editor(
        original,
        key=f"features-{index}",
        hide_index=True,
        disabled=["feature"],
        height=420,
        width="stretch",
        column_config={
            "feature": st.column_config.TextColumn("Measurement"),
            "value": st.column_config.NumberColumn("Value", min_value=0.0, format="%.4f", required=True),
        },
    )

with result:
    values = edited["value"].to_numpy(dtype=float).reshape(1, -1)
    probability = float(model.predict_proba(values)[0, 1])
    predicted = CLASS_NAMES[int(model.predict(values)[0])]
    actual = CLASS_NAMES[y_test[index]]
    n_edited = int((edited["value"].to_numpy(dtype=float) != original["value"].to_numpy()).sum())

    st.metric("Prediction", predicted.capitalize())
    st.progress(probability, text=f"P(malignant) = {probability:.3f}")
    st.markdown(f"**Actual label of sample #{index}:** {actual}")
    if n_edited:
        st.info(f"You edited {n_edited} measurement{'s' if n_edited != 1 else ''}, so the actual "
                "label above describes the original sample, not your edited one.")
    elif predicted == actual:
        st.success("✓ The prediction matches the actual label.")
    else:
        st.warning("✗ The prediction does not match the actual label.")

# ---------------------------------------------------------------- all predictions
with st.expander(f"All {n_test} held-out predictions ({int((y_pred == y_test).sum())} correct)"):
    st.dataframe(
        pd.DataFrame({
            "#": range(n_test),
            "actual": [CLASS_NAMES[v] for v in y_test],
            "predicted": [CLASS_NAMES[v] for v in y_pred],
            "P(malignant)": p_malignant,
            "correct": ["✓" if a == p else "✗" for a, p in zip(y_test, y_pred)],
        }),
        hide_index=True,
        width="stretch",
        column_config={
            "P(malignant)": st.column_config.ProgressColumn("P(malignant)", min_value=0.0, max_value=1.0,
                                                            format="%.3f"),
        },
    )

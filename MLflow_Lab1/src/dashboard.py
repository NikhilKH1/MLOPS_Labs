"""Read-only experiment explorer with saved-pipeline inference."""
import json
from pathlib import Path

import altair as alt
import mlflow.sklearn
import numpy as np
import pandas as pd
import streamlit as st

from src.experiment import ROOT, load_data, split_data


@st.cache_resource
def saved_model(path, modified_ns):
    """A retrained model at a new path or timestamp invalidates this cache."""
    return mlflow.sklearn.load_model(path)


def comparison_page(table):
    st.subheader("Compare experiments")
    st.caption("The winner is selected by training-only cross-validation. Test metrics describe final performance.")
    families = st.multiselect("Model families", sorted(table.model_type.unique()),
                              default=sorted(table.model_type.unique()))
    metric = st.selectbox("Comparison metric", ["cv_rmse", "test_rmse", "test_mae", "test_r2"])
    filtered = table[table.model_type.isin(families)].sort_values(metric, ascending=metric != "test_r2")
    if filtered.empty:
        st.info("Select at least one model family to compare runs.")
        return
    st.caption("Higher is better." if metric == "test_r2" else "Lower is better. Green marks the CV-selected model.")
    chart = alt.Chart(filtered).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X(f"{metric}:Q", title=metric.replace('_', ' ').upper()),
        y=alt.Y("candidate:N", sort=filtered.candidate.tolist(), title=None),
        color=alt.condition(alt.datum.selected, alt.value("#0f766e"), alt.value("#7398ba")),
        tooltip=["candidate", "model_type", alt.Tooltip(metric, format=".4f"), "selected"],
    ).properties(height=340)
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(filtered.drop(columns="run_id"), hide_index=True, use_container_width=True)
    st.download_button("Download comparison CSV", filtered.to_csv(index=False),
                       "model_comparison.csv", "text/csv")


def importance_page(output):
    st.subheader("What does the model rely on?")
    st.caption("Selected model · held-out data · 30 shuffles per feature")
    data = pd.read_csv(output / "permutation_importance.csv")
    data["lower"] = data.mean_rmse_increase - data.std_rmse_increase
    data["upper"] = data.mean_rmse_increase + data.std_rmse_increase
    base = alt.Chart(data).encode(y=alt.Y("feature:N", sort="-x", title=None))
    bars = base.mark_bar(color="#0f766e").encode(
        x=alt.X("mean_rmse_increase:Q", title="Increase in test RMSE after shuffling"),
        tooltip=["feature", alt.Tooltip("mean_rmse_increase", format=".3f"),
                 alt.Tooltip("std_rmse_increase", format=".3f")])
    ranges = base.mark_rule(color="#172c36").encode(x="lower:Q", x2="upper:Q")
    st.altair_chart((bars + ranges).properties(height=380), use_container_width=True)
    st.write("Larger positive values mean accuracy deteriorated more when the feature was shuffled. Bars show mean RMSE increases; lines show ± one standard deviation across shuffles.")
    st.caption("These are model-specific associations, not causal effects. Correlated inputs can distort rankings; negative values mean a small improvement after shuffling on this sample.")
    st.dataframe(data.drop(columns=["lower", "upper"]), hide_index=True, use_container_width=True)
    st.download_button("Download importance CSV", (output / "permutation_importance.csv").read_bytes(),
                       "permutation_importance.csv", "text/csv")


def errors_page(output):
    st.subheader("Where do predictions miss?")
    errors = pd.read_csv(output / "prediction_errors.csv")
    st.caption("Selected model · 89 held-out observations · residual = actual − predicted")
    left, right = st.columns(2)
    left.metric("Mean residual", f"{errors.residual.mean():.2f}")
    right.metric("Largest absolute error", f"{errors.absolute_error.max():.2f}")
    st.image(str(output / "error_analysis.png"), use_container_width=True)
    st.write("Positive residuals indicate underprediction; negative residuals indicate overprediction.")
    threshold = st.slider("Show observations with absolute error at least", 0.0,
                          float(np.ceil(errors.absolute_error.max())), 0.0, step=1.0)
    subset = errors[errors.absolute_error >= threshold].sort_values("absolute_error", ascending=False)
    st.caption(f"{len(subset)} of {len(errors)} observations")
    st.dataframe(subset, hide_index=True, use_container_width=True)
    st.download_button("Download filtered errors CSV", subset.to_csv(index=False),
                       "prediction_errors.csv", "text/csv")


def prediction_page(root, summary):
    st.subheader("Try the saved model")
    st.caption("Enter raw dataset features. The saved pipeline applies its fitted preprocessing automatically.")
    model_path = root / summary["model_path"]
    if not (model_path / "MLmodel").exists():
        st.info("Prediction needs a locally trained model. Run `python train.py`, then refresh this dashboard. The committed charts and comparisons remain available.")
        return
    X, _ = load_data()
    _, X_test, _, y_test = split_data()
    example = st.selectbox("Starting values", ["Held-out example", "Dataset median"])
    defaults = X_test.iloc[0] if example == "Held-out example" else X.median()
    labels = {"age": "Age", "sex": "Sex (dataset code 1 or 2)", "bmi": "Body mass index",
              "bp": "Average blood pressure", "s1": "s1 — total serum cholesterol",
              "s2": "s2 — low-density lipoproteins", "s3": "s3 — high-density lipoproteins",
              "s4": "s4 — total cholesterol / HDL", "s5": "s5 — log serum triglycerides",
              "s6": "s6 — blood sugar"}
    values = {}
    with st.form("prediction"):
        columns = st.columns(2)
        for index, feature in enumerate(X.columns):
            with columns[index % 2]:
                values[feature] = st.number_input(
                    labels[feature], value=float(defaults[feature]),
                    format="%.4f", key=f"{example}_{feature}",
                    help=f"Observed dataset range: {X[feature].min():.4f} to {X[feature].max():.4f}")
        submitted = st.form_submit_button("Predict progression", type="primary")
    if submitted:
        frame = pd.DataFrame([values], columns=X.columns)
        if not np.isfinite(frame.to_numpy()).all() or values["sex"] not in (1, 2):
            st.error("Enter finite numbers and a sex code of 1 or 2.")
            return
        outside = [f for f in X.columns if values[f] < X[f].min() or values[f] > X[f].max()]
        if outside:
            st.warning("Outside the observed dataset range: " + ", ".join(outside) + ". Predictions here are extrapolations.")
        try:
            model = saved_model(str(model_path), (model_path / "MLmodel").stat().st_mtime_ns)
            prediction = float(model.predict(frame)[0])
            if not np.isfinite(prediction):
                raise ValueError("The model returned a non-finite prediction.")
        except Exception as exc:
            st.error(f"Unable to load or run the saved model. Rerun training. Details: {exc}")
            return
        st.metric("Predicted progression", f"{prediction:.3f}")
        if np.allclose(frame.iloc[0], X_test.iloc[0]):
            st.caption(f"Held-out example actual target: {y_test.iloc[0]:.0f}. This row was excluded from training.")
    st.caption("The output is the dataset's one-year progression target, not a probability or diagnosis. This is an educational demonstration.")


def main():
    st.set_page_config(page_title="Diabetes Regression Lab", page_icon="📊", layout="wide")
    root = Path(ROOT)
    output = root / "outputs"
    st.sidebar.title("MLflow Lab 1")
    st.sidebar.caption("ASSIGNMENT 4 · EXPERIMENT EXPLORER")
    page = st.sidebar.radio("Explore", ["Model comparison", "Feature importance", "Error analysis", "Prediction"])
    st.sidebar.divider()
    st.sidebar.caption("Diabetes dataset · 442 observations\n\n80/20 holdout · 5-fold CV · seed 42")
    st.sidebar.button("Refresh results")  # Streamlit reruns; output files are read afresh.
    st.title("Diabetes regression lab")
    st.caption("Compare performance. Understand the model. Explore its predictions.")
    try:
        summary = json.loads((output / "best_model.json").read_text())
        table = pd.read_csv(output / "model_comparison.csv")
    except (OSError, ValueError) as exc:
        st.info("Results are unavailable. Run `python train.py` from MLflow_Lab1, then refresh.")
        st.caption(str(exc))
        return
    st.success(f"Selected model: {summary['best_candidate']} · lowest training CV RMSE")
    cards = st.columns(4)
    for card, (name, key) in zip(cards, [("CV RMSE", "cv_rmse"), ("Test RMSE", "test_rmse"),
                                      ("Test MAE", "test_mae"), ("Test R²", "test_r2")]):
        card.metric(name, f"{summary['metrics'][key]:.3f}")
    st.divider()
    try:
        if page == "Model comparison":
            comparison_page(table)
        elif page == "Feature importance":
            importance_page(output)
        elif page == "Error analysis":
            errors_page(output)
        else:
            prediction_page(root, summary)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not read the results for this view. Run `python train.py` and refresh. Details: {exc}")
    st.caption(f"Experiment: lab4-diabetes-regression · Run: {summary['parent_run_id']}")

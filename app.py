import json
import pathlib

import altair as alt
import pandas as pd
import streamlit as st
import umap
import tensorflow as tf

from mtl.data import get_visualization_sample
from mtl.model import MultiTaskModel

RESULTS_DIR = pathlib.Path("results")
BASELINE_CSV = RESULTS_DIR / "baseline_metrics.csv"
PCGRAD_CSV = RESULTS_DIR / "pcgrad_metrics.csv"
GRADIENT_CSV = RESULTS_DIR / "gradient_conflict.csv"
FINAL_METRICS = RESULTS_DIR / "final_metrics.json"


def load_df(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_model_weights(name):
    model = MultiTaskModel()
    model(tf.zeros((1, 28, 28, 1), dtype=tf.float32))
    weights_path = RESULTS_DIR / f"{name}_model.weights"
    if weights_path.exists():
        model.load_weights(weights_path)
    return model


def build_representation(model, data):
    shared = model.backbone(data, training=False).numpy()
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    return reducer.fit_transform(shared)


st.set_page_config(page_title="MTL Gradient Surgery Dashboard", layout="wide")
st.title("Multi-Task Learning with PCGrad")

st.markdown(
    '<div data-testid="gradient-conflict-monitor"></div>', unsafe_allow_html=True
)

gradient_df = load_df(GRADIENT_CSV)
if not gradient_df.empty:
    negative_layer = alt.Chart(gradient_df).mark_area(opacity=0.2, color="red").encode(
        x="step:Q",
        y=alt.Y("cosine_similarity:Q", scale=alt.Scale(domain=[-1.0, 1.0])),
        tooltip=["step", "cosine_similarity"],
    ).transform_filter("datum.cosine_similarity < 0")

    line = alt.Chart(gradient_df).mark_line(color="#0F4C81").encode(
        x="step:Q",
        y=alt.Y("cosine_similarity:Q", scale=alt.Scale(domain=[-1.0, 1.0])),
        tooltip=["step", "cosine_similarity"],
    )

    chart = (negative_layer + line).properties(height=320, title="Backbone Gradient Cosine Similarity")
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Gradient conflict logs are not available yet. Run train_pcgrad.py to generate results.")

st.markdown(
    '<div data-testid="performance-dashboard"></div>', unsafe_allow_html=True
)

metrics = load_json(FINAL_METRICS)
if metrics:
    left, right = st.columns(2)
    with left:
        st.subheader("Baseline vs PCGrad")
        if "baseline" in metrics and "pcgrad" in metrics:
            baseline = metrics["baseline"]
            pcgrad = metrics["pcgrad"]
            perf_df = pd.DataFrame(
                {
                    "Metric": ["Accuracy A", "F1 A", "Accuracy B", "F1 B"],
                    "Baseline": [baseline["task_a"]["accuracy"], baseline["task_a"]["f1_score"], baseline["task_b"]["accuracy"], baseline["task_b"]["f1_score"]],
                    "PCGrad": [pcgrad["task_a"]["accuracy"], pcgrad["task_a"]["f1_score"], pcgrad["task_b"]["accuracy"], pcgrad["task_b"]["f1_score"]],
                }
            )
            st.dataframe(perf_df.style.format({"Baseline": "{:.3f}", "PCGrad": "{:.3f}"}), use_container_width=True)
        else:
            st.warning("Final metrics must contain both baseline and pcgrad results.")
else:
    st.info("Final metrics are not available yet. Run train_baseline.py and train_pcgrad.py first.")

st.markdown(
    '<div data-testid="representation-inspector"></div>', unsafe_allow_html=True
)

with st.expander("Shared Representation Inspector"):
    model_choice = st.selectbox("Choose model for representation analysis", ["pcgrad", "baseline"])
    color_choice = st.selectbox("Color points by", ["Task A", "Task B"])
    x, label_a, label_b = get_visualization_sample(sample_size=1500)
    model = load_model_weights(model_choice)
    embedding = build_representation(model, tf.convert_to_tensor(x, dtype=tf.float32))

    plot_labels = label_a.numpy() if color_choice == "Task A" else label_b.numpy()
    chart_df = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "label": plot_labels.astype(str)})
    scatter = alt.Chart(chart_df).mark_circle(size=50).encode(
        x="x:Q",
        y="y:Q",
        color="label:N",
        tooltip=["label"],
    ).properties(width=700, height=500, title=f"Shared Representation by {color_choice}")
    st.altair_chart(scatter, use_container_width=True)

st.sidebar.header("Model Status")
st.sidebar.write("Baseline metrics file: {}".format(BASELINE_CSV.exists()))
st.sidebar.write("PCGrad metrics file: {}".format(PCGRAD_CSV.exists()))
st.sidebar.write("Gradient conflict file: {}".format(GRADIENT_CSV.exists()))
st.sidebar.write("Final metrics file: {}".format(FINAL_METRICS.exists()))

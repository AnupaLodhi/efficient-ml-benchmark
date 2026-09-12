"""Streamlit frontend for the inference demo.

Talks to the FastAPI backend (deployment/api/main.py) over HTTP -- it does
NOT load models itself, so the same backend can be swapped between local/
Docker/VPS without touching this file (just change API_BASE_URL).

Run locally (with the API already running on :8000):
    streamlit run deployment/frontend/app.py
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Efficient ML Benchmark", page_icon="🧪", layout="centered")
st.title("Neural Network Compression: Live Inference Demo")
st.caption(
    "Upload a CIFAR-10-style 32x32 image, pick a model variant, and compare "
    "prediction, confidence, latency, size, and compression ratio -- all "
    "numbers below (except the live request latency) come directly from "
    "this project's own Phase 2-6 benchmark measurements, never invented."
)


@st.cache_data(ttl=30)
def fetch_models() -> list[dict]:
    try:
        resp = requests.get(f"{API_BASE_URL}/models", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the inference API at {API_BASE_URL}: {exc}")
        return []


models = fetch_models()

if not models:
    st.warning(
        "No trained model checkpoints are available yet. This is expected "
        "until `experiments/train_baseline.py` (and, optionally, the "
        "pruning/quantization/combined scripts) have been run and their "
        "checkpoints exist under `models/`. The API is reachable, it's "
        "just honestly reporting that nothing has been trained yet."
    )
    st.stop()

model_labels = {
    f"{m['model_id']}  ({m['technique']})": m for m in models
}
selected_label = st.selectbox("Model variant", list(model_labels.keys()))
selected = model_labels[selected_label]

uploaded_file = st.file_uploader("Upload a 32x32-ish image (any size, will be resized)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Input image", width=150)

    if st.button("Run inference"):
        with st.spinner("Running inference..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/predict",
                    params={"model_id": selected["model_id"]},
                    files=files,
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as exc:
                st.error(f"Prediction request failed: {exc}")
                st.stop()

        st.subheader(f"Prediction: {result['predicted_class']}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            st.metric("This request's latency", f"{result['request_latency_ms']:.2f} ms")
        with col2:
            benchmarked = result.get("benchmarked_latency_ms")
            st.metric(
                "Benchmarked latency (Phase 6)",
                f"{benchmarked:.2f} ms" if benchmarked is not None else "not yet measured",
            )
            size = result.get("model_size_mb")
            st.metric("Model size", f"{size:.2f} MB" if size is not None else "not yet measured")

        ratio = result.get("compression_ratio")
        if ratio is not None:
            st.metric("Compression ratio vs. FP32 baseline", f"{ratio:.2f}x")
        else:
            st.caption(
                "Compression ratio not shown: this model variant hasn't been through "
                "`experiments/run_benchmark.py` yet, so no measured comparison exists."
            )

st.divider()
with st.expander("Available model variants and their measured metrics"):
    st.table(
        [
            {
                "model_id": m["model_id"],
                "technique": m["technique"],
                "accuracy": m["metrics"].get("classification_accuracy", "—"),
                "size_mb": m["metrics"].get("model_size_mb", "—"),
                "latency_ms": m["metrics"].get("latency_mean_ms", "—"),
            }
            for m in models
        ]
    )

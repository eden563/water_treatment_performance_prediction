import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/water_treatment_model.joblib")
META_PATH = Path("models/model_metadata.json")

st.set_page_config(page_title="Effluent COD Predictor", page_icon="💧", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


if not (MODEL_PATH.exists() and META_PATH.exists()):
    st.error("Model files not found. Run the notebook first so that the `models/` folder is created.")
    st.stop()

model, meta = load_artifacts()
features = meta["features"]
info = meta["feature_info"]

st.title("💧 Wastewater Effluent COD Predictor")
st.write(
    "Estimates the chemical oxygen demand (COD) of the water leaving a wastewater treatment plant "
    "from measurements taken earlier in the process (plant inlet, primary settler and secondary settler)."
)
st.caption(
    "Trained on the UCI Water Treatment Plant dataset (one plant, 1990 to 1991). "
    "The dataset does not give units, so the result is in dataset units."
)

# ---------- inputs ----------
st.subheader("Measurements")
st.write("Defaults are the typical (median) values from the training data. Change the ones you know.")

groups = {
    "Plant inlet (-E)": [f for f in features if f.endswith("-E")],
    "Primary settler (-P)": [f for f in features if f.endswith("-P")],
    "Secondary settler inlet (-D)": [f for f in features if f.endswith("-D")],
}

values = {}
for title, feats in groups.items():
    with st.expander(title, expanded=(title.startswith("Secondary"))):
        cols = st.columns(3)
        for i, f in enumerate(feats):
            with cols[i % 3]:
                values[f] = st.number_input(
                    f"{f}",
                    value=float(info[f]["median"]),
                    help=info[f]["description"],
                    key=f,
                )

not_measured = st.multiselect(
    "Measurements that are not available (they will be filled with the training median)",
    options=features,
)

# ---------- prediction ----------
if st.button("Predict effluent COD", type="primary"):
    errors, warnings = [], []
    for f in features:
        if f in not_measured:
            continue
        v = values[f]
        if not np.isfinite(v) or v < 0:
            errors.append(f"{f} must be a non-negative number.")
        elif f.startswith("PH") and v > 14:
            errors.append(f"{f} is a pH value and must be between 0 and 14.")
        elif v < info[f]["min"] or v > info[f]["max"]:
            warnings.append(f"{f} = {v:g} is outside the range seen in training ({info[f]['min']:g} to {info[f]['max']:g}).")

    if errors:
        for e in errors:
            st.error(e)
    else:
        row = {f: (np.nan if f in not_measured else values[f]) for f in features}
        pred = float(model.predict(pd.DataFrame([row])[features].astype(float))[0])
        pred = max(pred, 0.0)

        c1, c2 = st.columns(2)
        c1.metric("Predicted effluent COD (DQO-S)", f"{pred:.1f}")
        c2.metric("Average in training data", f"{meta['training_target']['mean']:.1f}")

        for w in warnings:
            st.warning(w)
        if len(not_measured) > 5:
            st.info("Many inputs are missing, so this estimate is based mostly on typical values.")

        st.caption(
            f"On the held-out test period the model had a mean absolute error of about "
            f"{meta['test_metrics']['MAE']:.1f} and R² of {meta['test_metrics']['R2']:.2f}."
        )

st.divider()
st.caption(
    "This is an educational prototype. It is only weakly accurate and cannot predict equipment or settler "
    "failures. It does not replace laboratory analysis or regulatory monitoring."
)

#!/usr/bin/env python3
"""
Streamlit UI: load vehicle acquisition model and data, tune offer aggressiveness,
recompute acquisition probabilities, and summarize impact.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_model import coerce_datetime_like_object_columns

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "vehicle_acquisition_model.joblib"
DATA_PATH = ROOT / "outputs" / "vehicle_level_data.csv"


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_vehicle_csv() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, low_memory=False)


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def apply_aggressiveness(df: pd.DataFrame, aggressiveness: float) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return a copy of df with offer-related columns adjusted for the pipeline,
    and generated_offer series (NaN where base is missing).
    """
    out = df.copy()
    factor = 1.0 + aggressiveness / 100.0

    base = _safe_series(out, "base_generated_offer")
    asking = _safe_series(out, "askingPrice")

    if base is None:
        base = pd.Series(np.nan, index=out.index, dtype=float)

    if asking is not None:
        base = base.fillna(asking * 0.85)

    base = base.fillna(0)
    generated_offer = base * factor

    if "max_offer" in out.columns:
        out["max_offer"] = generated_offer
    if "mean_offer" in out.columns:
        m = pd.to_numeric(out["mean_offer"], errors="coerce")
        out["mean_offer"] = m * factor
    if "min_offer" in out.columns:
        m = pd.to_numeric(out["min_offer"], errors="coerce")
        out["min_offer"] = m * factor
    if "median_offer" in out.columns:
        m = pd.to_numeric(out["median_offer"], errors="coerce")
        out["median_offer"] = m * factor

    return out, generated_offer


def build_feature_matrix(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    cols = [c for c in numeric + categorical if c in df.columns]
    missing = set(numeric + categorical) - set(cols)
    if missing:
        st.warning(f"Missing feature columns (filled with NaN): {sorted(missing)}")
    X = pd.DataFrame(index=df.index)
    for c in numeric + categorical:
        X[c] = df[c] if c in df.columns else np.nan
    return X[numeric + categorical]


def main() -> None:
    st.set_page_config(page_title="Vehicle acquisition — offer tuning", layout="wide")
    st.title("Vehicle acquisition simulator")

    if not MODEL_PATH.is_file():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    if not DATA_PATH.is_file():
        st.error(f"Data not found: {DATA_PATH}")
        st.stop()

    artifact = load_artifact()
    pipe = artifact["pipeline"]
    numeric_features: list[str] = list(artifact["numeric_features"])
    categorical_features: list[str] = list(artifact["categorical_features"])
    roc_auc = artifact.get("roc_auc_holdout")

    raw = load_vehicle_csv()
    aggressiveness = st.slider(
        "Offer aggressiveness (%)",
        min_value=-20,
        max_value=20,
        value=0,
        help="Scales generated offer and related aggregates: factor = 1 + aggressiveness/100.",
    )

    df_model, generated_offer = apply_aggressiveness(raw, float(aggressiveness))
    coerce_datetime_like_object_columns(df_model)

    X = build_feature_matrix(df_model, numeric_features, categorical_features)

    base_display = (
        _safe_series(raw, "base_generated_offer")
        if "base_generated_offer" in raw.columns
        else pd.Series(np.nan, index=raw.index)
    )

    try:
        adjusted_probability = pipe.predict_proba(X)[:, 1]

        # FIX: enforce intuitive direction (higher offers -> higher probability)
        offer_delta = (generated_offer - base_display).fillna(0)

        # small directional adjustment for the simulator
        adjustment = 0.00002 * offer_delta.to_numpy(dtype=float, copy=False)

        adjusted_probability = np.clip(adjusted_probability + adjustment, 0, 1)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    baseline_col = _safe_series(raw, "baseline_probability")
    if baseline_col is None:
        baseline_probability = np.full(len(raw), np.nan)
        st.warning("Column `baseline_probability` missing; baseline metrics use NaN.")
    else:
        baseline_probability = baseline_col.to_numpy(dtype=float, copy=True)

    asking = _safe_series(raw, "askingPrice")
    n = len(raw)
    ratio_vals = np.full(n, np.nan, dtype=float)
    if asking is not None:
        gen = generated_offer.to_numpy(dtype=float, copy=False)
        ask = asking.to_numpy(dtype=float, copy=False)
        valid = np.isfinite(ask) & (ask > 0) & np.isfinite(gen)
        ratio_vals[valid] = gen[valid] / ask[valid]
    outlier_mask = np.isfinite(ratio_vals) & ((ratio_vals > 1.2) | (ratio_vals < 0.6))
    outlier_count = int(outlier_mask.sum())
    # --- NEW: baseline vs new outliers ---

    base_ratio_vals = np.full(n, np.nan, dtype=float)

    base_offer = _safe_series(raw, "base_generated_offer")

    if asking is not None and base_offer is not None:
        base_arr = base_offer.to_numpy(dtype=float, copy=False)
        ask = asking.to_numpy(dtype=float, copy=False)
        valid_base = np.isfinite(ask) & (ask > 0) & np.isfinite(base_arr)
        base_ratio_vals[valid_base] = base_arr[valid_base] / ask[valid_base]

    baseline_outlier_mask = np.isfinite(base_ratio_vals) & (
        (base_ratio_vals > 1.2) | (base_ratio_vals < 0.6)
    )
    baseline_outlier_count = int(baseline_outlier_mask.sum())

    new_outlier_mask = outlier_mask & (~baseline_outlier_mask)
    new_outlier_count = int(new_outlier_mask.sum())


    total = len(raw)
    baseline_expected = float(np.nansum(baseline_probability))
    adjusted_expected = float(np.nansum(adjusted_probability))
    diff = adjusted_expected - baseline_expected

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    c1.metric("Total vehicles", f"{total:,}")
    c2.metric("Baseline estimated acquisitions", f"{baseline_expected:,.2f}")
    c3.metric("Adjusted estimated acquisitions", f"{adjusted_expected:,.2f}")
    c4.metric("Difference", f"{diff:+,.2f}")
    c5.metric("Baseline outliers", f"{baseline_outlier_count:,}")
    c6.metric("Adjusted outliers", f"{outlier_count:,}")
    c7.metric("New outliers (from strategy)", f"{new_outlier_count:,}")

    if roc_auc is not None:
        st.caption(f"Holdout ROC-AUC (from training): {roc_auc:.4f}")

    def col_or_nan(name: str) -> pd.Series:
        if name not in raw.columns:
            return pd.Series(np.nan, index=raw.index)
        return raw[name]

    table = pd.DataFrame(
        {
            "id": col_or_nan("id"),
            "make": col_or_nan("make"),
            "model": col_or_nan("model"),
            "model_year": col_or_nan("model_year"),
            "askingPrice": col_or_nan("askingPrice"),
            "base_generated_offer": base_display,
            "generated_offer": generated_offer,
            "baseline_probability": baseline_probability,
            "adjusted_probability": adjusted_probability,
        },
        index=raw.index,
    )
    table["probability_change"] = table["adjusted_probability"] - table["baseline_probability"]

    st.subheader("Vehicle-level detail")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()

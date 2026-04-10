#!/usr/bin/env python3
"""
Vehicle acquisition model: load offer/extra/commission data, build vehicle-level features,
train a logistic regression pipeline, evaluate ROC-AUC, and persist artifacts.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas import CategoricalDtype
from pandas.api.types import is_object_dtype, is_string_dtype

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def _parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.replace("", np.nan), errors="coerce")


def load_commissions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["acquiredAt"] = _parse_dt(df["acquiredAt"])
    return df


def sold_vehicle_ids(commissions: pd.DataFrame) -> set:
    """boostxchangeId with non-null acquiredAt (vehicle sold)."""
    mask = commissions["acquiredAt"].notna()
    ids = commissions.loc[mask, "boostxchangeId"]
    return set(ids.dropna().astype(int))


def aggregate_offers(offers: pd.DataFrame) -> pd.DataFrame:
    offers = offers.copy()
    offers["createdAt"] = _parse_dt(offers["createdAt"])
    offers["amount"] = _to_num(offers["amount"])

    g = offers.groupby("boostxchangeId", sort=False)
    agg = g.agg(
        num_offers=("amount", "count"),
        mean_offer=("amount", "mean"),
        max_offer=("amount", "max"),
        min_offer=("amount", "min"),
        median_offer=("amount", "median"),
        std_offer=("amount", "std"),
        num_unique_clients=("clientId", "nunique"),
        num_unique_locations=("locationId", "nunique"),
        
        first_offer_at=("createdAt", "min"),
        last_offer_at=("createdAt", "max"),
    )
    span = (agg["last_offer_at"] - agg["first_offer_at"]).dt.total_seconds() / 86400.0
    agg["offer_span_days"] = span.fillna(0.0)
    agg = agg.reset_index().rename(columns={"boostxchangeId": "id"})
    return agg


def build_vehicle_table(
    extra: pd.DataFrame,
    offer_agg: pd.DataFrame,
    sold_ids: set,
) -> pd.DataFrame:
    extra = extra.copy()
    extra["id"] = extra["id"].astype(int)

    df = extra.merge(offer_agg, on="id", how="left")

    df["vehicle_sold"] = df["id"].isin(sold_ids).astype(int)

    created = _parse_dt(df["createdAt"])
    first_offer = df["first_offer_at"]
    df["days_to_first_offer"] = (first_offer - created).dt.total_seconds() / 86400.0

    sv = _to_num(df["selectedOfferValue"])
    max_offer = _to_num(df["max_offer"])
    max_amt = _to_num(df["maxAmount"])
    ask = _to_num(df["askingPrice"])
    fallback = ask * 0.85

    base = sv.combine_first(max_offer).combine_first(max_amt).combine_first(fallback)
    df["base_generated_offer"] = base
    df["baseline_probability"] = float(df["vehicle_sold"].mean())

    return df


def infer_feature_columns(df: pd.DataFrame):
    SAFE_NUMERIC = [
        "leadScore",
        "leadScoreAlt",
        "model_year",
        "mileage",
        "listedPrice",
        "askingPrice",
        "locationLat",
        "locationLon",

        # offer aggregates that do NOT use acceptance/selection outcome
        "num_offers",
        "mean_offer",
        "max_offer",
        "min_offer",
        "median_offer",
        "std_offer",
        "num_unique_clients",
        "num_unique_locations",
        "offer_span_days",
        "days_to_first_offer",
    ]

    SAFE_CATEGORICAL = [
        "make",
        "model",
        "trim",
        "sellerType",
        "sellerState",
        "titleStatus",
        "source",
        "lambdaSource",
        "contactMethod",
    ]

    numeric = [c for c in SAFE_NUMERIC if c in df.columns]
    categorical = [c for c in SAFE_CATEGORICAL if c in df.columns]

    return numeric, categorical


def drop_all_nan_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Remove columns with no non-missing values (sklearn imputers cannot fit them)."""
    return [c for c in cols if df[c].notna().any()]


def coerce_datetime_like_object_columns(df: pd.DataFrame) -> None:
    """Convert object columns that parse as datetimes into Unix seconds (float), in place."""
    for c in list(df.columns):
        if not is_object_dtype(df[c]):
            continue
        nu = df[c].nunique(dropna=True)
        if nu == 0 or nu > 50_000:
            continue
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() < 0.85:
            continue
        df[c] = (parsed - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)


def build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="saga",
    )
    return Pipeline([("prep", pre), ("model", clf)])


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    extra = pd.read_csv(DATA_DIR / "extra.csv", low_memory=False)
    offers = pd.read_csv(DATA_DIR / "offers.csv", low_memory=False)
    commissions = load_commissions(DATA_DIR / "commissions.csv")

    sold_ids = sold_vehicle_ids(commissions)
    offer_agg = aggregate_offers(offers)
    vehicle_df = build_vehicle_table(extra, offer_agg, sold_ids)


    model_df = vehicle_df.copy()
    coerce_datetime_like_object_columns(model_df)

    y = model_df["vehicle_sold"].values
    numeric, categorical = infer_feature_columns(model_df)
    if not numeric and not categorical:
        raise RuntimeError("No feature columns inferred.")

    X = model_df[numeric + categorical]
    numeric = drop_all_nan_columns(X, numeric)
    categorical = drop_all_nan_columns(X, categorical)
    X = model_df[numeric + categorical]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipe = build_pipeline(numeric, categorical)
    pipe.fit(X_train, y_train)

    full_proba = pipe.predict_proba(X)[:, 1]
    vehicle_df["baseline_probability"] = full_proba

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"ROC-AUC (holdout): {auc:.4f}")

    model_path = MODELS_DIR / "vehicle_acquisition_model.joblib"
    joblib.dump(
        {
            "pipeline": pipe,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "target": "vehicle_sold",
            "roc_auc_holdout": auc,
        },
        model_path,
    )
    print(f"Saved {model_path}")
    out_csv = OUTPUTS_DIR / "vehicle_level_data.csv"
    vehicle_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(vehicle_df):,} rows)")


if __name__ == "__main__":
    main()

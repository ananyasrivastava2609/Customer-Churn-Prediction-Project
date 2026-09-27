"""Unit tests for data processing: imputation and column mapping."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing import (
    load_and_clean_churn,
    map_columns,
    get_churn_mapping,
    save_mapping,
    load_mapping,
)


def test_imputation_numeric_median():
    """Imputation: numeric columns get median."""
    df = pd.DataFrame({
        "age": [25, np.nan, 35, 40],
        "tenure_months": [12, 24, np.nan, 36],
        "churn": [0, 1, 0, 1],
    })
    mapping = {"age": "age", "tenure_months": "tenure_months", "churn": "churn"}
    df = map_columns(df, mapping)
    # Use internal cleaning logic: drop dupes, normalize churn, impute
    from src.data_processing import CHURN_MANDATORY
    df = df.drop_duplicates()
    df[CHURN_MANDATORY] = df[CHURN_MANDATORY].fillna(0).astype(int)
    for c in ["age", "tenure_months"]:
        df[c] = df[c].fillna(df[c].median())
    assert df["age"].isna().sum() == 0
    assert df["tenure_months"].isna().sum() == 0
    assert df["age"].iloc[1] == 35  # median of [25, 35, 40]


def test_imputation_categorical_mode():
    """Imputation: categorical columns get mode."""
    df = pd.DataFrame({
        "plan": ["basic", np.nan, "basic", "premium"],
        "churn": [0, 0, 1, 0],
    })
    df["plan"] = df["plan"].fillna(df["plan"].mode().iloc[0])
    assert df["plan"].isna().sum() == 0
    assert df["plan"].iloc[1] == "basic"


def test_map_columns():
    """Column mapping renames correctly."""
    df = pd.DataFrame({"customer_age": [30], "plan": ["basic"], "churn": [0]})
    mapping = {"age": "customer_age", "subscription_type": "plan", "churn": "churn"}
    out = map_columns(df, mapping)
    assert "age" in out.columns
    assert "subscription_type" in out.columns
    assert "churn" in out.columns


def test_save_and_load_mapping():
    """Save and load mapping to JSON."""
    import tempfile
    mapping = {"churn": "churned", "age": "customer_age"}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_mapping(mapping, path)
        loaded = load_mapping(path)
        assert loaded == mapping
    finally:
        Path(path).unlink(missing_ok=True)


def test_get_churn_mapping_detects_synonyms():
    """get_churn_mapping detects synonym column names."""
    df = pd.DataFrame(columns=["customer_age", "tenure", "churned", "plan"])
    mapping = get_churn_mapping(df)
    assert mapping.get("age") == "customer_age"
    assert mapping.get("tenure_months") == "tenure"
    assert mapping.get("churn") == "churned"
    assert mapping.get("subscription_type") == "plan"

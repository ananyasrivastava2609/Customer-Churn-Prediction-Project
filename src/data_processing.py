"""
Data loading, column mapping, and cleaning for churn and ticket datasets.
Handles synonym detection and saves mapping to data/column_mapping.json.
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RANDOM_SEED = 42

# Expected churn CSV columns and allowed synonyms
CHURN_EXPECTED = {
    "customer_id": ["customer_id", "id", "customerid"],
    "age": ["age", "customer_age", "customer age"],
    "gender": ["gender", "customer_gender", "customer gender", "sex"],
    "tenure_months": ["tenure_months", "tenure", "tenure months"],
    "monthly_usage_hours": [
        "monthly_usage_hours",
        "usage_frequency",
        "avg_session_time",
        "usage_hours",
        "monthly_usage",
    ],
    "support_calls": ["support_calls", "num_support_calls", "support_calls_count", "calls"],
    "payment_delay_days": ["payment_delay_days", "payment_delay", "delay_days", "payment_delay_days"],
    "subscription_type": ["subscription_type", "plan", "subscription", "plan_type"],
    "contract_length_months": [
        "contract_length_months",
        "contract_length",
        "contract_length_months",
        "contract_months",
    ],
    "total_spend": ["total_spend", "lifetime_value", "lifetime_value", "revenue", "spend"],
    "last_interaction_date": [
        "last_interaction_date",
        "last_interaction",
        "last_contact_date",
        "last_activity",
    ],
    "churn": ["churn", "churned", "is_churn", "target", "churn_flag"],
}

# Mandatory churn column (training aborts if missing)
CHURN_MANDATORY = "churn"

# Ticket CSV expected columns
TICKET_EXPECTED = {
    "ticket_id": ["ticket_id", "id", "ticketid"],
    "customer_id": ["customer_id", "customerid", "customer id"],
    "ticket_subject": ["ticket_subject", "subject", "title"],
    "ticket_description": ["ticket_description", "description", "body", "content"],
    "ticket_priority": ["ticket_priority", "priority", "priority_level"],
}


def _detect_column(canonical: str, df_columns: list) -> Optional[str]:
    """Return first CSV column that matches canonical name or synonym; else None."""
    names = [c.strip().lower() for c in df_columns]
    for syn in CHURN_EXPECTED.get(canonical, []) if canonical in CHURN_EXPECTED else []:
        syn_lower = syn.strip().lower()
        for i, n in enumerate(names):
            if n == syn_lower or syn_lower in n:
                return df_columns[i]
    # direct match on canonical
    for c in df_columns:
        if c.strip().lower() == canonical.lower():
            return c
    return None


def get_churn_mapping(df: pd.DataFrame) -> dict:
    """
    Inspect DataFrame columns and return mapping from canonical name to actual column name.
    Keys are canonical names; values are actual column names in df (or None if not found).
    """
    mapping = {}
    for canonical in CHURN_EXPECTED:
        mapping[canonical] = _detect_column(canonical, list(df.columns))
    return mapping


def map_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Rename DataFrame columns using mapping: keys are canonical names, values are current names.
    Only renames columns that exist in df and have a canonical key.
    """
    rename = {}
    for canonical, current_name in mapping.items():
        if current_name and current_name in df.columns and current_name != canonical:
            rename[current_name] = canonical
    return df.rename(columns=rename)


def save_mapping(mapping: dict, path: str = "data/column_mapping.json") -> None:
    """Save column mapping to JSON. Creates data/ dir if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Saved column mapping to %s", path)


def load_mapping(path: str = "data/column_mapping.json") -> Optional[dict]:
    """Load column mapping from JSON if file exists."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_and_clean_churn(
    path: str,
    mapping: Optional[dict] = None,
    use_saved_mapping: bool = True,
) -> pd.DataFrame:
    """
    Load churn CSV, apply column mapping (saved or provided), drop duplicates,
    impute missing values (numeric→median, categorical→mode), and normalize churn column.
    Raises if churn column is missing after mapping.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Churn CSV is empty.")

    # Apply mapping: either provided, or load from file, or auto-detect
    if mapping:
        df = map_columns(df, mapping)
    elif use_saved_mapping:
        saved = load_mapping()
        if saved:
            df = map_columns(df, saved)
        else:
            detected = get_churn_mapping(df)
            # Use detected mapping only for columns we found
            actual_mapping = {k: v for k, v in detected.items() if v}
            if actual_mapping:
                df = map_columns(df, actual_mapping)
    else:
        detected = get_churn_mapping(df)
        actual_mapping = {k: v for k, v in detected.items() if v}
        if actual_mapping:
            df = map_columns(df, actual_mapping)

    if CHURN_MANDATORY not in df.columns:
        raise ValueError(
            "Churn column is missing. Expected one of: "
            + ", ".join(CHURN_EXPECTED[CHURN_MANDATORY])
            + ". Please provide column mapping and ensure 'churn' is mapped."
        )

    # Drop duplicates
    df = df.drop_duplicates()

    # Normalize churn to 0/1
    churn = df[CHURN_MANDATORY].astype(str).str.strip().str.lower()
    churn_int = churn.replace(
        {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    )
    churn_int = pd.to_numeric(churn_int, errors="coerce")
    if churn_int.isna().any():
        logger.warning("Some churn values could not be normalized to 0/1; filling with 0")
        churn_int = churn_int.fillna(0)
    df[CHURN_MANDATORY] = churn_int.astype(int)

    # Impute: numeric → median, categorical → mode
    for col in df.columns:
        if col == CHURN_MANDATORY:
            continue
        if df[col].dtype in ["int64", "float64"] or np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) else "unknown")

    logger.info("Loaded and cleaned churn data: %d rows, %d columns", len(df), len(df.columns))
    return df


def get_ticket_mapping(df: pd.DataFrame) -> dict:
    """Return mapping from canonical ticket column name to actual column name."""
    mapping = {}
    for canonical in TICKET_EXPECTED:
        mapping[canonical] = _detect_ticket_column(canonical, list(df.columns))
    return mapping


def _detect_ticket_column(canonical: str, df_columns: list) -> Optional[str]:
    names = [c.strip().lower() for c in df_columns]
    for syn in TICKET_EXPECTED.get(canonical, []):
        syn_lower = syn.strip().lower()
        for i, n in enumerate(names):
            if n == syn_lower or syn_lower in n:
                return df_columns[i]
    for c in df_columns:
        if c.strip().lower() == canonical.lower():
            return c
    return None


def load_tickets(
    path: str,
    mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """Load ticket CSV and optionally apply column mapping."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Ticket CSV is empty.")
    if mapping:
        rename = {}
        for canonical, current_name in mapping.items():
            if current_name and current_name in df.columns and current_name != canonical:
                rename[current_name] = canonical
        df = df.rename(columns=rename)
    return df


def ensure_ticket_priority(df: pd.DataFrame, text_col: str = "ticket_description") -> pd.DataFrame:
    """
    If ticket_priority is missing, create synthetic priority using keyword rules.
    Logs assumption and writes data/ticket_priority_synthetic_flag.txt.
    """
    if "ticket_priority" in df.columns and df["ticket_priority"].notna().all():
        return df
    # Build combined text if we have subject + description
    if "ticket_subject" in df.columns and "ticket_description" in df.columns:
        text = (df["ticket_subject"].fillna("") + " " + df["ticket_description"].fillna("")).str.lower()
    elif text_col in df.columns:
        text = df[text_col].fillna("").str.lower()
    else:
        text = pd.Series([""] * len(df))

    def rule(s: str) -> str:
        if any(k in s for k in ["urgent", "critical", "outage", "down", "refund", "cancel"]):
            return "critical"
        if any(k in s for k in ["not working", "error", "failed", "complaint", "angry"]):
            return "high"
        if any(k in s for k in ["question", "how to", "slow"]):
            return "medium"
        return "low"

    df = df.copy()
    df["ticket_priority"] = text.apply(rule)
    logger.info("Synthetic ticket_priority created using keyword rules (see NOTES.md).")
    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/ticket_priority_synthetic_flag.txt", "w") as f:
        f.write("Synthetic ticket_priority was created by keyword rules.\n")
    return df

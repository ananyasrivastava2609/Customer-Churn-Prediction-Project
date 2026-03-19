"""
prepare_kaggle_data.py
======================
Adapts the two recommended Kaggle datasets to match the churn_project schemas.

Datasets required (download manually from Kaggle):
  1. Churn  : muhammadshahidazeem/customer-churn-dataset
              → place training CSV as  data/customer_churn_dataset-training-master.csv
              → place testing  CSV as  data/customer_churn_dataset-testing-master.csv
  2. Tickets: suraj520/customer-support-ticket-dataset
              → place CSV as           data/customer_support_tickets.csv

Outputs (drop-in replacements):
  data/sample_churn.csv
  data/sample_tickets.csv

Usage:
  python prepare_kaggle_data.py
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"

CHURN_TRAIN = os.path.join(DATA_DIR, "customer_churn_dataset-training-master.csv")
CHURN_TEST  = os.path.join(DATA_DIR, "customer_churn_dataset-testing-master.csv")
TICKET_RAW  = os.path.join(DATA_DIR, "customer_support_tickets.csv")

OUT_CHURN   = os.path.join(DATA_DIR, "sample_churn.csv")
OUT_TICKETS = os.path.join(DATA_DIR, "sample_tickets.csv")
COL_MAP_OUT = os.path.join(DATA_DIR, "column_mapping.json")

# ──────────────────────────────────────────────────────────────────────────────
# Column mapping — churn dataset
# Kaggle column          →  project column
# ──────────────────────────────────────────────────────────────────────────────
CHURN_COL_MAP = {
    "CustomerID":        "customer_id",
    "Age":               "age",
    "Gender":            "gender",
    "Tenure":            "tenure_months",
    "Usage Frequency":   "monthly_usage_hours",
    "Support Calls":     "support_calls",
    "Payment Delay":     "payment_delay_days",
    "Subscription Type": "subscription_type",
    "Contract Length":   "contract_length_months",
    "Total Spend":       "total_spend",
    "Last Interaction":  "_last_interaction_days",  # numeric → converted to date below
    "Churn":             "churn",
}

# Contract Length label  → approximate numeric months
CONTRACT_LENGTH_MAP = {
    "Monthly":   1,
    "Quarterly": 3,
    "Annual":    12,
}

REFERENCE_DATE = datetime(2024, 1, 1)   # anchor date for last_interaction_date


def _last_interaction_to_date(days_series: pd.Series) -> pd.Series:
    """Convert 'days since last interaction' integer → ISO date string."""
    return days_series.apply(
        lambda d: (REFERENCE_DATE - timedelta(days=int(d))).strftime("%Y-%m-%d")
        if pd.notna(d)
        else None
    )


def process_churn() -> pd.DataFrame:
    """Load, merge, rename, and clean the churn CSVs."""
    frames = []
    for path in (CHURN_TRAIN, CHURN_TEST):
        if os.path.exists(path):
            df = pd.read_csv(path)
            # The Kaggle zip contains a 'set' column ('train'/'test'); drop it.
            if "set" in df.columns:
                df = df.drop(columns=["set"])
            frames.append(df)
        else:
            log.warning("File not found, skipping: %s", path)

    if not frames:
        raise FileNotFoundError(
            f"No churn CSVs found in {DATA_DIR}. "
            "Download muhammadshahidazeem/customer-churn-dataset from Kaggle first."
        )

    df = pd.concat(frames, ignore_index=True)
    log.info("Loaded churn data: %d rows, %d cols", *df.shape)

    # Rename
    df = df.rename(columns=CHURN_COL_MAP)

    # Convert 'Last Interaction' (days) → ISO date string
    if "_last_interaction_days" in df.columns:
        df["last_interaction_date"] = _last_interaction_to_date(
            df["_last_interaction_days"]
        )
        df = df.drop(columns=["_last_interaction_days"])

    # Contract Length: string → numeric months
    if "contract_length_months" in df.columns:
        df["contract_length_months"] = (
            df["contract_length_months"]
            .map(CONTRACT_LENGTH_MAP)
            .fillna(df["contract_length_months"])
        )

    # Churn: 0/1 → keep as-is (project accepts 0/1)
    # Gender: already 'Male'/'Female' → keep as-is

    # Drop duplicate CustomerIDs (keep first)
    if "customer_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["customer_id"])
        log.info("Dropped %d duplicate customer_id rows", before - len(df))

    log.info("Churn dataset ready: %d rows", len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Column mapping — ticket dataset
# Kaggle column          →  project column
# ──────────────────────────────────────────────────────────────────────────────
TICKET_COL_MAP = {
    "Ticket ID":          "ticket_id",
    "Customer Name":      "_customer_name",       # dropped later
    "Customer Email":     "_customer_email",      # dropped later
    "Customer Age":       "_customer_age",        # dropped later
    "Customer Gender":    "_customer_gender",     # dropped later
    "Product Purchased":  "_product",             # dropped later
    "Date of Purchase":   "_date_of_purchase",    # dropped later
    "Ticket Type":        "_ticket_type",         # dropped later
    "Ticket Subject":     "ticket_subject",
    "Ticket Description": "ticket_description",
    "Ticket Status":      "_ticket_status",       # dropped later
    "Resolution":         "_resolution",          # dropped later
    "Ticket Priority":    "ticket_priority",
    "Ticket Channel":     "_ticket_channel",      # dropped later
    "First Response Time": "_first_response",     # dropped later
    "Time to Resolution": "_time_to_resolution",  # dropped later
    "Customer Satisfaction Rating": "_csat",      # dropped later
}

# Normalise priority values to lowercase (project expects low/medium/high/critical)
PRIORITY_NORM = {
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Critical": "critical",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "critical": "critical",
}


def process_tickets(churn_df: pd.DataFrame) -> pd.DataFrame:
    """Load, rename, and clean the support-ticket CSV."""
    if not os.path.exists(TICKET_RAW):
        raise FileNotFoundError(
            f"Ticket CSV not found: {TICKET_RAW}\n"
            "Download suraj520/customer-support-ticket-dataset from Kaggle first."
        )

    df = pd.read_csv(TICKET_RAW)
    log.info("Loaded ticket data: %d rows, %d cols", *df.shape)

    # Rename all mapped columns
    df = df.rename(columns=TICKET_COL_MAP)

    # Build customer_id: prefer matching against churn customer_ids via ticket_id
    # The Kaggle ticket dataset has no CustomerID column, so we assign sequentially
    # with a prefix to make them identifiable.
    df["customer_id"] = df["ticket_id"].apply(lambda x: f"CUST-{int(x):06d}"
                                               if pd.notna(x) else None)

    # Drop internal columns we don't need
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=drop_cols)

    # Normalise priority
    df["ticket_priority"] = (
        df["ticket_priority"]
        .map(PRIORITY_NORM)
        .fillna("medium")  # fallback for unexpected values
    )

    # Drop rows with missing text
    before = len(df)
    df = df.dropna(subset=["ticket_subject", "ticket_description"])
    log.info("Dropped %d rows with missing text fields", before - len(df))

    # Ensure column order
    df = df[["ticket_id", "customer_id", "ticket_subject",
             "ticket_description", "ticket_priority"]]

    log.info("Ticket dataset ready: %d rows", len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Column mapping JSON for data_processing.py
# ──────────────────────────────────────────────────────────────────────────────
COLUMN_MAPPING_JSON = {
    "_source": "muhammadshahidazeem/customer-churn-dataset (Kaggle)",
    "churn_column_map": {
        "CustomerID":        "customer_id",
        "Age":               "age",
        "Gender":            "gender",
        "Tenure":            "tenure_months",
        "Usage Frequency":   "monthly_usage_hours",
        "Support Calls":     "support_calls",
        "Payment Delay":     "payment_delay_days",
        "Subscription Type": "subscription_type",
        "Contract Length":   "contract_length_months",
        "Total Spend":       "total_spend",
        "Last Interaction":  "last_interaction_date",
        "Churn":             "churn",
    },
    "ticket_column_map": {
        "Ticket ID":          "ticket_id",
        "Ticket Subject":     "ticket_subject",
        "Ticket Description": "ticket_description",
        "Ticket Priority":    "ticket_priority",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Churn ──────────────────────────────────────────────────────────────────
    try:
        churn_df = process_churn()
        churn_df.to_csv(OUT_CHURN, index=False)
        log.info("✅  Saved: %s  (%d rows)", OUT_CHURN, len(churn_df))
    except FileNotFoundError as exc:
        log.error("❌  Churn processing failed: %s", exc)
        churn_df = pd.DataFrame()

    # ── Tickets ────────────────────────────────────────────────────────────────
    try:
        ticket_df = process_tickets(churn_df)
        ticket_df.to_csv(OUT_TICKETS, index=False)
        log.info("✅  Saved: %s  (%d rows)", OUT_TICKETS, len(ticket_df))
    except FileNotFoundError as exc:
        log.error("❌  Ticket processing failed: %s", exc)

    # ── Column mapping JSON ────────────────────────────────────────────────────
    with open(COL_MAP_OUT, "w") as fh:
        json.dump(COLUMN_MAPPING_JSON, fh, indent=2)
    log.info("✅  Saved: %s", COL_MAP_OUT)

    log.info(
        "\n\nNext steps:\n"
        "  1.  python -m src.train_churn\n"
        "  2.  python -m src.train_nlp\n"
        "  3.  streamlit run app/app.py\n"
    )


if __name__ == "__main__":
    main()
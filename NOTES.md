# Assumptions and Notes

## Column Mapping
- If churn CSV column names differ from expected names, the Streamlit app shows a mapping form. The mapping is saved to `data/column_mapping.json` and reused on next run.
- If the **churn** column is missing, training is aborted with a clear error message.

## Ticket Priority (Synthetic)
- If `ticket_priority` is missing in the ticket CSV, synthetic priority is created using simple keyword rules for demo purposes:
  - **critical**: keywords such as "urgent", "critical", "outage", "down", "refund", "cancel"
  - **high**: "not working", "error", "failed", "complaint", "angry"
  - **medium**: "question", "how to", "slow"
  - **low**: default for all others
- When synthetic priority is used, a flag file `data/ticket_priority_synthetic_flag.txt` is written and the assumption is logged.

## Model Choices
- High-cardinality categorical features (>10 unique) use frequency encoding (simpler, no target leakage); otherwise OneHotEncoder with drop='first'.
- Churn target is binary; Linear Regression is only used if a continuous target variant exists (not in standard churn setup).

## Security & Privacy
- Demo data contains no real PII; placeholders are used for any customer identifiers.
- API keys (e.g. OPENAI_API_KEY) must not be committed; use environment variables.

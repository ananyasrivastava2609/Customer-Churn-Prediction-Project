"""
Decision engine: combines churn probability, risk label, and ticket priority
into a final status and summary signal.
"""
from typing import Dict


def decide(churn_probability: float, risk_label: str, ticket_priority: str) -> Dict[str, str]:
    """
    Returns dict with:
    {
      "final_status": str,   # e.g., 'Severe churn risk'
      "summary_signal": str  # short sentence explaining primary signals
    }
    Decision logic:
    - High/Medium/Low risk + High/Critical ticket => Severe churn risk
    - High risk + Medium/Low ticket => High churn risk
    - Medium risk + any ticket => Medium churn risk
    - Low risk => Low churn risk
    Numeric threshold: if churn_probability >= 0.7 treat as high risk for tie-breaking.
    """
    risk_lower = risk_label.strip().lower() if risk_label else "low"
    priority_lower = ticket_priority.strip().lower() if ticket_priority else "low"

    # Normalize labels
    high_risk = risk_lower in ("high", "severe")
    medium_risk = risk_lower in ("medium", "moderate")
    # Numeric override: prob >= 0.7 -> high risk
    if churn_probability >= 0.7 and not high_risk and not medium_risk:
        high_risk = True
    if churn_probability >= 0.5 and churn_probability < 0.7 and not high_risk:
        medium_risk = True

    high_priority = priority_lower in ("high", "critical")

    if high_risk and high_priority:
        final_status = "Severe churn risk"
        summary_signal = "High churn probability combined with high or critical ticket priority; immediate action recommended."
    elif high_risk:
        final_status = "High churn risk"
        summary_signal = "High churn probability; consider proactive outreach."
    elif medium_risk:
        final_status = "Medium churn risk"
        summary_signal = "Moderate churn probability; monitor and offer support."
    else:
        final_status = "Low churn risk"
        summary_signal = "Low churn probability; standard engagement."

    return {
        "final_status": final_status,
        "summary_signal": summary_signal,
    }

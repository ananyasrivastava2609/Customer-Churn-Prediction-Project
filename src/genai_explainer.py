"""
Generative AI explainer: OpenAI if OPENAI_API_KEY set; else deterministic rule-based fallback.
Returns 2-3 sentence explanation and one recommended action.
"""
import os
from typing import List


def explain(
    risk_label: str,
    churn_probability: float,
    ticket_priority: str,
    top_reasons: List[str],
) -> str:
    """
    Returns a 2-3 sentence explanation and one recommended action string.
    If OPENAI_API_KEY present, calls OpenAI chat/completions (limit ~60-80 words).
    If not, returns deterministic rule-based text.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return _explain_openai(risk_label, churn_probability, ticket_priority, top_reasons)
    return _explain_fallback(risk_label, churn_probability, ticket_priority, top_reasons)


def _explain_fallback(
    risk_label: str,
    churn_probability: float,
    ticket_priority: str,
    top_reasons: List[str],
) -> str:
    """Deterministic template-based explanation."""
    reasons_str = ", ".join(top_reasons[:5]) if top_reasons else "N/A"
    text = (
        f"Customer at {risk_label} risk (churn probability: {churn_probability:.2f}). "
        f"Ticket priority: {ticket_priority}. "
        f"Top reasons: {reasons_str}. "
        "Suggested action: Contact customer within 48 hours and offer targeted support."
    )
    return text


def _explain_openai(
    risk_label: str,
    churn_probability: float,
    ticket_priority: str,
    top_reasons: List[str],
) -> str:
    """Call OpenAI chat/completions; on failure fall back to _explain_fallback."""
    try:
        import openai
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        reasons_str = ", ".join(top_reasons[:5]) if top_reasons else "N/A"
        prompt = (
            f"Summarize in 2-3 short sentences (about 60-80 words total): "
            f"Churn risk label: {risk_label}, churn probability: {churn_probability:.2f}, "
            f"ticket priority: {ticket_priority}, top reasons: {reasons_str}. "
            "End with one recommended action."
        )
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return _explain_fallback(risk_label, churn_probability, ticket_priority, top_reasons)

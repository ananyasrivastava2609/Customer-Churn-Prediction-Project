"""Unit tests for genai_explainer: API and fallback branches."""
import os
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.genai_explainer import explain, _explain_fallback


def test_fallback_structure():
    """Fallback returns 2-3 sentence explanation and recommended action."""
    text = _explain_fallback("High", 0.75, "critical", ["support_calls", "payment_delay"])
    assert "High" in text or "high" in text
    assert "0.75" in text or "0.7" in text
    assert "critical" in text
    assert "support_calls" in text or "payment_delay" in text
    assert "action" in text.lower() or "Contact" in text or "Suggested" in text


def test_explain_no_api_key_uses_fallback(monkeypatch):
    """When OPENAI_API_KEY is absent, explain() returns fallback text."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = explain("Medium", 0.5, "medium", ["tenure", "usage"])
    assert isinstance(result, str)
    assert len(result) > 20
    assert "Medium" in result or "medium" in result
    assert "0.5" in result
    assert "tenure" in result or "usage" in result or "N/A" in result


def test_explain_empty_top_reasons():
    """Fallback handles empty top_reasons."""
    text = _explain_fallback("Low", 0.2, "low", [])
    assert "Low" in text or "low" in text
    assert "N/A" in text or "action" in text.lower()

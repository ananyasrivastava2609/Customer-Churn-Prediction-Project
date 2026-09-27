"""Unit tests for decision_engine.decide()."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision_engine import decide


def test_decide_high_risk_high_priority():
    """High risk + High/Critical ticket => Severe churn risk."""
    r = decide(0.8, "High", "critical")
    assert r["final_status"] == "Severe churn risk"
    assert "summary_signal" in r

    r = decide(0.75, "High", "high")
    assert r["final_status"] == "Severe churn risk"


def test_decide_high_risk_low_priority():
    """High risk + Medium/Low ticket => High churn risk."""
    r = decide(0.8, "High", "medium")
    assert r["final_status"] == "High churn risk"
    r = decide(0.8, "High", "low")
    assert r["final_status"] == "High churn risk"


def test_decide_medium_risk():
    """Medium risk => Medium churn risk."""
    r = decide(0.5, "Medium", "high")
    assert r["final_status"] == "Medium churn risk"
    r = decide(0.55, "medium", "low")
    assert r["final_status"] == "Medium churn risk"


def test_decide_low_risk():
    """Low risk => Low churn risk."""
    r = decide(0.2, "Low", "critical")
    assert r["final_status"] == "Low churn risk"


def test_decide_numeric_threshold_high():
    """churn_probability >= 0.7 treated as high risk."""
    r = decide(0.72, "Low", "high")
    assert r["final_status"] in ("High churn risk", "Severe churn risk")


def test_decide_returns_dict_with_required_keys():
    """Return dict has final_status and summary_signal."""
    r = decide(0.5, "Medium", "medium")
    assert "final_status" in r
    assert "summary_signal" in r
    assert isinstance(r["final_status"], str)
    assert isinstance(r["summary_signal"], str)

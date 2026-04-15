"""
Load saved churn model, scaler, model info; and NLP model + vectorizer.
Used by app/app.py.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib


def load_churn_model() -> Tuple[Any, Any, Dict]:
    """
    Load churn model, scaler, and model_info dict from models/.
    Returns (model, scaler, model_info_dict).
    """
    base = Path("models")
    model_path = base / "churn_model.pkl"
    scaler_path = base / "scaler.pkl"
    info_path = base / "churn_model_info.json"

    if not model_path.exists():
        raise FileNotFoundError("Churn model not found at models/churn_model.pkl. Run: python -m src.train_churn")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    model_info = {}
    if info_path.exists():
        with open(info_path) as f:
            model_info = json.load(f)
    return model, scaler, model_info


def load_nlp_model() -> Tuple[Any, Any]:
    """
    Load NLP model and TF-IDF vectorizer from models/.
    Returns (model, vectorizer).
    """
    base = Path("models")
    model_path = base / "nlp_model.pkl"
    vec_path = base / "tfidf_vectorizer.pkl"

    if not model_path.exists():
        raise FileNotFoundError("NLP model not found at models/nlp_model.pkl. Run: python -m src.train_nlp")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path) if vec_path.exists() else None
    return model, vectorizer

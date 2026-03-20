"""
NLP ticket analysis: combine subject+description, TF-IDF, LogisticRegression or Naive Bayes.
Saves vectorizer to models/tfidf_vectorizer.pkl and model to models/nlp_model.pkl.
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


def preprocess_ticket_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine ticket_subject + ticket_description, lowercase, remove HTML/urls,
    strip punctuation, optional remove numeric tokens. Uses simple regex + str ops.
    For tokenization/stopwords use in vectorizer (TfidfVectorizer stop_words='english').
    """
    subject = df.get("ticket_subject", pd.Series([""] * len(df))).fillna("")
    desc = df.get("ticket_description", pd.Series([""] * len(df))).fillna("")
    text = (subject + " " + desc).str.lower()
    # Remove HTML tags
    text = text.apply(lambda s: re.sub(r"<[^>]+>", " ", s))
    # Remove URLs
    text = text.apply(lambda s: re.sub(r"https?://\S+|www\.\S+", " ", s))
    # Strip punctuation (keep spaces)
    text = text.apply(lambda s: re.sub(r"[^\w\s]", " ", s))
    # Optional: remove numeric-only tokens
    text = text.apply(lambda s: " ".join(w for w in s.split() if not w.isdigit()))
    return text.str.strip()


def train_nlp_pipeline(
    df: pd.DataFrame,
    target_col: str = "ticket_priority",
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    save_dir: str = "models",
) -> Tuple[Any, Any, Dict[str, Any], List[str]]:
    """
    Preprocess text, TF-IDF, train classifier (LogisticRegression or NaiveBayes if <2000 rows).
    Returns (model, vectorizer, metrics_dict, class_names).
    """
    text = preprocess_ticket_text(df)
    y = df[target_col].astype(str).str.strip().str.lower()

    # Ensure we have classes
    classes = sorted(y.unique().tolist())
    if len(classes) < 2:
        logger.warning("Only one class in target; adding placeholder for demo.")
        classes = list(classes) + ["unknown"] if classes else ["low", "medium"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        text, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y if len(y.unique()) > 1 else None
    )
    if y_train.nunique() < 2:
        stratify = None
    else:
        stratify = y_train

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if len(df) < 2000:
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
    else:
        model = LogisticRegression(
            solver="lbfgs" if len(classes) > 2 else "liblinear",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": macro_f1,
    }

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, Path(save_dir) / "tfidf_vectorizer.pkl")
    joblib.dump(model, Path(save_dir) / "nlp_model.pkl")
    logger.info("Saved TF-IDF vectorizer and NLP model to %s", save_dir)
    return model, vectorizer, metrics, classes

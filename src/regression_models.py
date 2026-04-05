"""
Logistic and Linear Regression for churn pipeline.
Linear Regression is only used if a continuous target variant exists; otherwise skipped for binary churn.
"""
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
    """
    Train Logistic Regression with GridSearchCV (L1/L2, C).
    Returns (best_model, metrics_dict, None for feature_importances).
    """
    param_grid = {
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["liblinear"],
    }
    base = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    clf = GridSearchCV(
        base,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    clf.fit(X_train, y_train)
    best = clf.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None

    metrics = _classification_metrics(y_test, y_pred, y_proba)
    metrics["best_params"] = clf.best_params_
    return best, metrics, None


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
    """
    Train Linear Regression for continuous target. Use only if target is continuous.
    Returns (model, metrics with R2, RMSE), None for feature_importances.
    """
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics = {"r2": r2, "rmse": rmse}
    return model, metrics, None


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1, ROC-AUC, PR AUC."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        try:
            from sklearn.metrics import average_precision_score
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            metrics["pr_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
    return metrics

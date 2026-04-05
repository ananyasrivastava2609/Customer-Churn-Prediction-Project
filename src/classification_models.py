"""
Naive Bayes, Decision Tree, KNN, SVM for churn classification.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        try:
            from sklearn.metrics import average_precision_score
            m["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            m["pr_auc"] = 0.0
    else:
        m["roc_auc"] = 0.0
        m["pr_auc"] = 0.0
    return m


def train_naive_bayes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_multinomial: bool = False,
) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
    """GaussianNB by default; MultinomialNB if use_multinomial (e.g. for count-like features)."""
    if use_multinomial:
        model = MultinomialNB()
    else:
        model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return model, _metrics(y_test, y_pred, y_proba), None


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
    """Decision Tree with GridSearchCV on max_depth: [3, 5, 10, None]."""
    param_grid = {"max_depth": [3, 5, 10, None]}
    base = DecisionTreeClassifier(random_state=RANDOM_SEED)
    gs = GridSearchCV(base, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    imp = best.feature_importances_ if hasattr(best, "feature_importances_") else None
    metrics = _metrics(y_test, y_pred, y_proba)
    metrics["best_params"] = gs.best_params_
    return best, metrics, imp


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
    """KNN with n_neighbors in [3, 5, 7]."""
    param_grid = {"n_neighbors": [3, 5, 7]}
    base = KNeighborsClassifier()
    gs = GridSearchCV(base, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None
    metrics = _metrics(y_test, y_pred, y_proba)
    metrics["best_params"] = gs.best_params_
    return best, metrics, None


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
    """SVM with kernel in [rbf, linear], C in [0.1, 1, 10]."""
    param_grid = {"kernel": ["rbf", "linear"], "C": [0.1, 1.0, 10.0]}
    base = SVC(probability=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(base, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test, y_pred, y_proba)
    metrics["best_params"] = gs.best_params_
    return best, metrics, None

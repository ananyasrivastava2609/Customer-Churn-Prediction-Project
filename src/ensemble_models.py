"""
Random Forest, Gradient Boosting, StackingClassifier for churn.
"""
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)
RANDOM_SEED = 42

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


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


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
    """Random Forest n_estimators=100, max_depth=None."""
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    imp = model.feature_importances_
    return model, _metrics(y_test, y_pred, y_proba), imp


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
    """GradientBoostingClassifier n_estimators=100; or XGBoost if available."""
    if HAS_XGB:
        model = xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric="logloss")
    else:
        model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    imp = model.feature_importances_ if hasattr(model, "feature_importances_") else None
    return model, _metrics(y_test, y_pred, y_proba), imp


def train_stacking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
    """StackingClassifier with meta-learner LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    estimators = [
        ("rf", RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)),
        ("svc", SVC(probability=True, random_state=RANDOM_SEED)),
    ]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=RANDOM_SEED),
        cv=5,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # No single feature_importances_ for stacking
    return model, _metrics(y_test, y_pred, y_proba), None

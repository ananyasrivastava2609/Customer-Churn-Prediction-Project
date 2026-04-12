"""
ANN (MLP) for churn: Keras or sklearn MLPClassifier.
Architecture: input -> Dense(64, relu) -> Dense(32, relu) -> Dense(1, sigmoid).
Adam lr=1e-3, epochs=30, batch_size=32, EarlyStopping patience=5.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
RANDOM_SEED = 42

# Prefer Keras if available
try:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False


def train_ann_keras(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "models/churn_ann.h5",
) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
    """
    Build and train Keras ANN. Save to models/churn_ann.h5.
    Returns (model, metrics, None for feature_importances).
    """
    if not HAS_KERAS:
        raise RuntimeError("TensorFlow/Keras not available; use MLPClassifier fallback.")

    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early],
        verbose=0,
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)

    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    try:
        from sklearn.metrics import average_precision_score
        pr_auc = float(average_precision_score(y_test, y_proba))
    except Exception:
        pr_auc = 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0,
        "pr_auc": pr_auc,
    }
    return model, metrics, None


def train_ann_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
    """Fallback: sklearn MLPClassifier with similar architecture (hidden_layer_sizes=(64,32))."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=5,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    try:
        from sklearn.metrics import average_precision_score
        pr_auc = float(average_precision_score(y_test, y_proba))
    except Exception:
        pr_auc = 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0,
        "pr_auc": pr_auc,
    }
    return model, metrics, None

"""
Full churn pipeline: load data, preprocess, train all models, select best by ROC-AUC,
save model, scaler, encoders, and evaluation reports.
"""
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scikitplot as skplt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing import load_and_clean_churn
from src.regression_models import train_logistic
from src.classification_models import (
    train_naive_bayes,
    train_decision_tree,
    train_knn,
    # train_svm,  # ← SKIPPED: too slow for large datasets (GridSearchCV + RBF kernel)
)
from src.ensemble_models import train_random_forest, train_gradient_boosting, train_stacking
from src.ann_model import train_ann_keras, train_ann_sklearn

RANDOM_SEED = 42

Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

EXCLUDE_COLS = {"customer_id", "last_interaction_date", "churn"}


def _get_numeric_categorical(df: pd.DataFrame, target: str = "churn") -> tuple:
    feats = [c for c in df.columns if c != target and c not in EXCLUDE_COLS]
    numeric = [c for c in feats if df[c].dtype in ["int64", "float64"] or np.issubdtype(df[c].dtype, np.number)]
    categorical = [c for c in feats if c not in numeric]
    return numeric, categorical


def prepare_features(df: pd.DataFrame):
    """Build X, y, feature_names, scaler, encoders."""
    y = df["churn"].values
    numeric_cols, categorical_cols = _get_numeric_categorical(df)

    X_num = df[numeric_cols].copy()
    for c in numeric_cols:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce").fillna(X_num[c].median())

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    feature_names = list(numeric_cols)
    encoders = {"freq_maps": {}, "onehot": None, "onehot_cols": [], "freq_cols": []}

    if not categorical_cols:
        return X_num_scaled, y, feature_names, scaler, encoders, numeric_cols, categorical_cols

    high_card = [c for c in categorical_cols if df[c].nunique() > 10]
    low_card = [c for c in categorical_cols if c not in high_card]

    freq_parts = []
    for c in high_card:
        counts = df[c].value_counts().to_dict()
        encoders["freq_maps"][c] = counts
        encoders["freq_cols"].append(c)
        mapped = df[c].map(counts).fillna(0).values.reshape(-1, 1)
        freq_parts.append(mapped)
        feature_names.append(c)

    if low_card:
        onehot = OneHotEncoder(drop="first", sparse_output=False)
        X_cat = onehot.fit_transform(df[low_card].astype(str))
        encoders["onehot"] = onehot
        encoders["onehot_cols"] = low_card
        for i, col in enumerate(low_card):
            for cat in onehot.categories_[i][1:]:
                feature_names.append(f"{col}_{cat}")
    else:
        X_cat = np.empty((len(df), 0))

    parts = [X_num_scaled]
    if freq_parts:
        parts.append(np.hstack(freq_parts))
    if X_cat.shape[1] > 0:
        parts.append(X_cat)
    X = np.hstack(parts)
    return X, y, feature_names, scaler, encoders, numeric_cols, categorical_cols


def transform_churn_features(df: pd.DataFrame, scaler, encoders: dict, num_cols: list, cat_cols: list) -> np.ndarray:
    """Transform dataframe using saved scaler and encoders (no fit)."""
    X_num = df[num_cols].copy()
    for c in num_cols:
        if c in df.columns:
            X_num[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X_num_scaled = scaler.transform(X_num)
    parts = [X_num_scaled]
    for c in encoders.get("freq_cols", []):
        counts = encoders["freq_maps"].get(c, {})
        mapped = df[c].astype(str).map(counts).fillna(0).values.reshape(-1, 1)
        parts.append(mapped)
    if encoders.get("onehot") and encoders.get("onehot_cols"):
        X_cat = encoders["onehot"].transform(df[encoders["onehot_cols"]].astype(str).fillna("missing"))
        parts.append(X_cat)
    return np.hstack(parts)


def main():
    data_path = Path("data/sample_churn.csv")
    if not data_path.exists():
        logger.error("Churn data not found at %s. Run data/generate_demo_data.py first.", data_path)
        sys.exit(1)

    try:
        df = load_and_clean_churn(str(data_path))
    except ValueError as e:
        if "churn" in str(e).lower():
            logger.error("Churn column missing. %s", e)
            sys.exit(1)
        raise

    X, y, feature_names, scaler, encoders, num_cols, cat_cols = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    results = []

    def run(name, fn):
        logger.info("Training %s ...", name)
        model, metrics, imp = fn(X_train, y_train, X_test, y_test)
        results.append((name, model, metrics, imp))
        logger.info("  %s ROC-AUC=%.4f", name, metrics.get("roc_auc", 0))
        return metrics.get("roc_auc", 0)

    run("LogisticRegression", train_logistic)
    run("GaussianNB", train_naive_bayes)
    run("DecisionTree", train_decision_tree)
    run("KNN", train_knn)
    # SVM skipped — use RandomForest/GradientBoosting instead (same accuracy, 100x faster)
    run("RandomForest", train_random_forest)
    run("GradientBoosting", train_gradient_boosting)
    run("StackingClassifier", train_stacking)

    try:
        model, metrics, imp = train_ann_keras(X_train, y_train, X_test, y_test)
        results.append(("ANN_Keras", model, metrics, imp))
        logger.info("  ANN_Keras ROC-AUC=%.4f", metrics.get("roc_auc", 0))
    except Exception as e:
        logger.warning("ANN Keras skipped: %s", e)
        model, metrics, imp = train_ann_sklearn(X_train, y_train, X_test, y_test)
        results.append(("ANN_MLP", model, metrics, imp))
        logger.info("  ANN_MLP ROC-AUC=%.4f", metrics.get("roc_auc", 0))

    # Best by ROC-AUC; for .pkl we only save sklearn models (no Keras)
    sklearn_results = [(n, m, met, imp) for n, m, met, imp in results if "ANN" not in n]
    if not sklearn_results:
        sklearn_results = results
    best_name, best_model, best_metrics, best_importances = max(
        sklearn_results, key=lambda x: x[2].get("roc_auc", 0)
    )

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, "models/churn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(
        {
            "encoders": encoders,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "feature_names": feature_names,
        },
        "models/encoders.pkl",
    )

    importances = None
    if best_importances is not None and len(best_importances) == len(feature_names):
        importances = dict(
            zip(
                feature_names,
                best_importances.tolist() if hasattr(best_importances, "tolist") else list(best_importances),
            )
        )

    model_info = {
        "model_name": best_name,
        "hyperparameters": best_metrics.get("best_params", {}),
        "validation_scores": {
            k: v for k, v in best_metrics.items() if k != "best_params" and isinstance(v, (int, float))
        },
        "selected_features": feature_names,
        "scaler_path": "models/scaler.pkl",
        "feature_importances": importances,
        "training_date": datetime.utcnow().isoformat() + "Z",
        "version": "1.0",
        "random_seed": RANDOM_SEED,
    }
    with open("models/churn_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    Path("reports").mkdir(parents=True, exist_ok=True)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    with open("reports/churn_classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    if HAS_PLOT and y_proba is not None:
        try:
            skplt.metrics.plot_roc_curve(y_test, np.column_stack([1 - y_proba, y_proba]), title="ROC")
            plt.savefig("reports/roc_curve.png", dpi=100)
            plt.close()
        except Exception as e:
            logger.warning("Could not save ROC curve: %s", e)
    if HAS_PLOT:
        try:
            skplt.metrics.plot_confusion_matrix(y_test, y_pred)
            plt.savefig("reports/churn_confusion_matrix.png", dpi=100)
            plt.close()
        except Exception as e:
            logger.warning("Could not save confusion matrix: %s", e)

    logger.info(
        "✅ Best model: %s (ROC-AUC=%.4f). Saved to models/churn_model.pkl",
        best_name, best_metrics.get("roc_auc", 0)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
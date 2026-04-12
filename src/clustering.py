"""
KMeans clustering with elbow plot export for churn/tabular data.
"""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


def fit_kmeans_elbow(
    X: np.ndarray,
    k_range: range = range(2, 11),
    report_dir: str = "reports/eda",
) -> Tuple[KMeans, list]:
    """
    Fit KMeans for each k in k_range, compute inertia, optionally plot elbow.
    Returns (best_kmeans_by_elbow, list of (k, inertia)).
    """
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km.fit(X)
        inertias.append((k, km.inertia_))
    # Use k=3 as default "elbow" for saving; user can inspect list
    best_k = 3
    best_km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    best_km.fit(X)

    Path(report_dir).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = [x[0] for x in inertias]
        inerts = [x[1] for x in inertias]
        plt.figure(figsize=(6, 4))
        plt.plot(ks, inerts, "bo-")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.title("KMeans Elbow")
        plt.savefig(Path(report_dir) / "kmeans_elbow.png", dpi=100)
        plt.close()
    except Exception as e:
        logger.warning("Could not save elbow plot: %s", e)

    return best_km, inertias

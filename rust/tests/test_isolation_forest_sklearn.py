import math

import pytest

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest as SkIsolationForest
except Exception:  # pragma: no cover
    np = None
    SkIsolationForest = None

import aiwaf_rust


pytestmark = pytest.mark.skipif(
    np is None or SkIsolationForest is None,
    reason="scikit-learn and numpy are required for sklearn parity tests",
)


def _rankdata(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _spearman(a, b):
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return 0.0
    return float((ra * rb).sum() / denom)


def _make_data(seed, n_samples=200, n_features=3):
    rng = np.random.default_rng(seed)
    inliers = rng.normal(0.0, 1.0, size=(n_samples - 10, n_features))
    outliers = rng.normal(6.0, 0.5, size=(10, n_features))
    return np.vstack([inliers, outliers])


def test_isolation_forest_sklearn_parity_100_variations():
    correlations = []
    maes = []

    for seed in range(100):
        X = _make_data(seed)
        X_list = X.tolist()

        sk = SkIsolationForest(
            n_estimators=100,
            max_samples="auto",
            contamination=0.05,
            max_features=1.0,
            bootstrap=False,
            random_state=seed,
        )
        sk.fit(X)
        sk_scores = sk.decision_function(X)

        rust = aiwaf_rust.IsolationForest(
            n_estimators=100,
            max_samples="auto",
            contamination=0.05,
            max_features=1.0,
            bootstrap=False,
            random_state=seed,
            warm_start=False,
        )
        rust.fit(X_list)
        rust_scores = rust.decision_function(X_list)

        # Compare rank correlation and MAE
        corr = _spearman(sk_scores, rust_scores)
        mae = float(np.mean(np.abs(sk_scores - np.array(rust_scores))))
        correlations.append(corr)
        maes.append(mae)

    # Expect strong rank agreement and reasonable absolute error
    assert float(np.mean(correlations)) >= 0.85
    assert float(np.mean(maes)) <= 0.25

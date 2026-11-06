"""
Evaluation metrics for protein-ligand binding affinity regression.

All metrics are computed on the whole set of predictions at once (e.g. the full
validation or test set), not averaged over batches. Definitions follow the
CASF-2016 benchmark and the papers we compare against (Pafnucy, IGN, PLAGCA, ...).

    RMSE      root mean squared error                                   (lower is better)
    MAE       mean absolute error                                       (lower is better)
    Pearson   Pearson correlation coefficient R                         (higher is better)
    Spearman  Spearman rank correlation                                 (higher is better)
    SD        std of the residuals after a linear fit y ~ a*y_hat + b   (lower is better)
    CI        concordance index: fraction of correctly ranked pairs     (higher is better)

Usage:
    from gmnbind.metrics import all_metrics
    scores = all_metrics(y_true, y_pred)   # dict with the six metrics
"""

import numpy as np
from scipy import stats

__all__ = ["rmse", "mae", "pearson", "spearman", "sd", "concordance_index", "all_metrics"]


def _to_arrays(y, y_hat):
    """Convert inputs (lists, numpy arrays, torch tensors) to flat float64 numpy arrays."""
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    if hasattr(y_hat, "detach"):
        y_hat = y_hat.detach().cpu().numpy()
    y = np.asarray(y, dtype=np.float64).ravel()
    y_hat = np.asarray(y_hat, dtype=np.float64).ravel()
    if y.shape != y_hat.shape:
        raise ValueError(f"Shape mismatch: y {y.shape} vs y_hat {y_hat.shape}")
    return y, y_hat


def rmse(y, y_hat):
    y, y_hat = _to_arrays(y, y_hat)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def mae(y, y_hat):
    y, y_hat = _to_arrays(y, y_hat)
    return float(np.mean(np.abs(y - y_hat)))


def pearson(y, y_hat):
    y, y_hat = _to_arrays(y, y_hat)
    if len(y) < 2 or np.std(y) == 0 or np.std(y_hat) == 0:
        return float("nan")  # undefined for constant inputs
    return float(stats.pearsonr(y, y_hat)[0])


def spearman(y, y_hat):
    y, y_hat = _to_arrays(y, y_hat)
    if len(y) < 2 or np.std(y) == 0 or np.std(y_hat) == 0:
        return float("nan")
    return float(stats.spearmanr(y, y_hat)[0])


def sd(y, y_hat):
    """
    Standard deviation of the regression residuals (CASF-2016 definition).

    A linear fit y ~ a * y_hat + b is computed first; SD measures how far the true
    values are from that line:  sqrt( sum((y - (a*y_hat + b))^2) / (N - 1) ).
    This is NOT the standard deviation of the errors (y - y_hat).
    """
    y, y_hat = _to_arrays(y, y_hat)
    if len(y) < 3 or np.std(y_hat) == 0:
        return float("nan")
    a, b = np.polyfit(y_hat, y, deg=1)
    residuals = y - (a * y_hat + b)
    return float(np.sqrt(np.sum(residuals ** 2) / (len(y) - 1)))


def concordance_index(y, y_hat):
    """
    Concordance index (as in DeepDTA and following works).

    Over all pairs (i, j) with y_i > y_j, the fraction where y_hat_i > y_hat_j.
    Ties in the predictions count 0.5. Pairs with equal true values are ignored.
    0.5 = random ranking, 1.0 = perfect ranking.

    Vectorized O(N^2) in memory: fine for a few thousand samples (CASF core = 285).
    """
    y, y_hat = _to_arrays(y, y_hat)
    true_diff = y[:, None] - y[None, :]          # [N, N]
    pred_diff = y_hat[:, None] - y_hat[None, :]
    pairs = true_diff > 0                        # each unordered pair counted once
    n_pairs = pairs.sum()
    if n_pairs == 0:
        return float("nan")
    concordant = (pred_diff[pairs] > 0).sum() + 0.5 * (pred_diff[pairs] == 0).sum()
    return float(concordant / n_pairs)


def all_metrics(y, y_hat):
    """All six metrics as a dict: rmse, mae, pearson, spearman, sd, ci."""
    y, y_hat = _to_arrays(y, y_hat)
    return {
        "rmse": rmse(y, y_hat),
        "mae": mae(y, y_hat),
        "pearson": pearson(y, y_hat),
        "spearman": spearman(y, y_hat),
        "sd": sd(y, y_hat),
        "ci": concordance_index(y, y_hat),
    }


if __name__ == "__main__":
    # Smoke test with synthetic data: python -m gmnbind.metrics
    rng = np.random.default_rng(0)
    y = rng.uniform(2, 12, size=285)
    print("perfect :", all_metrics(y, y))
    print("noisy   :", all_metrics(y, y + rng.normal(0, 1.0, size=y.shape)))
    print("random  :", all_metrics(y, rng.permutation(y)))

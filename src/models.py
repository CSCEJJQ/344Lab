"""Prediction models for the MovieLens recommendation project."""
import numpy as np


def bias_baseline(R):
    """Overall mean plus a user effect and a movie effect.

    R is a user-by-movie matrix with NaN for missing ratings.
    Returns (mu, b_user, b_item). Users/movies with no ratings get effect 0.
    """
    known = ~np.isnan(R)
    mu = np.nanmean(R)

    # Movie effect: average deviation from mu for each movie.
    n_item = known.sum(axis=0)
    b_item = np.zeros(R.shape[1])
    has_item = n_item > 0
    b_item[has_item] = (np.nansum(R - mu, axis=0)[has_item] / n_item[has_item])

    # User effect: average leftover deviation after removing mu and movie effect.
    resid = R - mu - b_item[None, :]
    n_user = known.sum(axis=1)
    b_user = np.zeros(R.shape[0])
    has_user = n_user > 0
    b_user[has_user] = np.nansum(resid, axis=1)[has_user] / n_user[has_user]
    return mu, b_user, b_item


def baseline_matrix(R):
    """Full matrix: overall mean plus user and movie effects."""
    mu, b_user, b_item = bias_baseline(R)
    return mu + b_user[:, None] + b_item[None, :]


def rank_k(U, s, Vt, k):
    """Build a rank-k matrix from an already computed SVD."""
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def iterative_svd_predict(R, k, n_iter=10):
    """Predict ratings by repeatedly imputing missing deviations."""
    known = ~np.isnan(R)
    B = baseline_matrix(R)
    X = np.where(known, R - B, 0.0)
    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = rank_k(U, s, Vt, k)
        X = np.where(known, R - B, L)
    return L + B

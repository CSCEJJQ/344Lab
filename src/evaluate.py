"""Evaluation helpers for the MovieLens recommendation project."""
import numpy as np


def split_ratings(ratings, test_frac=0.2, seed=0):
    """Randomly split a ratings DataFrame into (train, test).

    A fraction `test_frac` of the rows goes to the second DataFrame.
    The same seed always gives the same split.
    """
    rng = np.random.default_rng(seed)
    n = len(ratings)
    perm = rng.permutation(n)
    n_test = int(round(test_frac * n))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    train = ratings.iloc[np.sort(train_idx)].reset_index(drop=True)
    test = ratings.iloc[np.sort(test_idx)].reset_index(drop=True)
    return train, test


def to_matrix(df, n_users=943, n_items=1682):
    """Build a user-by-movie matrix with NaN where there is no rating."""
    R = np.full((n_users, n_items), np.nan)
    R[df["user"].to_numpy() - 1, df["item"].to_numpy() - 1] = df["rating"].to_numpy()
    return R


def rmse(pred, truth):
    """Root mean squared error between predictions and true ratings."""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))

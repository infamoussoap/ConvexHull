import numpy as np
import sys

from .KKTConditions import validate_kkt_conditions
from .utils import verbose_callback


def clip(x, min_val, max_val):
    if x >= max_val:
        return max_val
    elif x <= min_val:
        return min_val
    return x


def cauchy_simplex_optimizer(X, y, kkt_tol=1e-4, max_iter=-1, verbose=False, w=None, e=1e-10):
    if w is None:
        w = np.ones(len(X)) / len(X)
    else:
        assert abs(np.sum(w) - 1) < e, "w must sum to 1"
        assert len(w) == len(X), "Length of w must be the same length as the hull"

    count = 0
    while count < max_iter or max_iter < 0:
        grad = (w @ X - y) @ X.T

        if validate_kkt_conditions(w, grad, tol=kkt_tol, e=e):
            break

        dw_dt = w * (grad - w @ grad)
        cauchy_learning_rate = dw_dt @ grad / np.sum((dw_dt @ X) ** 2)

        non_active_set = w > e
        max_learning_rate = 1 / (np.max(grad[non_active_set]) - w @ grad)

        learning_rate = clip(cauchy_learning_rate, 0, max_learning_rate)

        w = w - learning_rate * dw_dt
        w[~non_active_set] = 0.0
        w = w / np.sum(w)

        count += 1

        if verbose:
            verbose_callback(count, max_iter, w, X, y)

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    distance = np.sum((w @ X - y) ** 2) / 2
    return distance, count + 1, w

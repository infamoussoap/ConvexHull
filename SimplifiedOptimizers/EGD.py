import numpy as np
import sys

from .BisectionMethod import BisectionMethod
from .KKTConditions import validate_kkt_conditions
from .utils import verbose_callback


def egd_optimizer(X, y, kkt_tol=1e-4, max_iter=-1, verbose=False, w=None, e=1e-10):
    if w is None:
        w = np.ones(len(X)) / len(X)
    else:
        assert abs(np.sum(w) - 1) < 1e-10, "w must sum to 1"
        assert len(w) == len(X), "Length of w must be the same length as the hull"

    search_method = BisectionMethod(X)

    count = 0
    while count < max_iter or max_iter < 0:
        grad = (w @ X - y) @ X.T

        if validate_kkt_conditions(w, grad, tol=kkt_tol, e=e):
            break

        t_max = min(2 * (count + 1), 1000)
        learning_rate = search_method.search(w, y, t_max, grad, search_type='classical')

        z = -learning_rate * grad
        x = w * np.exp(z - np.max(z))  # For numerical stability
        w = x / np.sum(x)

        count += 1

        if verbose:
            verbose_callback(count, max_iter, w, X, y)

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    distance = np.sum((w @ X - y) ** 2) / 2
    return distance, count + 1, w

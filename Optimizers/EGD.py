import numpy as np
import sys

from .BisectionMethod import BisectionMethod
from .StoppingCondition import validate_stopping_conditions
from .utils import verbose_callback


def egd_optimizer(X, y, max_iter=-1, verbose=False, w=None, tol=1e-6, e=1e-10,
                  stopping_type="TOL"):
    if w is None:
        w = np.ones(len(X)) / len(X)
    else:
        assert abs(np.sum(w) - 1) < 1e-10, "w must sum to 1"
        assert len(w) == len(X), "Length of w must be the same length as the hull"

    search_method = BisectionMethod(X)

    count = 0
    while count < max_iter or max_iter < 0:
        grad = (w @ X - y) @ X.T

        # The limit is relative to the gradient for numerical stability
        t_max = min(2 * (count + 1), abs(100 / np.max(grad)))
        learning_rate = search_method.search(w, y, t_max, grad, search_type='classical')

        z = -learning_rate * grad
        x = w * np.exp(z - np.max(z))  # For numerical stability
        w = x / np.sum(x)

        count += 1

        if verbose:
            verbose_callback(count, max_iter, w, X, y)

        if validate_stopping_conditions(w, X, y, tol=tol, e=e, stopping_type=stopping_type):
            break

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    w[w < e] = 0
    w = w / np.sum(w)

    distance = np.sum((w @ X - y) ** 2) / 2
    return distance, count + 1, w

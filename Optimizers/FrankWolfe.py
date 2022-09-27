import numpy as np
import sys

from .StoppingCondition import validate_stopping_conditions
from .utils import verbose_callback, clip


def frank_wolfe_optimizer(X, y, max_iter=-1, verbose=False, w=None, tol=1e-6, e=1e-10,
                          stopping_type="TOL"):

    if w is None:
        w = np.zeros(len(X))
        w[0] = 1
    else:
        assert abs(np.sum(w) - 1) < e, "w must sum to 1"
        assert len(w) == len(X), "Length of w must be the same length as the hull"

    count = 0
    oracle = np.zeros(len(X))
    while count < max_iter or max_iter < 0:
        grad = (w @ X - y) @ X.T

        s_index = np.argmin(grad)
        oracle[s_index] = 1

        wX = (oracle - w) @ X
        cauchy_learning_rate = - wX @ (w @ X - y) / (wX @ wX)

        learning_rate = clip(cauchy_learning_rate, 0, 1)

        w = (1 - learning_rate) * w + learning_rate * oracle

        count += 1

        oracle[s_index] = 0

        if verbose:
            verbose_callback(count, max_iter, w, X, y)

        if validate_stopping_conditions(w, X, y, tol=tol, e=e, stopping_type=stopping_type):
            break

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    distance = np.sum((w @ X - y) ** 2) / 2
    return distance, count + 1, w

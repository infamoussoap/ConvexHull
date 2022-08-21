import numpy as np
import sys

from .StoppingCondition import validate_stopping_conditions
from .utils import verbose_callback


def frank_wolfe_optimizer(X, y, max_iter=-1, verbose=False, w=None, tol=1e-6, e=1e-10,
                          stopping_type="TOL"):
    """ This implements the pairwise frank-wolfe algorithm """

    if w is None:
        w = np.zeros(len(X))
        w[0] = 1
    else:
        assert abs(np.sum(w) - 1) < e, "w must sum to 1"
        assert len(w) == len(X), "Length of w must be the same length as the hull"

    count = 0
    while count < max_iter or max_iter < 0:
        grad = (w @ X - y) @ X.T

        s_index = np.argmin(grad)

        non_active_set = w > e
        v_masked_index = np.argmax(grad[non_active_set])
        v_index = np.argwhere(non_active_set).flatten()[v_masked_index]

        alpha = w[v_index]

        # We are taking mass away from the v and putting it into s
        d_X = alpha * (X[s_index] - X[v_index])
        learning_rate = - (w @ X - y) @ d_X / (d_X @ d_X)

        if learning_rate < e:
            break

        learning_rate = min(learning_rate, 1)

        w[s_index] += learning_rate * alpha
        w[v_index] -= learning_rate * alpha

        count += 1

        if verbose:
            verbose_callback(count, max_iter, w, X, y)

        if validate_stopping_conditions(w, X, y, tol=tol, e=e, stopping_type=stopping_type):
            break

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    distance = np.sum((w @ X - y) ** 2) / 2
    return distance, count + 1, w

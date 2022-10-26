import numpy as np
import sys

from .StoppingCondition import validate_stopping_conditions
from .utils import verbose_callback
from .utils import clip


def cauchy_simplex_v4_optimizer(X, y, max_iter=-1, verbose=False, w=None, tol=1e-6, e=1e-10,
                                stopping_type="TOL"):
    """ Version 3 alternates between cauchy-simplex and pairwise frank-wolfe

        Returns
        -------
        float
            The distance to the hull of the result
        int
            The number of steps the optimizer has taken
        np.array
            The result
    """

    if w is None:
        w = np.ones(len(X)) / len(X)
    else:
        assert abs(np.sum(w) - 1) < e, "w must sum to 1"
        assert len(w) == len(X), "Length of w must be the same length as the hull"

    count = 0
    while count < max_iter or max_iter < 0:
        grad = (w @ X - y) @ X.T

        if count % 2 == 0:
            learning_rate, w = _cauchy_simplex(X, y, w, grad, e=e)
        else:
            learning_rate, w = _pairwise_frank_wolfe(X, y, w, grad, e=e)

        count += 1

        if verbose:
            verbose_callback(count, max_iter, w, X, y)

        if validate_stopping_conditions(w, X, y, tol=tol, e=e, stopping_type=stopping_type):
            break

        if learning_rate < e:
            break

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    distance = np.sum((w @ X - y) ** 2) / 2
    return distance, count, w


def _cauchy_simplex(X, y, w, grad, e=1e-10):
    dw_dt = w * (grad - w @ grad)
    cauchy_learning_rate = dw_dt @ grad / np.sum((dw_dt @ X) ** 2)

    non_active_set = w > e
    max_learning_rate = 1 / (np.max(grad) - w @ grad)

    learning_rate = clip(cauchy_learning_rate, 0, max_learning_rate)

    w = w - learning_rate * dw_dt
    w[~non_active_set] = 0.0
    w = w / np.sum(w)

    return learning_rate, w


def _pairwise_frank_wolfe(X, y, w, grad, e=1e-10):
    s_index = np.argmin(grad)

    non_active_set = w > e
    v_masked_index = np.argmax(grad[non_active_set])
    v_index = np.argwhere(non_active_set).flatten()[v_masked_index]

    alpha = w[v_index]

    # We are taking mass away from the v and putting it into s
    d_X = alpha * (X[s_index] - X[v_index])
    learning_rate = - (w @ X - y) @ d_X / (d_X @ d_X)

    learning_rate = clip(learning_rate, 0, 1)

    w[s_index] += learning_rate * alpha
    w[v_index] -= learning_rate * alpha

    return learning_rate, w

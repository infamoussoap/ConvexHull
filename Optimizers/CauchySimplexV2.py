import numpy as np
import sys

from .StoppingCondition import validate_stopping_conditions
from .utils import verbose_callback
from .utils import clip


def cauchy_simplex_v2_optimizer(X, y, max_iter=-1, verbose=False, w=None, tol=1e-6, e=1e-10,
                                stopping_type="TOL"):
    """ Version 2 will perform a frank-wolfe optimisation if the stability condition is
        not satisfied

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

        unstable_active_set_indices = _check_indices(w, grad)
        if len(unstable_active_set_indices) > 0:
            learning_rate, w = _frank_wolfe_index(X, y, w, grad, unstable_active_set_indices)
        else:
            learning_rate, w = _cauchy_simplex(X, y, w, grad, e=e)

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


def _check_indices(w, grad):
    stability = grad - w @ grad

    active_set = w == 0
    unstable_set = stability < 0

    return np.argwhere(active_set * unstable_set).flatten()


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


def _frank_wolfe_index(X, y, w, grad, indices):
    for i in indices:
        oracle = np.zeros(len(X))
        oracle[i] = 1

        wX = (oracle - w) @ X
        cauchy_learning_rate = - wX @ (w @ X - y) / (wX @ wX)

        learning_rate = clip(cauchy_learning_rate, 0, 1)

        w = (1 - learning_rate) * w + learning_rate * oracle

    return learning_rate, w

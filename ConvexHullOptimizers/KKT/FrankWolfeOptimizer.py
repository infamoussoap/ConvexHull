import numpy as np
import sys

from .KKTConditions import validate_kkt_conditions

from ..utils import verbose_callback


def clip(x, min_val, max_val):
    """ np.clip tends to be quite slower than a manual implementation """
    if x < min_val:
        return min_val
    elif x > max_val:
        return max_val
    return x


def frank_wolfe_optimizer(points, y, kkt_tol=1e-3, max_iter=-1, verbose=False, w=None, tol=1e-10):
    if w is None:
        w = np.zeros(len(points))
        w[0] = 1
    else:
        assert abs(np.sum(w) - 1) < 1e-10, "w must sum to 1"
        assert len(w) == len(points), "Length of w must be the same length as the hull"

    status = 'Failed'

    oracle = np.zeros(len(points))
    count = 0
    while count < max_iter or max_iter < 0:
        b = (w @ points - y)
        grad = b @ points.T
        if validate_kkt_conditions(w, grad, kkt_tol):
            status = 'KKT'
            break

        i = np.argmin(grad)
        oracle[i] = 1

        a = (oracle - w) @ points
        gamma = - (a @ b) / (a @ a)

        if gamma < tol:
            break

        gamma = clip(gamma, 0, 1)

        w = (1 - gamma) * w + gamma * oracle
        w = w / np.sum(w)

        oracle[i] = 0

        if verbose:
            verbose_callback(count, max_iter, w, points, y)

        count += 1

    distance = np.sum((w @ points - y) ** 2)

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return status, distance, count + 1, w

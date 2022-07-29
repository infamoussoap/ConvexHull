import numpy as np
import sys

from .utils import verbose_callback
from .Tape import Tape


def squared_optimizer(points, y, tol=1e-8, max_iter=-1, verbose=False, w=None, e=1e-10):
    if w is None:
        w = np.ones(len(points)) / len(points)
    else:
        assert abs(np.sum(w) - 1) < e, "w must sum to 1"
        assert len(w) == len(points), "Length of w must be the same length as the hull"

    status = 'Failed'

    tape = Tape(tol)

    count = 0
    while count < max_iter or max_iter < 0:
        grad = (w @ points - y) @ points.T

        dw_dt = w * (grad - w @ grad)
        cauchy_learning_rate = dw_dt @ grad / np.sum((dw_dt @ points) ** 2)

        non_active_set = w > e
        max_learning_rate = 1 / (np.max(grad[non_active_set]) - w @ grad)
        
        learning_rate = min(cauchy_learning_rate, max_learning_rate)

        w = w - learning_rate * dw_dt
        w[~non_active_set] = 0.0
        w = w / np.sum(w)

        if verbose:
            verbose_callback(count, max_iter, w, points, y)

        if tape.watch(w):
            status = 'TOL'
            break

        count += 1

    distance = np.sum((w @ points - y) ** 2)

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return status, distance, count + 1, w

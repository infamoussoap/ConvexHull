import numpy as np
import sys

from .KKTConditions import validate_kkt_conditions

from .utils import project_onto_standard_simplex, verbose_callback


def pgd_optimizer(points, y, kkt_tol=1e-3, max_iter=-1, verbose=False, w=None):
    if w is None:
        w = np.ones(len(points)) / len(points)
    else:
        assert abs(np.sum(w) - 1) < 1e-10, "w must sum to 1"
        assert len(w) == len(points), "Length of w must be the same length as the hull"

    status = 'Failed'

    count = 0
    while count < max_iter or max_iter < 0:
        grad = (w @ points - y) @ points.T
        if validate_kkt_conditions(w, grad, kkt_tol):
            status = 'KKT'
            break

        learning_rate = grad @ grad / np.sum((grad @ points) ** 2)
        x = w - learning_rate * grad
        w = project_onto_standard_simplex(x)

        if verbose:
            verbose_callback(count, max_iter, w, points, y)

        count += 1

    distance = np.sum((w @ points - y) ** 2)

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return status, distance, count + 1, w

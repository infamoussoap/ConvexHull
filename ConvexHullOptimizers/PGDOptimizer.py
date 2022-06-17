import numpy as np
import sys

from .KKTConditions import validate_kkt_conditions

from .utils import verbose_callback


def pgd_optimizer(points, y, kkt_tol=1e-3, max_iter=-1, verbose=False, w=None, tol=1e-20):
    if w is None:
        w = np.ones(len(points)) / len(points)
    else:
        assert abs(np.sum(w) - 1) < 1e-10, "w must sum to 1"
        assert len(w) == len(points), "Length of w must be the same length as the hull"

    indices = np.arange(len(w))
    original_length = len(w)

    status = 'Failed'

    count = 0
    while count < max_iter or max_iter < 0:
        mask = w > tol
        w = w[mask]
        points = points[mask]
        indices = indices[mask]

        grad = (w @ points - y) @ points.T
        if validate_kkt_conditions(w, grad, kkt_tol):
            status = 'KKT'
            break

        normalized_grad = (grad - np.mean(grad)) / len(grad)
        max_learning_rate = 1 / np.max(normalized_grad / w)

        cauchy_learning_rate = normalized_grad @ grad / np.sum((normalized_grad @ points) ** 2)

        learning_rate = min(cauchy_learning_rate, max_learning_rate)

        w = w - learning_rate * grad
        w = w / np.sum(w)

        if verbose:
            verbose_callback(count, max_iter, w, points, y)

        count += 1

    distance = np.sum((w @ points - y) ** 2)

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    w_copy = w.copy()
    w = np.zeros(original_length)
    w[indices] = w_copy

    return status, distance, count + 1, w

import numpy as np
import sys

from .KKTConditions import validate_kkt_conditions
from .BisectionMethod import BisectionMethod

from .utils import project_onto_standard_simplex, verbose_callback


def squared_optimizer(points, y, kkt_tol=1e-3, max_iter=-1, verbose=False):
    w = np.ones(len(points)) / len(points)
    status = 'Failed'

    count = 0
    while count < max_iter or max_iter < 0:
        grad = (w @ points - y) @ points.T
        if validate_kkt_conditions(w, grad, kkt_tol):
            status = 'KKT'
            break

        dw_dt = w * (grad - w @ grad)
        cauchy_learning_rate = dw_dt @ grad / np.sum((dw_dt @ points) ** 2)

        max_learning_rate = 1 / (np.max(grad) - w @ grad)
        learning_rate = min(cauchy_learning_rate, max_learning_rate)

        w = w - learning_rate * dw_dt
        w = w / np.sum(w)

        if verbose:
            verbose_callback(count, max_iter, w, points, y)

        count += 1

    distance = np.sum((w @ points - y) ** 2)

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return status, distance, count + 1, w


def egd_optimizer(points, y, kkt_tol=1e-3, max_iter=-1, verbose=False):
    search_method = BisectionMethod(points)

    w = np.ones(len(points)) / len(points)
    status = 'Failed'

    count = 0
    while count < max_iter or max_iter < 0:
        grad = (w @ points - y) @ points.T
        if validate_kkt_conditions(w, grad, kkt_tol):
            status = 'KKT'
            break

        t_max = min(2 * (count + 1), 1000)
        learning_rate = search_method.search(w, y, t_max, grad, search_type='classical')

        if learning_rate < 1e-7:  # No more learning to be done
            status = 'Gradient'
            break

        x = w * np.exp(-learning_rate * grad)
        w = x / np.sum(x)

        if verbose:
            verbose_callback(count, max_iter, w, points, y)

        count += 1

    distance = np.sum((w @ points - y) ** 2)

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return status, distance, count + 1, w


def pgd_optimizer(points, y, kkt_tol=1e-3, max_iter=-1, verbose=False):
    w = np.ones(len(points)) / len(points)
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

import numpy as np

from .KKTConditions import validate_kkt_conditions
from .BisectionMethod import BisectionMethod


def squared_optimizer(points, y, kkt_tol=1e-3, max_iter=1000, tol=1e-8):
    w = np.ones(len(points)) / len(points)
    status = 'Failed'

    for count in range(max_iter):
        grad = (w @ points - y) @ points.T
        if validate_kkt_conditions(w, grad, kkt_tol):
            status = 'KKT'
            break

        dw_dt = w * (grad - w @ grad)
        cauchy_learning_rate = dw_dt @ grad / np.sum((dw_dt @ points) ** 2)

        mask = dw_dt > tol
        if np.any(mask):  # Check non-empty
            max_learning_rate = np.min(w[mask] / dw_dt[mask])
            learning_rate = min(cauchy_learning_rate, max_learning_rate)
        else:
            learning_rate = cauchy_learning_rate

        w = w - learning_rate * dw_dt
        w = w / np.sum(w)

    distance = np.sum((w @ points - y) ** 2)

    return status, distance, count + 1, w


def egd_optimizer(points, y, kkt_tol=1e-3, max_iter=1000):
    search_method = BisectionMethod(points)

    w = np.ones(len(points)) / len(points)
    status = 'Failed'

    for count in range(max_iter):
        grad = (w @ points - y) @ points.T
        if validate_kkt_conditions(w, grad, kkt_tol):
            status = 'KKT'
            break

        t_max = 2 * (count + 1)
        learning_rate = search_method.search(w, y, t_max, grad, search_type='classical')

        if learning_rate < 1e-7:  # No more learning to be done
            status = 'Gradient'
            break

        x = w * np.exp(-learning_rate * grad)
        w = x / np.sum(x)

    distance = np.sum((w @ points - y) ** 2)

    return status, distance, count + 1, w

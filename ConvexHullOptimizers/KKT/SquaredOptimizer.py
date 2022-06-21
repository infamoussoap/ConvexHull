import numpy as np
import sys

from .KKTConditions import validate_kkt_conditions

from .utils import verbose_callback


def squared_optimizer(points, y, kkt_tol=1e-3, max_iter=-1, verbose=False, w=None, reset_optimizer=100):
    if w is None:
        w = np.ones(len(points)) / len(points)
    else:
        assert abs(np.sum(w) - 1) < 1e-10, "w must sum to 1"
        assert len(w) == len(points), "Length of w must be the same length as the hull"

    original_length = len(w)

    current_iter = 0

    non_active_set = np.arange(len(w))
    status, distance, current_iter, w = _squared_optimizer(points, y, kkt_tol, current_iter, max_iter,
                                                           verbose, w, reset_optimizer)

    while status == 'reset':
        non_zero_mask = w != 0
        non_active_set = non_active_set[non_zero_mask]

        w = w[non_zero_mask]
        points = points[non_zero_mask]

        status, distance, current_iter, w = _squared_optimizer(points, y, kkt_tol, current_iter, max_iter,
                                                               verbose, w, reset_optimizer)

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    if len(w) < original_length:
        w_values = w.copy()
        w = np.zeros(original_length)
        w[non_active_set] = w_values

    return status, distance, current_iter + 1, w


def _squared_optimizer(points, y, kkt_tol, current_iter, max_iter, verbose, w, reset_optimizer):
    non_active_set = np.ones(len(w))
    active_set_length = 0

    status = 'Failed'

    while (current_iter < max_iter or max_iter < 0) \
        and (active_set_length < reset_optimizer or reset_optimizer < 0):

        grad = (w @ points - y) @ points.T
        if validate_kkt_conditions(w, grad, kkt_tol):
            status = 'KKT'
            break

        dw_dt = w * (grad - w @ grad)
        cauchy_learning_rate = dw_dt @ grad / np.sum((dw_dt @ points) ** 2)

        masked_grad = grad * non_active_set
        i = np.argmax(masked_grad)

        max_learning_rate = 1 / (masked_grad[i] - w @ grad)
        learning_rate = min(cauchy_learning_rate, max_learning_rate)

        w = w - learning_rate * dw_dt

        if learning_rate == max_learning_rate:
            non_active_set[i] = 0
            w[i] = 0
            active_set_length += 1

        w = w / np.sum(w)

        if verbose:
            verbose_callback(current_iter, max_iter, w, points, y)

        current_iter += 1

    distance = np.sum((w @ points - y) ** 2)

    if active_set_length == reset_optimizer:
        status = 'reset'

    return status, distance, current_iter, w

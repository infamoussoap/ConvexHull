import numpy as np
import sys

from .KKTConditions import validate_kkt_conditions

from ..utils import verbose_callback


def pgd_optimizer(points, y, kkt_tol=1e-3, max_iter=-1, subspace_minimization_iter=10,
                  verbose=False, w=None, e=1e-10):
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

        w = subspace_minimize(grad, w, points, max_iter=subspace_minimization_iter, tol=e)

        if verbose:
            verbose_callback(count, max_iter, w, points, y)
        count += 1

    distance = np.sum((w @ points - y) ** 2)

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return status, distance, count + 1, w


def subspace_minimize(grad, w, points, max_iter=20, tol=1e-10):
    working_indices, w_vals = _subspace_minimize(grad, w, points, np.arange(len(w)), 0, max_iter, tol)

    w_out = np.zeros(len(w))
    w_out[working_indices] = w_vals

    return w_out


def _subspace_minimize(grad, w, points, working_indices, count, max_iter, tol):
    assert len(w) > 0, f"{count}"

    for count in range(max_iter):
        non_binding_indices = get_non_binding_indices(w, grad, np.arange(len(w)), tol)

        non_binding_grad = grad[non_binding_indices]
        non_binding_w = w[non_binding_indices]
        non_binding_points = points[non_binding_indices]
        non_binding_working_indices = working_indices[non_binding_indices]

        normalized_grad = (non_binding_grad - np.mean(non_binding_grad)) / len(non_binding_grad)

        # Compute the cauchy learning rate
        max_learning_rate = compute_max_learning_rate(non_binding_grad, non_binding_w, tol)
        cauchy_learning_rate = compute_cauchy_learning_rate(normalized_grad, non_binding_grad,
                                                            non_binding_points)

        learning_rate = min(cauchy_learning_rate, max_learning_rate)
        if learning_rate < tol:
            break

        w = w[non_binding_indices] - learning_rate * normalized_grad
        w = w / np.sum(w)  # Normalization for Stability

        grad = non_binding_grad
        points = non_binding_points
        working_indices = non_binding_working_indices

        if learning_rate == cauchy_learning_rate:
            break

    return working_indices, w


def get_non_binding_indices(w, grad, working_indices, tol):
    normalized_grad = (grad - np.mean(grad)) / len(grad)
    binding_indices = (w <= tol) * (normalized_grad >= tol)

    while sum(binding_indices) > 0:
        non_binding_indices = ~binding_indices

        w = w[non_binding_indices]
        grad = grad[non_binding_indices]
        working_indices = working_indices[non_binding_indices]

        normalized_grad = (grad - np.mean(grad)) / len(grad)
        binding_indices = (w <= tol) * (normalized_grad >= tol)

    return working_indices


def compute_max_learning_rate(grad, w, tol):
    mask = w > tol
    max_learning_rate = 1 / np.max(grad[mask] / w[mask])
    return max_learning_rate


def compute_cauchy_learning_rate(normalized_grad, grad, points):
    numerator = normalized_grad @ grad
    denomenator = np.sum((normalized_grad @ points) ** 2)
    cauchy_learning_rate = numerator / denomenator

    return cauchy_learning_rate

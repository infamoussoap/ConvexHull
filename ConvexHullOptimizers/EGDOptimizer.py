import numpy as np
import sys

from .BisectionMethod import BisectionMethod

from .utils import verbose_callback
from .Tape import Tape


def egd_optimizer(points, y, tol=1e-8, max_iter=-1, verbose=False, w=None):
    if w is None:
        w = np.ones(len(points)) / len(points)
    else:
        assert abs(np.sum(w) - 1) < 1e-10, "w must sum to 1"
        assert len(w) == len(points), "Length of w must be the same length as the hull"

    search_method = BisectionMethod(points)
    status = 'Failed'

    tape = Tape(tol)

    count = 0
    while count < max_iter or max_iter < 0:
        grad = (w @ points - y) @ points.T

        t_max = min(2 * (count + 1), 1000)
        learning_rate = search_method.search(w, y, t_max, grad, search_type='classical')

        if learning_rate < 1e-7:  # No more learning to be done
            status = 'Gradient'
            break

        x = w * np.exp(-learning_rate * grad)
        w = x / np.sum(x)

        if verbose:
            verbose_callback(count, max_iter, w, points, y)

        if tape.watch(w):
            status = "TOL"
            break

        count += 1

    distance = np.sum((w @ points - y) ** 2)

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return status, distance, count + 1, w

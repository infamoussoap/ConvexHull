import numpy as np
import sys

from .Log import Log
from .BisectionMethod import BisectionMethod
from .KKTConditions import validate_kkt_conditions


class EGDOptimizer:
    def __init__(self, points):
        self.points = points
        self.log = None
        self.H = None

    def optimize(self, y, w=None, learning_rate=None,
                 search_type=None, tol=1e-3, max_iter=5000,
                 log=None, verbose=True, log_weights=False):
        if log is None:
            log = Log()

        if w is None:
            w = np.ones(len(self.points)) / len(self.points)

        if search_type is None:
            search_type = 'classical'

        return self._optimize(y, w, learning_rate, search_type, tol,
                              max_iter, log, verbose, log_weights)

    def _optimize(self, y, w, learning_rate, search_type, tol, max_iter,
                  log, verbose, log_weights):
        search_method = BisectionMethod(self.points)

        status = None
        current_distance = np.sum((w @ self.points - y) ** 2)
        for count in range(max_iter):
            grad = (w @ self.points - y) @ self.points.T
            if validate_kkt_conditions(w, grad, tol):
                status = 'KKT'
                break

            # t0, t1 = 0, 2 / np.max(abs(grad))
            t_max = 2 * (count + 1)
            learning_rate = search_method.search(w, y, t_max, grad, search_type=search_type)

            if learning_rate < 1e-7:  # No more learning to be done
                status = 'gradient'
                break

            x = w * np.exp(-learning_rate * grad)
            w = x / np.sum(x)

            current_distance = np.sum((w @ self.points - y) ** 2)

            # Callbacks
            if log_weights:
                log.log(distance=current_distance, w=w)
            else:
                log.log(distance=current_distance)

            self.verbose_callback(verbose, current_distance, count, max_iter)

        if status is None:
            status = "failed"

        if verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()

        return status, log, w

    def verbose_callback(self, verbose, current_distance, count, max_iter):
        if not verbose:
            return

        sys.stdout.write(f"\r{count + 1} of {max_iter}: Distance {current_distance:.5E}")
        sys.stdout.flush()

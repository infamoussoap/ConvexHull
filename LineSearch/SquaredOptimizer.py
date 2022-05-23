import numpy as np
import sys

from .Log import Log
from .KKTConditions import validate_kkt_conditions


class SquaredOptimizer:
    def __init__(self, points):
        self.points = points
        self.log = None

    def optimize(self, y, w=None, kkt_tol=1e-3, max_iter=1000,
                 log=None, verbose=True, log_weights=False, tol=1e-8):
        if log is None:
            log = Log()

        if w is None:
            w = np.ones(len(self.points)) / len(self.points)

        return self._optimize(y, w, kkt_tol, max_iter, log, verbose, log_weights, tol)

    def _optimize(self, y, w, kkt_tol, max_iter, log, verbose, log_weights, tol):
        status = None
        current_distance = np.sum((w @ self.points - y) ** 2)

        for count in range(max_iter):
            grad = (w @ self.points - y) @ self.points.T
            if validate_kkt_conditions(w, grad, kkt_tol):
                status = 'KKT'
                break

            dw_dt = w * (grad - w @ grad)
            cauchy_learning_rate = dw_dt @ grad / np.sum((dw_dt @ self.points) ** 2)

            mask = dw_dt > tol
            if np.any(mask):  # Check non-empty
                max_learning_rate = np.min(w[mask] / dw_dt[mask])
                learning_rate = min(cauchy_learning_rate, max_learning_rate)
            else:
                learning_rate = cauchy_learning_rate

            w = w - learning_rate * dw_dt
            w = w / np.sum(w)

            current_distance = np.sum((w @ self.points - y) ** 2)

            # Callbacks
            if log_weights:
                log.log(distance=current_distance, w=w, learning_rate=learning_rate)
            else:
                log.log(distance=current_distance, learning_rate=learning_rate)
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

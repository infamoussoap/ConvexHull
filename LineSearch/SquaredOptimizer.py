import numpy as np
import sys

from .Log import Log
from .KKTConditions import validate_kkt_conditions


class SquaredOptimizer:
    def __init__(self, points):
        self.points = points
        self.log = None

    def optimize(self, y, learning_rate='cauchy', w=None, kkt_tol=1e-3, max_iter=1000,
                 log=None, verbose=True, log_weights=False, tol=1e-8):
        if log is None:
            log = Log()

        if w is None:
            w = np.ones(len(self.points)) / len(self.points)

        return self._optimize(y, learning_rate, w, kkt_tol, max_iter, log, verbose, log_weights)

    def _optimize(self, y, learning_rate, w, kkt_tol, max_iter, log, verbose, log_weights):
        status = None
        current_distance = np.sum((w @ self.points - y) ** 2)

        for count in range(max_iter):
            grad = (w @ self.points - y) @ self.points.T
            if validate_kkt_conditions(w, grad, kkt_tol):
                status = 'KKT'
                break

            dw_dt = w * (grad - w @ grad)
            if learning_rate == 'cauchy':
                mask = dw_dt > tol
                if np.any(mask):  # Check non-empty
                    max_learning_rate = np.min(w[mask] / dw_dt[mask])

                    cauchy_learning_rate_ = dw_dt @ grad / np.sum((dw_dt @ self.points) ** 2)
                    learning_rate_ = min(cauchy_learning_rate_, max_learning_rate)

            else:
                learning_rate_ = learning_rate

            w = w - learning_rate_ * dw_dt
            assert np.all(w > -tol), 'negative values not allowed'
            w = w / np.sum(w)

            current_distance = np.sum((w @ self.points - y) ** 2)

            # Callbacks
            if log_weights:
                log.log(distance=current_distance, w=w, learning_rate=learning_rate_)
            else:
                log.log(distance=current_distance, learning_rate=learning_rate_)
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

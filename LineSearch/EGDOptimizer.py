import numpy as np
import sys

from .Log import Log
from .BisectionMethod import BisectionMethod


class EGDOptimizer:
    def __init__(self, points):
        self.points = points
        self.log = None
        self.H = None

    def optimize(self, y, w=None, learning_rate=None, l2_reg=0.0,
                 search_method=None, tol=1e-7, max_iter=5000,
                 log=None, verbose=True):
        if log is None:
            log = Log()

        if w is None:
            w = np.ones(len(self.points)) / len(self.points)

        if search_method is None:
            search_method = BisectionMethod(self.points)

        return self._optimize(y, w, learning_rate, l2_reg, search_method, tol,
                              max_iter, log, verbose)


    def _optimize(self, y, w, learning_rate, l2_reg, search_method, tol, max_iter,
                  log, verbose):
        current_distance = np.sum((w @ self.points - y) ** 2)
        grad_squared = 0
        for count in range(max_iter):
            # Weights are always positive, so only need to check one side
            l2_grad = l2_reg * w
            grad = (w @ self.points - y) @ self.points.T + l2_grad

            t0, t1 = 0, 2 / np.max(abs(grad))
            learning_rate = search_method.search(w, y, t0, t1)

            x = w * np.exp(-learning_rate * grad)
            w = x / np.sum(x)

            new_distance = np.sum((w @ self.points - y) ** 2)
            if abs(current_distance - new_distance) < tol:
                break
            else:
                current_distance = new_distance

            # Callbacks
            log.log(current_distance=current_distance, w=w)
            self.verbose_callback(verbose, current_distance, count, max_iter)

        return log, w

    def verbose_callback(self, verbose, current_distance, count, max_iter):
        if not verbose:
            return

        sys.stdout.write(f"{count + 1} of {max_iter}: Distance {current_distance}\r")
        sys.stdout.flush()

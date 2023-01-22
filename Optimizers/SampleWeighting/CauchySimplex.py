import numpy as np

from Optimizers import ArmijoSearch, Optimizer
from .SampleWeighting import SampleWeighting


class CauchySimplex(SampleWeighting, ArmijoSearch, Optimizer):
    def __init__(self, data, integration_points, target_distribution, e=0.01, tol=1e-10):
        SampleWeighting.__init__(self, data, integration_points, target_distribution, e=e)
        self.tol = tol

    def update(self, x, d, step_size):
        z = x - step_size * d
        z[x < self.tol] = 0

        return z / np.sum(z)

    def search(self, x, step_size=None, c1=1e-4, c2=0.5, max_iter=100):
        grad = self.f(x, grad=True)
        d = x * (grad - grad @ x)

        max_step_size = self.max_step_size(x, grad, tol=self.tol) if step_size is None else step_size

        step_size = self.backtracking_armijo_line_search(x, d, max_step_size,
                                                         c1=c1, c2=c2, max_iter=max_iter)

        return self.update(x, d, step_size)

    @staticmethod
    def max_step_size(x, grad, tol=1e-10):
        support = x > tol
        return 1 / (np.max(grad[support]) - x @ grad)

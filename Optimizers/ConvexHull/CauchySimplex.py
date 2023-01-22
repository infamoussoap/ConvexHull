import numpy as np

from Optimizers import ArmijoSearch, Optimizer
from .ConvexHull import ConvexHull
from Optimizers.utils import clip


class CauchySimplex(ConvexHull, ArmijoSearch, Optimizer):
    def __init__(self, X, y, tol=1e-10):
        ConvexHull.__init__(self, X, y)
        self.tol = tol

    def update(self, x, d, step_size):
        z = x - step_size * d
        z[x < self.tol] = 0

        return z / np.sum(z)

    def search(self, x, step_size=None, c1=1e-4, c2=0.5, max_iter=100):
        grad = self.f(x, grad=True)
        d = x * (grad - grad @ x)

        max_step_size = self.max_step_size(x, grad, tol=self.tol) if step_size is None else step_size
        cauchy_step_size = d @ grad / np.sum((d @ self.X) ** 2)

        step_size = clip(cauchy_step_size, 0, max_step_size)
        return self.update(x, d, step_size)

    @staticmethod
    def max_step_size(x, grad, tol=1e-10):
        support = x > tol
        return 1 / (np.max(grad[support]) - x @ grad)
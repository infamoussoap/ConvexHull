import numpy as np

from Optimizers import ArmijoSearch, Optimizer
from .SampleWeighting import SampleWeighting


class EGD(SampleWeighting, ArmijoSearch, Optimizer):
    def __init__(self, data, integration_points, target_distribution, e=0.01):
        SampleWeighting.__init__(self, data, integration_points, target_distribution, e=e)

    def update(self, x, d, step_size):
        z = x * np.exp(-step_size * d)
        return z / np.sum(z)

    def search(self, x, step_size=1, c1=1e-4, c2=0.5, max_iter=100):
        d = self.f(x, grad=True)
        step_size = self.backtracking_armijo_line_search(x, d, step_size,
                                                         c1=c1, c2=c2, max_iter=max_iter)

        return self.update(x, d, step_size)
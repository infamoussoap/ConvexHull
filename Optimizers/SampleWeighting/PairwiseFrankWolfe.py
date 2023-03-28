import numpy as np

from Optimizers import ArmijoSearch, Optimizer
from .SampleWeighting import SampleWeighting
from Optimizers.utils import clip


class PairwiseFrankWolfe(SampleWeighting, ArmijoSearch, Optimizer):
    def __init__(self, data, integration_points, target_distribution, e=0.01, tol=1e-10):
        SampleWeighting.__init__(self, data, integration_points, target_distribution, e=e)
        self.tol = tol

    def update(self, x, d, step_size):
        s_index, v_index = d

        w = x.copy()
        alpha = w[v_index]

        w[s_index] += step_size * alpha
        w[v_index] -= step_size * alpha

        return w

    def search(self, x, step_size=None, c1=1e-4, c2=0.5, max_iter=100):
        grad = self.f(x, grad=True)

        d = self.frank_wolfe_pair(grad, x, tol=self.tol)

        max_step_size = 1 if step_size is None else step_size

        step_size = self.backtracking_armijo_line_search(x, d, max_step_size,
                                                         c1=c1, c2=c2, max_iter=max_iter)

        return self.update(x, d, step_size)

    @staticmethod
    def frank_wolfe_pair(grad, w, tol=1e-10):
        s_index = np.argmin(grad)

        non_active_set = w > tol
        v_masked_index = np.argmax(grad[non_active_set])
        v_index = np.argwhere(non_active_set).flatten()[v_masked_index]

        d = [s_index, v_index]

        return d

import numpy as np

from Optimizers import ArmijoSearch, Optimizer
from .ConvexHull import ConvexHull
from Optimizers.utils import clip


class PairwiseFrankWolfe(ConvexHull, ArmijoSearch, Optimizer):
    def __init__(self, X, y, tol=1e-10):
        ConvexHull.__init__(self, X, y)
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

        s_index, v_index = self.frank_wolfe_pair(grad, x, tol=self.tol)

        alpha = x[v_index]
        d_X = alpha * (self.X[s_index] - self.X[v_index])

        cauchy_step_size = - (x @ self.X - self.y) @ d_X / (d_X @ d_X)
        step_size = clip(cauchy_step_size, 0, 1)

        return self.update(x, [s_index, v_index], step_size)

    @staticmethod
    def frank_wolfe_pair(grad, w, tol=1e-10):
        s_index = np.argmin(grad)

        non_active_set = w > tol
        v_masked_index = np.argmax(grad[non_active_set])
        v_index = np.argwhere(non_active_set).flatten()[v_masked_index]

        d = [s_index, v_index]

        return d

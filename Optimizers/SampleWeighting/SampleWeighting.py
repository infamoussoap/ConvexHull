import numpy as np

from .Distributions import TruncatedUnitNormal


class SampleWeighting:
    def __init__(self, data, integration_points, target_distribution, base_distribution, e=0.01):
        self.n, self.d = data.shape

        self.data = data

        self.integration_points = integration_points[:-1]
        self.dx = integration_points[1:] - integration_points[:-1]

        self.target_distribution = target_distribution

        self.e = e

        self.base_distribution = base_distribution

    def f(self, w, grad=False):
        rho = self.rho(w)
        rho_grad = self.rho(w, grad=True) if grad else None
        f = self.target_distribution(self.integration_points)
        dx = self.dx

        if grad:
            return (dx * (self.safe_log(rho / f) + 1)) @ rho_grad

        return (rho * self.safe_log(rho / f)) @ dx

    def __call__(self, w, grad=False):
        return self.f(w, grad=grad)

    def rho(self, w, grad=False):
        """ Kernel Density Estimation evaluated at the integration_points """
        X = self.data @ w
        x = self.integration_points

        if grad:
            C = - 1 / (self.n * (self.e ** 2))
            return C * self.base_distribution(x, X, self.e, grad=True) @ self.data

        return np.mean(self.base_distribution(x, X, self.e), axis=1) / self.e

    @staticmethod
    def safe_log(x):
        """ x is assumed to be np.ndarray """
        mask = x > 0
        if np.all(mask):
            return np.log(x)

        out = np.zeros_like(x)
        out[mask] = np.log(x[mask])

        return out

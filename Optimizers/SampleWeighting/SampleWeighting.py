import numpy as np

from .Distributions import UnitNormal


class SampleWeighting:
    normal = UnitNormal()

    def __init__(self, data, integration_points, target_distribution, e=0.01):
        self.n, self.d = data.shape

        self.data = data

        self.integration_points = integration_points[:-1]
        self.dx = integration_points[1:] - integration_points[:-1]

        self.target_distribution = target_distribution

        self.e = e

    def f(self, w, grad=False):
        rho = self.rho(w)
        f = self.target_distribution(self.integration_points)

        if grad:
            return (self.dx * (np.log(rho / f) + f)) @ self.rho(w, grad=True)

        return (rho * np.log(rho / f)) @ self.dx

    def __call__(self, w, grad=False):
        return self.f(w, grad=grad)

    def rho(self, w, grad=False):
        """ Kernel Density Estimation evaluated at the integration_points """
        X = self.data @ w
        x = self.integration_points

        if grad:
            C = -1 / (self.n * (self.e ** 2))
            return C * self.normal((x[:, None] - X[None, :]) / self.e, grad=True) @ self.data

        return np.mean(self.normal((x - X[:, None]) / self.e), axis=0) / self.e

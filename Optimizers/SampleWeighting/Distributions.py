import numpy as np
from scipy.special import erf


class UnitNormal:
    C = np.sqrt(2 * np.pi)

    def __call__(self, x, mu, std, grad=False):
        """ Returns the pdf evaluated at the x-points

            Parameters
            ----------
            x : (n,) np.ndarray
            mu : (m,) np.ndarray
            std : float
        """
        scaled_x = (x[:, None] - mu[None, :]) / std

        if grad:
            return -(scaled_x / self.C) * np.exp(-0.5 * (scaled_x ** 2))

        return np.exp(-0.5 * (scaled_x ** 2)) / self.C


class TruncatedUnitNormal:
    unit_normal = UnitNormal()
    
    def __init__(self, a=0, b=1):
        self.a = a
        self.b = b
        
        self.c = self._Phi(self.b) - self._Phi(self.a)

    def __call__(self, x, mu, std, grad=False):
        """ Returns the pdf evaluated at the x-points

            Parameters
            ----------
            x : (n,) np.ndarray
            mu : (m,) np.ndarray
            std : float
        """
        # c = std * (self._Phi((self.b - mu) / std) - self._Phi((self.a - mu) / std))
        c = self._Phi((self.b - mu) / std) - self._Phi((self.a - mu) / std)

        mask = (self.a <= x) * (x <= self.b)
        return self.unit_normal(x, mu, std, grad=grad) * mask[:, None] / c[None, :]

    @staticmethod
    def _Phi(x):
        return (1 + erf(x / np.sqrt(2))) / 2


class Gaussian:
    C = np.sqrt(2 * np.pi)

    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def __call__(self, x, grad=False):
        if grad:
            return ((self.mu - x) / (self.C * self.std ** 3)) * np.exp(-0.5 * ((x - self.mu) / self.std) ** 2)

        return np.exp(-0.5 * ((x - self.mu) / self.std) ** 2) / (self.std * self.C)

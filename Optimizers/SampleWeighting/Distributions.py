import numpy as np
import math


class UnitNormal:
    C = np.sqrt(2 * np.pi)

    def __call__(self, x, grad=False):
        if grad:
            return -(x / self.C) * np.exp(-0.5 * (x ** 2))

        return np.exp(-0.5 * (x ** 2)) / self.C


class TruncatedUnitNormal:
    unit_normal = UnitNormal()
    
    def __init__(self, a=0, b=1):
        self.a = a
        self.b = b
        
        self.c = self._Phi(self.b) - self._Phi(self.a)

    def __call__(self, x, grad=False):
        mask = (self.a <= x) * (x <= self.b)
        return self.unit_normal(x, grad=grad) * mask / self.c

    @staticmethod
    def _Phi(x):
        return (1 + math.erf(x / math.sqrt(2))) / 2


class Gaussian:
    C = np.sqrt(2 * np.pi)

    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def __call__(self, x, grad=False):
        if grad:
            return ((self.mu - x) / (self.C * self.std ** 3)) * np.exp(-0.5 * ((x - self.mu) / self.std) ** 2)

        return np.exp(-0.5 * ((x - self.mu) / self.std) ** 2) / (self.std * self.C)

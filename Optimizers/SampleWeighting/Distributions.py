import numpy as np


class UnitNormal:
    C = np.sqrt(2 * np.pi)

    def __call__(self, x, grad=False):
        if grad:
            return -(x / self.C) * np.exp(-0.5 * (x ** 2))

        return np.exp(-0.5 * (x ** 2)) / self.C


class Gaussian:
    C = np.sqrt(2 * np.pi)

    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def __call__(self, x, grad=False):
        if grad:
            return ((self.mu - x) / (self.C * self.std ** 3)) * np.exp(-0.5 * ((x - self.mu) / self.std) ** 2)

        return np.exp(-0.5 * ((x - self.mu) / self.std) ** 2) / (self.std * self.C)

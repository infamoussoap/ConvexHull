import numpy as np

from Optimizers import ArmijoSearch, Optimizer
from .ConvexHull import ConvexHull
from Optimizers.utils import clip


class CauchySimplex(ConvexHull, ArmijoSearch, Optimizer):
    """ Projection onto a convex hull using the Cauchy-Simplex Optimizer

        Attributes
        ----------
        X : (n, d) np.ndarray
            The n-points that make up the convex hull
        y : (d, ) np.ndarray
            The point to project into the hull
        tol : float
            The tolerance for the zero-set
    """
    def __init__(self, X, y, tol=1e-10):
        """ Initialize Cauchy-Simplex Optimizer Class

            Parameters
            ----------
            X : (n, d) np.ndarray
                The n-points that make up the convex hull
            y : (d, ) np.ndarray
                The point to project into the hull
            tol : float
                The tolerance for the zero-set
        """
        ConvexHull.__init__(self, X, y)
        self.tol = tol

    def update(self, x, d, step_size):
        """ Perform a step using the Cauchy-Simplex scheme

            Parameters
            ----------
            x : (n, ) np.ndarray
                The starting point to take the step
            d : (n, ) np.ndarray
                The direction to take
            step_size : float

            Returns
            -------
            (n, ) np.ndarray
                The point after the step has been taken
        """
        if np.sum(x > 0) == 1:
            return x

        z = x - step_size * d
        z[x < self.tol] = 0

        return z / np.sum(z)

    def search(self, x, gamma=1):
        """ Perform a step using the Cauchy-Simplex scheme using the optimal step size

            Parameters
            ----------
            x : (n, ) np.ndarray
                The starting point to take the step
            gamma : float
                Expected to be a float between [0, 1]. It represents the percent of the maximum step size
                to be taken.

            Returns
            -------
            (n, ) np.ndarray
                The point after the step has been taken

            Notes
            -----
            The step size is determined through a line search
        """
        grad = self.f(x, grad=True)
        d = x * (grad - grad @ x)

        max_step_size = self.max_step_size(x, grad, tol=self.tol) * gamma
        
        cauchy_step_size = d @ grad / np.sum((d @ self.X) ** 2)

        step_size = clip(cauchy_step_size, 0, max_step_size)
        return self.update(x, d, step_size)

    @staticmethod
    def max_step_size(x, grad, tol=1e-10):
        """ Compute the maximum step size

            Parameters
            ----------
            x : (n, ) np.ndarray
                A point in the probability simplex
            grad : (n, ) np.ndarray
                Gradient at the point `x`
            tol : float
                Tolerance for the zero set

            Returns
            -------
            float
        """
        support = x > tol

        diff = np.max(grad[support]) - x @ grad
        return 1 / diff if diff > 1e-6 else 1e6

import numpy as np

from Optimizers import ArmijoSearch, Optimizer
from .ConvexHull import ConvexHull


class EGD(ConvexHull, ArmijoSearch, Optimizer):
    """ Projection onto a convex hull using the Exponentiated Gradient Descent Optimizer

        Attributes
        ----------
        X : (n, d) np.ndarray
            The n-points that make up the convex hull
        y : (d, ) np.ndarray
            The point to project into the hull
    """
    def __init__(self, X, y):
        """ Initialize EGD Optimizer Class

            Parameters
            ----------
            X : (n, d) np.ndarray
                The n-points that make up the convex hull
            y : (d, ) np.ndarray
                The point to project into the hull
        """
        ConvexHull.__init__(self, X, y)

    def update(self, x, d, step_size):
        """ Perform a step using the EGD scheme

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
        z = x * np.exp(-step_size * d)
        return z / np.sum(z)

    def search(self, x, step_size=1, c1=1e-4, c2=0.5, max_iter=100):
        """ Perform a step using the Cauchy-Simplex scheme using the optimal step size

            Parameters
            ----------
            x : (n, ) np.ndarray
                The starting point to take the step
            step_size : float
                The maximum candidate step size
            c1 : float
                Parameter for the armijo line search
            c2 : float
                Parameter for the armijo line search
            max_iter : int
                Maximum iterations for the armijo line search

            Returns
            -------
            (n, ) np.ndarray
                The point after the step has been taken

            Notes
            -----
            The step size is determined through an Armijo line search
        """
        d = self.f(x, grad=True)
        step_size = self.backtracking_armijo_line_search(x, d, step_size,
                                                         c1=c1, c2=c2, max_iter=max_iter)

        return self.update(x, d, step_size)

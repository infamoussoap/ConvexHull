import numpy as np

from Optimizers import ArmijoSearch, Optimizer
from .ConvexHull import ConvexHull
from Optimizers.utils import clip


class PairwiseFrankWolfe(ConvexHull, ArmijoSearch, Optimizer):
    """ Projection onto a convex hull using the Pairwise Frank-Wolfe Optimizer

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
        """ Initialize PFW Optimizer Class

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
        """ Perform a step using the PFW scheme

            Parameters
            ----------
            x : (n, ) np.ndarray
                The starting point to take the step
            d : tuple
                2-tuple containing the ('to', 'from') index pair
            step_size : float

            Returns
            -------
            (n, ) np.ndarray
                The point after the step has been taken

            Notes
            -----
            In the pairwise frank-wolfe, mass is taken from the 'from' index and
            given to the 'to' index
        """
        s_index, v_index = d

        w = x.copy()
        alpha = w[v_index]

        w[s_index] += step_size * alpha
        w[v_index] -= step_size * alpha

        return w

    def search(self, x):
        """ Perform a step using the PFW scheme using the optimal step size

            Parameters
            ----------
            x : (n, ) np.ndarray
                The starting point to take the step

            Returns
            -------
            (n, ) np.ndarray
                The point after the step has been taken

            Notes
            -----
            The step size is determined through a line search
        """
        grad = self.f(x, grad=True)

        s_index, v_index = self.frank_wolfe_pair(x, grad, tol=self.tol)

        alpha = x[v_index]
        d_X = alpha * (self.X[s_index] - self.X[v_index])

        cauchy_step_size = - (x @ self.X - self.y) @ d_X / (d_X @ d_X)
        step_size = clip(cauchy_step_size, 0, 1)

        return self.update(x, (s_index, v_index), step_size)

    @staticmethod
    def frank_wolfe_pair(x, grad, tol=1e-10):
        """ Returns the 'from' and 'to' index pair used in the PFW algorithm

            Parameters
            ----------
            x : (n, ) np.ndarray
                A point in the probability simplex
            grad : (n, ) np.ndarray
                Gradient at the point `x`
            tol : float
                Tolerance for the zero-set

            Returns
            -------
            tuple
                Tuple containing the ('to', 'from') index pair
        """
        s_index = np.argmin(grad)

        non_active_set = x > tol
        v_masked_index = np.argmax(grad[non_active_set])
        v_index = np.argwhere(non_active_set).flatten()[v_masked_index]

        return s_index, v_index

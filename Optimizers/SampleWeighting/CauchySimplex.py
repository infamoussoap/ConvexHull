import numpy as np

from Optimizers import ArmijoSearch, Optimizer
from .SampleWeighting import SampleWeighting

from .Distributions import TruncatedUnitNormal


class CauchySimplex(SampleWeighting, ArmijoSearch, Optimizer):
    """ Optimal question weighting using the Cauchy-Simplex Optimizer

        Attributes
        ----------
        data : (d, n) np.ndarray
            Array containing the student marks. Assumed to be d students and n questions
        integration_points : (m, ) np.ndarray
            Points to evaluate the target and base distribution at
        target_distribution : Distributions.TruncatedGaussian or Distributions.Gaussian
        base_distribution : Distributions.TruncatedUnitNormal or Distributions.UnitNormal
            The base distribution used in the kernel density approximation
        e : float
            The scaling parameter for the kernel density approximation
        tol : float
            Tolerance for the zero-set
    """
    def __init__(self, data, integration_points, target_distribution, base_distribution=TruncatedUnitNormal(),
                 e=0.01, tol=1e-10):
        """ Initialize the Cauchy-Simplex Optimizer class

            Parameters
            ----------
            data : (d, n) np.ndarray
                Array containing the student marks. Assumed to be d students and n questions
            integration_points : (m + 1, ) np.ndarray
                Points to evaluate the target and base distribution at
            target_distribution : Distributions.TruncatedGaussian or Distributions.Gaussian
            base_distribution : Distributions.TruncatedUnitNormal or Distributions.UnitNormal
                The base distribution used in the kernel density approximation
            e : float
                The scaling parameter for the kernel density approximation
            tol : float
                Tolerance for the zero-set

            Notes
            -----
            The given `integration_points` is not the same as the `integration_points` attribute, as we
            only store `integration_points[:-1]`, that is, everything except the last point
        """
        SampleWeighting.__init__(self, data, integration_points, target_distribution,
                                 base_distribution=base_distribution, e=e)
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
        z = x - step_size * d
        z[x < self.tol] = 0

        return z / np.sum(z)

    def search(self, x, c1=1e-4, c2=0.5, max_iter=100, gamma=1):
        """ Perform a step using the Cauchy-Simplex scheme using the optimal step size

            Parameters
            ----------
            x : (n, ) np.ndarray
                The starting point to take the step
            c1 : float
                Parameter for the armijo line search
            c2 : float
                Parameter for the armijo line search
            max_iter : int
                Maximum iterations for the armijo line search
            gamma : float
                Expected to be a float between [0, 1]. It represents the percent of the maximum step size
                to be taken.

            Returns
            -------
            (n, ) np.ndarray
                The point after the step has been taken

            Notes
            -----
            The step size is determined through an Armijo line search
        """
        grad = self.f(x, grad=True)
        d = x * (grad - grad @ x)

        max_step_size = self.max_step_size(x, grad, tol=self.tol) * gamma

        step_size = self.backtracking_armijo_line_search(x, d, max_step_size,
                                                         c1=c1, c2=c2, max_iter=max_iter)

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
        return 1 / (np.max(grad[support]) - x @ grad)

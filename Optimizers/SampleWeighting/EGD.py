import numpy as np

from Optimizers import ArmijoSearch, Optimizer
from .SampleWeighting import SampleWeighting

from .Distributions import TruncatedUnitNormal


class EGD(SampleWeighting, ArmijoSearch, Optimizer):
    """ Optimal question weighting using the EGD Optimizer

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
    """
    def __init__(self, data, integration_points, target_distribution, base_distribution=TruncatedUnitNormal(), e=0.01):
        """ Initialize the EGD Optimizer class

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

            Notes
            -----
            The given `integration_points` is not the same as the `integration_points` attribute, as we
            only store `integration_points[:-1]`, that is, everything except the last point
        """
        SampleWeighting.__init__(self, data, integration_points, target_distribution,
                                 base_distribution=base_distribution, e=e)

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

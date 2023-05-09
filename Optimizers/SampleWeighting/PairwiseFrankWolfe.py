import numpy as np

from Optimizers import ArmijoSearch, Optimizer
from .SampleWeighting import SampleWeighting

from .Distributions import TruncatedUnitNormal


class PairwiseFrankWolfe(SampleWeighting, ArmijoSearch, Optimizer):
    """ Optimal question weighting using the PFW Optimizer

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
        """ Initialize the PFW Optimizer class

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
        """
        s_index, v_index = d

        w = x.copy()
        alpha = w[v_index]

        w[s_index] += step_size * alpha
        w[v_index] -= step_size * alpha

        return w

    def search(self, x, c1=1e-4, c2=0.5, max_iter=100):
        """ Perform a step using the PFW scheme using the optimal step size

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

            Returns
            -------
            (n, ) np.ndarray
                The point after the step has been taken

            Notes
            -----
            The step size is determined through an Armijo line search
        """
        grad = self.f(x, grad=True)

        d = self.frank_wolfe_pair(x, grad, tol=self.tol)

        max_step_size = 1
        step_size = self.backtracking_armijo_line_search(x, d, max_step_size,
                                                         c1=c1, c2=c2, max_iter=max_iter)

        return self.update(x, d, step_size)

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

        d = [s_index, v_index]

        return d

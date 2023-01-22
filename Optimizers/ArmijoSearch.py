from abc import ABC
from abc import abstractmethod


class ArmijoSearch(ABC):
    def armijo_condition(self, x_old, x_new, f_old=None, grad_old=None, c1=1e-4):
        """ Returns True if the Armijo condition is satisfied

            Parameters
            ----------
            x_old : np.ndarray
                The current position of the search
            x_new : np.ndarray
                The suggested position of the search
            c1 : float
                The constant in the Armijo condition
            f_old : float, optional
                The function evaluated at x_old
            grad_old : np.ndarray, optional
                The gradient of the function evaluated at x_old
        """

        f_old = self.f(x_old) if f_old is None else f_old
        grad_old = self.f(x_old, grad=True) if grad_old is None else grad_old

        return self.f(x_new) <= f_old + c1 * grad_old @ (x_new - x_old)

    def backtracking_armijo_line_search(self, x, d, step_size, c1=1e-4, c2=0.5, max_iter=100):
        """ Returns the step_size that satisfies the Armijo condition

            Parameters
            ----------
            x : np.ndarray
                The current position of the search
            step_size : float
                The initial step_size to try
            c1 : float
                The constant in the Armijo condition
            c2 : float
                The amount to decrease the step_size at each iteration
            max_iter : int
                The maximum number of steps to try
        """
        f0 = self.f(x)
        grad0 = self.f(x, grad=True)

        count = 0
        x_new = self.update(x, d, step_size)

        while (not self.armijo_condition(x, x_new, f_old=f0, grad_old=grad0, c1=c1)) and count < max_iter:
            step_size = step_size * c2
            x_new = self.update(x, d, step_size)

            count += 1

        if count == max_iter:
            if f0 < self.f(x_new):
                return 0

        return step_size

    @abstractmethod
    def f(self, x, grad=False):
        """ The function to be minimised """
        pass

    @abstractmethod
    def update(self, x, d, step_size):
        """ Returns the new position according to the update rule """
        pass

    @abstractmethod
    def search(self, *args):
        """ Perform the search """
        pass

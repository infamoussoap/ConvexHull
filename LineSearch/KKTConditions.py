import numpy as np


def kkt_conditions(w, grad, tol=1e-5):
    non_active_set = w > tol

    non_active_grad = grad[non_active_set]
    active_grad = grad[~non_active_set]

    valid, b = check_kkt_conditions_for_non_active_set(non_active_grad, tol=tol)
    if valid:
        return check_kkt_conditions_for_active_set(active_grad, b, tol=1e-5)

    return False


def check_kkt_conditions_for_non_active_set(non_active_grad, tol=1e-5):
    """ KKT conditions for the non-active-set requires that the respective gradients
        are equal

        Note: Can probably make this faster by using the single pass variance
    """

    b = np.mean(non_active_grad) # The Lagrange Multiplier
    return np.max(abs(non_active_grad - b)) < tol, -b

def check_kkt_conditions_for_active_set(active_grad, b, tol=1e-5):
    """ KKT conditions for the active set requires that the respective
        gradients + b > 0, where b is the lagrange multipler computed from the
        non active set, 
    """
    return np.all((active_grad + b) > -tol)

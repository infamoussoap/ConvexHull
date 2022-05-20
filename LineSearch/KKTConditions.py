import numpy as np


def validate_kkt_conditions(w, grad, tol=1e-5):
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
    if len(non_active_grad) == 0:
        # This implies w1,...,wn=0 which violates the KKT condition dL/db = 0
        return False


    b = np.mean(non_active_grad) # The Lagrange Multiplier
    return np.max(abs(non_active_grad - b)) < tol, -b

def check_kkt_conditions_for_active_set(active_grad, b, tol=1e-5):
    """ KKT conditions for the active set requires that the respective
        gradients + b > 0, where b is the lagrange multipler computed from the
        non active set,
    """
    if len(active_grad) == 0:
        # The assumption is that the kkt conditions is valid for the
        # non-active-set. As such all the lagrange multipliers are satisfied
        return True

    return np.all((active_grad + b) > -tol)

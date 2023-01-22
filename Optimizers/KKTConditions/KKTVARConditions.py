import numpy as np


def validate_kkt_var_conditions(w, grad, tol=1e-3, e=1e-10):
    active_set = w <= e
    active_grad = grad[active_set]

    valid, b = check_kkt_var_conditions_for_non_active_set(w, grad, tol=tol)
    if valid:
        return check_kkt_conditions_for_active_set(active_grad, b, tol=tol)

    return False


def check_kkt_var_conditions_for_non_active_set(w, grad, tol=1e-3):
    """ KKT conditions for the non-active-set requires that the respective gradients
        are equal

        Note: Can probably make this faster by using the single pass variance
    """
    b = grad @ w  # The Lagrange Multiplier
    return np.sqrt((grad ** 2) @ w - (b ** 2)) < tol, -b


def check_kkt_conditions_for_active_set(active_grad, b, tol=1e-3):
    """ KKT conditions for the active set requires that the respective
        gradients + b > 0, where b is the lagrange multipler computed from the
        non active set,
    """
    if len(active_grad) == 0:
        # The assumption is that the kkt conditions is valid for the
        # non-active-set. As such all the lagrange multipliers are satisfied
        return True

    return np.all((active_grad + b) > -tol)

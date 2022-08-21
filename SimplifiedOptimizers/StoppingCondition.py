import numpy as np

from .KKTConditions import validate_kkt_conditions
from .TolConditions import validate_tol_conditions


def validate_stopping_conditions(w, X, y, tol=1e-6, e=1e-10, stopping_type="TOL"):
    if stopping_type == "KKT":
        grad = (w @ X - y) @ X.T
        return validate_kkt_conditions(w, grad, tol=tol, e=e)

    elif stopping_type == "TOL":
        return validate_tol_conditions(w, X, y, tol=tol)

    else:
        raise ValueError("stopping_type can only be KKT or TOL.")
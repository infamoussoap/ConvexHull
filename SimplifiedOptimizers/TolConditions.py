import numpy as np


def validate_tol_conditions(w, X, y, tol=1e-6):
    distance = np.sum((w @ X - y) ** 2) / 2

    return distance < tol

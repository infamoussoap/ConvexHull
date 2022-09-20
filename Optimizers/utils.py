import numpy as np
import sys


def verbose_callback(count, max_iter, w, points, y):
    distance = np.sum((w @ points - y) ** 2) / 2

    if max_iter > 0:
        sys.stdout.write(f'\rIter {count + 1} of {max_iter}: Distance to Hull: {distance:.5e}')
    else:
        sys.stdout.write(f'\rIter {count + 1}: Distance to Hull: {distance:.5e}')

    sys.stdout.flush()


def project_onto_standard_simplex(y):
    """ https://gist.github.com/mgritter/4bf003cd399da2e57096af1050d64ddd """
    n = len(y)
    y_s = sorted(y, reverse=True)

    sum_y = 0
    for i, y_i, y_next in zip(range(1, n+1), y_s, y_s[1:] + [0.0]):
        sum_y += y_i
        t = (sum_y - 1) / i
        if t >= y_next:
            break

    return np.maximum(0, y - t)


def clip(val, min_val, max_val):
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    return val

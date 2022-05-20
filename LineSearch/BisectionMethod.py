import numpy as np


class BisectionMethod:
    def __init__(self, points, resolution=1e-4, tol=1e-4, max_iter=100):
        self.points = points

        self.resolution = resolution
        self.tol = tol
        self.max_iter = max_iter

    def search(self, w, y, t_0, t_1, grad, search_type='classical', **search_kwargs):
        if search_type == 'classical':
            return self._classical_search(w, y, t_max, grad, max_iter, **search_kwargs)

        raise ValueError(f"{search_type} invalid.")

    def _classical_search(self, w, y, t_max, grad, max_iter, e=1e-5):
        q_0 = self.merit(w, y, grad, 0)

        t_left = 0
        t_right = t_max

        for i in range(max_iter):
            t_new = (t_left + t_right) / 2
            q_new = self.merit(w, y, grad, t_new)
            dq_new = self.merit_grad(w, y, grad, t_new)

            if q_new < q_0 and abs(dq_new) <= e:
                return t_new
            elif q_new >= q_0 or dq_new > e:
                t_right = t_new
            else:
                t_left = t_new

        return [0, t_left, t_right][np.argmin([q_0, q_left, q_right])]

    def cost(self, w, y):
        return np.sum((w @ self.points - y) ** 2)

    def grad(self, w, y):
        return (w @ self.points - y) @ self.points.T

    def merit(self, w, y, grad, t):
        w_new = self.step(w, grad, t)

        return np.sum((w_new @ self.points - y) ** 2)

    def merit_grad(self, w, y, grad, t):
        w_new = self.step(w, grad, t)
        w_grad = self.w_grad(w, grad, t)

        return (w_new @ self.points - y) @ self.points.T @ w_grad

    @staticmethod
    def w_grad(w, grad, t):
        z = -t * grad
        x_new = w * np.exp(z - np.max(z))
        gx_new = grad * x_new

        Z = np.sum(x_new)
        gZ = np.sum(gx_new)

        return (x_new * gZ - gx_new * Z) / (Z ** 2)

    @staticmethod
    def step(w, grad, t):
        z = -t * grad
        x_new = w * np.exp(z - np.max(z))  # For numerical stability
        return x_new / np.sum(x_new)

import numpy as np


class BisectionMethod:
    def __init__(self, points, resolution=1e-4, tol=1e-4, max_iter=100):
        self.points = points

        self.resolution = resolution
        self.tol = tol
        self.max_iter = max_iter

    def search(self, w, y, t_max, grad, search_type='classical', **search_kwargs):
        if search_type == 'classical':
            return self._classical_search(w, y, t_max, grad, self.max_iter, **search_kwargs)
        elif search_type == 'wolfe':
            return self._wolfe_search(w, y, t_max, grad, self.max_iter, **search_kwargs)
        elif search_type == 'goldstein':
            return self._goldstein_search(w, y, t_max, grad, self.max_iter, **search_kwargs)

        raise ValueError(f"{search_type} invalid. Try one of ['classical', 'wolfe', 'goldstein'].")

    def _classical_search(self, w, y, t_max, grad, max_iter, e=1e-5):
        q_0 = self.merit(w, y, grad, 0)

        t_left = 0
        t_right = t_max

        q_left = self.merit(w, y, grad, t_left)
        q_right = self.merit(w, y, grad, t_right)

        for i in range(max_iter):
            t_new = (t_left + t_right) / 2
            q_new = self.merit(w, y, grad, t_new)
            dq_new = self.merit_grad(w, y, grad, t_new)

            if q_new < q_0 and abs(dq_new) <= e:
                return t_new
            elif q_new >= q_0 or dq_new > e:
                t_right = t_new
                q_right = q_new
            else:
                t_left = t_new
                q_left = q_new

        return [0, t_left, t_right][np.argmin([q_0, q_left, q_right])]

    def _wolfe_search(self, w, y, t_max, grad, max_iter, m1=1e-4, m2=0.9):
        # Using the values from Numerical Optimization
        # (Jorge Nocedal and Stephen J. Wright) pg 60

        q_0 = self.merit(w, y, grad, 0)
        dq_0 = self.merit_grad(w, y, grad, 0)

        t_left = 0
        t_right = t_max

        q_left = self.merit(w, y, grad, t_left)
        q_right = self.merit(w, y, grad, t_right)

        for i in range(max_iter):
            t_new = (t_left + t_right) / 2
            q_new = self.merit(w, y, grad, t_new)
            dq_new = self.merit_grad(w, y, grad, t_new)

            upper_bound = q_0 + m1 * t_new * dq_0
            if q_new <= upper_bound and dq_new >= m2 * dq_0:
                return t_new
            elif q_new > upper_bound:
                t_right = t_new
                q_right = q_new
            else:
                t_left = t_new
                q_left = q_new

        return [0, t_left, t_right][np.argmin([q_0, q_left, q_right])]

    def _goldstein_search(self, w, y, t_max, grad, max_iter, m1=1e-4, m2=0.9, e=1e-8):
        q_0 = self.merit(w, y, grad, 0)
        dq_0 = self.merit_grad(w, y, grad, 0)

        m1_dq = dq_0 * m1
        m2_dq = dq_0 * m2

        t_left = 0
        t_right = t_max

        q_left = self.merit(w, y, grad, t_left)
        q_right = self.merit(w, y, grad, t_right)

        for i in range(max_iter):
            t_new = (t_left + t_right) / 2
            q_new = self.merit(w, y, grad, t_new)
            dq_new = (q_new - q_0) / (t_new + e)

            if m2_dq <= dq_new and dq_new <= m1_dq:
                return t_new
            elif m1_dq < dq_new:
                t_right = t_new
                q_right = q_new
            else:
                t_left = t_new
                q_left = q_new

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

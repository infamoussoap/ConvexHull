import numpy as np


class BisectionMethod:
    def __init__(self, points, resolution=1e-4, tol=1e-4, max_iter=100):
        self.points = points

        self.resolution = resolution
        self.tol = tol
        self.max_iter = max_iter

    def search(self, w, y, t0, t1):
        grad = self.grad(w, y)
        q_0 = self.merit(w, y, grad, t0)
        q_1 = self.merit(w, y, grad, t1)

        for i in range(self.max_iter):
            t_new = (t0 + t1) / 2
            q_new = self.merit(w, y, grad, t_new)
            dq_new = self.merit_grad(w, y, grad, t_new)

            if abs(t0 - t1) < self.resolution:
                break

            if q_new < q_0 and abs(dq_new) < self.tol:
                return t_new
            elif q_new < q_0 and dq_new > 0:
                t_0 = t_new
                q_0 = q_new
            elif q_new < q_1 and dq_new < 0:
                t_1 = t_new
                q_1 = q_new
            else:
                raise ValueError("Non Convex?")

        return [t_0, t_1, t_new][np.argmin([q_0, q_1, q_new])]


    def cost(self, w, y):
        return np.sum((w @ self.points - y) ** 2)

    def grad(self, w, y):
        return (w @ self.points - y) @ self.points.T

    def merit(self, w, y, grad, t):
        w_new = self.step(w, grad, t)

        return np.sum((w_new @ self.points - y) ** 2)

    def merit_grad(self, w, y, grad, t):
        w_new = self.step(w, grad, t)

        return (y - w_new @ self.points) @ self.points.T @ (grad * w_new)

    @staticmethod
    def step(w, grad, t):
        x_new = w * np.exp(-t * grad)
        return x_new / np.sum(x_new)

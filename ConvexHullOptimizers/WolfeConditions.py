import numpy as np


class WolfeConditions:
    def __init__(self, points):
        self.points = points

    def search(self, w, y, t0, t1, c1, c2, max_iter=100):
        g = self.grad(w, y)

        w0 = self.step(w, g, t0)
        w1 = self.step(w, g, t1)

        q0 = self.cost(w0, y)
        q1 = self.cost(w1, y)

        dq_0 = self.grad(w, y)

        for i in range(max_iter):
            t_new = (t0 + t1) / 2
            w_new = self.step(w, g, t_new)
            q_new = self.cost(w_new, y)

            raise NotImplementedError

    def cost(self, w, y):
        return np.sum((w @ self.points - y) ** 2)

    def grad(self, w, y):
        return (w @ self.points - y) @ self.points.T

    def merit(self, w, y, grad, t):
        w_new = self.step(w, grad, t)

        return np.sum((w_new @ self.points - y) ** 2)

    def merit_grad(self, w, y, grad, t):
        w_new = self.step(w, grad, t)

        return (y - w_new @ self.points) @ self.points.T @ (g * w_new)

    @staticmethod
    def step(w, grad, t):
        x_new = w * np.exp(-t * grad)
        return x_new / np.sum(x_new)

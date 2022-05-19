import numpy as np


class BisectionMethod:
    def __init__(self, points, resolution=1e-4, tol=1e-4, max_iter=100):
        self.points = points

        self.resolution = resolution
        self.tol = tol
        self.max_iter = max_iter

    def search(self, w, y, t_0, t_1, grad):
        q_0 = self.merit(w, y, grad, t_0)
        q_1 = self.merit(w, y, grad, t_1)

        for i in range(self.max_iter):
            t_new = (t_0 + t_1) / 2
            q_new = self.merit(w, y, grad, t_new)
            dq_new = self.merit_grad(w, y, grad, t_new)

            if abs(t_0 - t_1) < self.resolution:
                break

            if abs(dq_new) < self.tol:
                break

            if dq_new < 0:
                # The derivative is approximate, so not always
                # will be correct
                if q_new < q_0:
                    t_0 = t_new
                    q_0 = q_new
                else:
                    break
            else:
                if q_new < q_1:
                     t_1 = t_new
                     q_1 = q_new
                else:
                    break

        val = [t_0, t_1, t_new][np.argmin([q_0, q_1, q_new])]
        # print(val)
        return val


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
        x_new = w * np.exp(-t * grad)
        gx_new = grad * x_new

        Z = np.sum(x_new)
        gZ = np.sum(gx_new)

        return (x_new * gZ - gx_new * Z) / (Z ** 2)

    @staticmethod
    def step(w, grad, t):
        x_new = w * np.exp(-t * grad)
        return x_new / np.sum(x_new)

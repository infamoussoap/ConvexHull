class ConvexHull:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def f(self, x, grad=False):
        z = x @ self.X - self.y

        if grad:
            return z @ self.X.T
        return (z @ z) / 2

    def __call__(self, x, grad=False):
        return self.f(x, grad=grad)

import numpy as np


class Tape:
    def __init__(self, tol=1e-8):
        self.tol = tol
        self.w = None

    def watch(self, w):
        if self.w is None:
            self.w = w
            return False

        state = np.max(abs(self.w - w)) < self.tol
        self.w = w

        return state

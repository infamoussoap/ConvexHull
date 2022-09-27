import numpy as np
import pandas as pd

import time


class OptimizerTest:
    def __init__(self, optimizer, support_lengths, num_trials=20, max_iter=10_000,
                 tol=1e-6, e=1e-10, stopping_type='TOL'):

        self.optimizer = optimizer

        self.support_lengths = support_lengths

        self.num_trials = num_trials

        self.max_iter = max_iter

        self.tol = tol
        self.e = e
        self.stopping_type = stopping_type

    def run(self, X, filename):
        index = pd.MultiIndex.from_product([self.support_lengths, np.arange(self.num_trials)],
                                           names=["support length", "trial"])
        results = pd.DataFrame(0.0, index=index, columns=["Time", "Distance", "Steps"])

        for support_length in self.support_lengths:
            print(f"    Starting {filename}: with support length {support_length}")
            for i in range(self.num_trials):
                w_true, y = OptimizerTest.generate_on_hull(X, support_length, random_seed=i)

                time, distance, steps = self.test_optimizer(X, y)

                results.loc[(support_length, i)] = time, distance, steps

                results.to_csv(filename)
            print(f"    Finished {filename}: with support length {support_length}")

    def test_optimizer(self, X, y):
        start_time = time.time()
        results = self.optimizer(X, y, w=None, verbose=False, max_iter=self.max_iter,
                                 tol=self.tol, e=self.e, stopping_type=self.stopping_type)
        end_time = time.time()

        distance, num_steps, w = results

        return end_time - start_time, distance, num_steps

    @staticmethod
    def generate_on_hull(X, support_length, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        n = len(X)
        support_index = np.random.choice(n, size=support_length, replace=False)

        w_true = np.zeros(n)
        w_true[support_index] = np.random.rand(support_length)
        w_true = w_true / np.sum(w_true)

        return w_true, w_true @ X

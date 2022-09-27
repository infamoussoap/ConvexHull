import numpy as np

from Optimizers import egd_optimizer, cs_optimizer
from Optimizers import pfw_optimizer, fw_optimizer

from OptimizerTest import OptimizerTest


if __name__ == '__main__':
    a = np.arange(1, 10)
    total_support_lengths = np.concatenate([a, a * 10, a * 100, a * 1_000]).astype(int)

    optimizers = {'cs': cs_optimizer,
                  'fw': fw_optimizer,
                  'pfw': pfw_optimizer,
                  'egd': egd_optimizer}

    for n in [1000, 5000]:
        d = int(n * 0.1)

        np.random.seed(100)
        X = np.random.normal(size=(n, d))

        support_lengths = total_support_lengths[total_support_lengths < n]

        support_lengths = support_lengths[:4]

        for key, optimizer in optimizers.items():
            filename = f"{key}_{n}.csv"

            print(f"Starting {filename}")

            optimizer_test = OptimizerTest(optimizer, support_lengths, num_trials=3,
                                           max_iter=10_000, tol=1e-6, e=1e-10, stopping_type='TOL')

            optimizer_test.run(X, filename)

        print("FINISHED")

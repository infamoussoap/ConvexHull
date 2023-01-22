from abc import ABC
from abc import abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def update(self, x, d, step_size):
        pass

    @abstractmethod
    def search(self, x, step_size=1, c1=1e-4, c2=0.5, max_iter=100):
        pass

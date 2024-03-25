from typing import Protocol

import numpy as np
from cma.bbobbenchmarks import instantiate


class Problem(Protocol):
    id: int
    optimum_value: float
    lower: np.ndarray
    upper: np.ndarray
    bounds: np.ndarray

    def __call__(self, x: np.ndarray) -> float:
        ...


class BBOBProblem(Problem):
    _LOWER = -5
    _UPPER = 5

    def __init__(self, id: int, dim: int):
        self.function, self.optimum_value = instantiate(id)
        self.dim = dim
        self.id = id
        self.lower = np.full(dim, self._LOWER)
        self.upper = np.full(dim, self._UPPER)
        self.bounds = np.stack([self.lower, self.upper], axis=1).astype(float)

    def __call__(self, x: np.ndarray) -> float:
        return self.function(x)

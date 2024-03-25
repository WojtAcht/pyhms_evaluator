from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from ConfigSpace import ConfigurationSpace

from ..problem import Problem


@dataclass
class Solution:
    x: np.ndarray
    fitness: float
    problem: Problem
    optimizer_result: Any


class Solver(Protocol):
    def __init__(self, config: dict | None = None):
        ...

    def __call__(self, problem: Problem, max_n_evals: int, random_state: int) -> Solution:
        ...

    @property
    def configspace(self) -> ConfigurationSpace:
        ...

    @classmethod
    def from_config(cls):
        ...

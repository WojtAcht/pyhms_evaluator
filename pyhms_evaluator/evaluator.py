from time import time

import numpy as np
import pandas as pd
from multiprocess import Pool

from .problem import Problem
from .solver import Solver


class ProblemEvaluator:
    def __init__(
        self,
        solvers: list[Solver],
        problems: list[Problem],
        seed_count: int = 30,
        max_n_evals: int = 10000,
        max_pool: int = 10,
    ):
        self.seed_count = seed_count
        self.solvers = solvers
        self.max_n_evals = max_n_evals
        self.max_pool = max_pool
        self.problems = problems

    def evaluate_solver(
        self,
        solver: Solver,
        problem: Problem,
        max_n_evals: int,
    ) -> np.ndarray:
        fitness_values = []
        for random_state in range(1, self.seed_count + 1):
            try:
                solution = solver(problem, max_n_evals, random_state)
                fitness_values.append(solution.fitness - problem.optimum_value)
            except Exception as exc:
                print(f"{solver.__class__.__name__} failed, {exc}")
        return np.array(fitness_values)

    def evaluate_problem(self, problem: Problem) -> pd.DataFrame:
        start = time()
        rows = []
        for solver in self.solvers:
            fitness_values = self.evaluate_solver(solver, problem, self.max_n_evals)
            rows.append(
                {
                    "problem_id": problem.id,
                    "solver": solver.__class__.__name__,
                    "fitness_mean": np.mean(fitness_values),
                    "fitness_std": np.std(fitness_values),
                    "fitness_values": fitness_values,
                    "success_rate": np.mean(fitness_values < precision),
                }
            )
        end = time()
        print(f"Problem {problem.id} evaluated in {(end - start):.2f} seconds")
        return pd.DataFrame(rows)

    def __call__(self, precision: float | None = 1e-8) -> pd.DataFrame:
        with Pool(self.max_pool) as p:
            pool_outputs = p.map(
                lambda problem: self.evaluate_problem(problem, precision), self.problems
            )
        return pd.concat(pool_outputs)

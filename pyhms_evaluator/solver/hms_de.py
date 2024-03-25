import json
from typing import TypedDict

from ConfigSpace import ConfigurationSpace
from leap_ec.problem import FunctionProblem
from pyhms import (
    CMALevelConfig,
    DELevelConfig,
    DontStop,
    EvalCutoffProblem,
    MetaepochLimit,
    SingularProblemEvalLimitReached,
    hms,
)
from pyhms.sprout import DemeLimit, LevelLimit, NBC_FarEnough, NBC_Generator, SproutMechanism

from ..problem import Problem
from .solver import Solution, Solver


class HMSDEConfig(TypedDict, total=False):
    de_generations: int
    de_pop_size: int
    de_dither: bool
    de_scaling: float
    de_crossover: float
    cma_generations: int
    cma_sigma0: float | None
    cma_metaepochs: int
    nbc_cut: float
    nbc_trunc: float
    nbc_far: float
    level_limit: int


DEFAULT_HMS_DE_CONFIG: HMSDEConfig = {
    "de_generations": 2,
    "de_pop_size": 20,
    "de_dither": False,
    "de_scaling": 0.8,
    "de_crossover": 0.9,
    "cma_generations": 5,
    "cma_sigma0": None,
    "cma_metaepochs": 15,
    "nbc_cut": 3.0,
    "nbc_trunc": 0.7,
    "nbc_far": 3.0,
    "level_limit": 4,
}


class HMSDESolver(Solver):
    def __init__(self, config: HMSDEConfig = {}):
        self.config = DEFAULT_HMS_DE_CONFIG | config

    def __call__(
        self,
        problem: Problem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> Solution:
        function_problem = FunctionProblem(problem, maximize=False)
        problem_with_cutoff = EvalCutoffProblem(function_problem, max_n_evals)
        config = [
            DELevelConfig(
                generations=self.config["de_generations"],
                problem=problem_with_cutoff,
                bounds=problem.bounds,
                pop_size=self.config["de_pop_size"],
                lsc=DontStop(),
                dither=self.config["de_dither"],
                scaling=self.config["de_scaling"],
                crossover=self.config["de_crossover"],
            ),
            CMALevelConfig(
                generations=self.config["cma_generations"],
                problem=problem_with_cutoff,
                bounds=problem.bounds,
                sigma0=None,
                lsc=MetaepochLimit(self.config["cma_metaepochs"]),
            ),
        ]
        global_stop_condition = SingularProblemEvalLimitReached(max_n_evals)
        sprout_condition = SproutMechanism(
            NBC_Generator(self.config["nbc_cut"], self.config["nbc_trunc"]),
            [NBC_FarEnough(self.config["nbc_far"], 2), DemeLimit(1)],
            [LevelLimit(self.config["level_limit"])],
        )
        hms_tree = hms(
            config,
            global_stop_condition,
            sprout_condition,
            {"random_seed": random_state},
        )
        return Solution(
            hms_tree.best_individual.genome,
            hms_tree.best_individual.fitness,
            problem,
            hms_tree,
        )

    @property
    def configspace(self) -> ConfigurationSpace:
        return ConfigurationSpace(
            {
                "nbc_cut": (1.5, 4.0),
                "nbc_trunc": (0.1, 0.9),
                "nbc_far": (1.5, 4.0),
                "level_limit": (2, 10),
                "de_pop_size": (20, 300),
                "de_generations": (1, 10),
                "de_dither": [False, True],
                "de_scaling": (0.0, 2.0),
                "de_crossover": (0.0, 1.0),
                "cma_generations": (3, 30),
                "cma_metaepochs": (30, 300),
                "cma_sigma0": (0.1, 3.0),
            }
        )

    @classmethod
    def from_config(cls) -> "HMSDESolver":
        with open(f"config/{cls.__name__}.json", "r") as json_file:
            config = json.load(json_file)
        return cls(config)

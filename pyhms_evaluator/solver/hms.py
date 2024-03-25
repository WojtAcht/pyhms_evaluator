import json
from typing import TypedDict

from ConfigSpace import ConfigurationSpace
from leap_ec.problem import FunctionProblem
from pyhms import (
    SEA,
    CMALevelConfig,
    DontStop,
    EALevelConfig,
    EvalCutoffProblem,
    MetaepochLimit,
    SingularProblemEvalLimitReached,
    hms,
)
from pyhms.sprout import DemeLimit, LevelLimit, NBC_FarEnough, NBC_Generator, SproutMechanism

from ..problem import Problem
from .solver import Solution, Solver


class HMSConfig(TypedDict, total=False):
    ea_generations: int
    ea_pop_size: int
    ea_mutation_std: float
    cma_generations: int
    cma_sigma0: float | None
    cma_metaepochs: int
    nbc_cut: float
    nbc_trunc: float
    nbc_far: float
    level_limit: int


DEFAULT_HMS_CONFIG: HMSConfig = {
    "ea_generations": 2,
    "ea_pop_size": 20,
    "ea_mutation_std": 10.0,
    "cma_generations": 5,
    "cma_sigma0": None,
    "cma_metaepochs": 15,
    "nbc_cut": 3.0,
    "nbc_trunc": 0.7,
    "nbc_far": 3.0,
    "level_limit": 4,
}


class HMSSolver(Solver):
    def __init__(self, config: HMSConfig = {}):
        self.config = DEFAULT_HMS_CONFIG | config

    def __call__(
        self,
        problem: Problem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> Solution:
        function_problem = FunctionProblem(problem, maximize=False)
        problem_with_cutoff = EvalCutoffProblem(function_problem, max_n_evals)

        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=self.config["ea_generations"],
                problem=problem_with_cutoff,
                bounds=problem.bounds,
                pop_size=self.config["ea_pop_size"],
                mutation_std=self.config["ea_mutation_std"],
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=self.config["cma_generations"],
                problem=problem_with_cutoff,
                bounds=problem.bounds,
                sigma0=self.config["cma_sigma0"],
                lsc=MetaepochLimit(self.config["cma_metaepochs"]),
            ),
        ]
        global_stop_condition = SingularProblemEvalLimitReached(max_n_evals)
        sprout_condition = SproutMechanism(
            NBC_Generator(self.config["nbc_cut"], self.config["nbc_trunc"]),
            [NBC_FarEnough(self.config["nbc_far"], 2), DemeLimit(1)],
            [
                LevelLimit(self.config["level_limit"]),
            ],
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
                "ea_pop_size": (20, 300),
                "ea_generations": (1, 10),
                "ea_mutation_std": (0.25, 3.0),
                "cma_generations": (3, 30),
                "cma_metaepochs": (30, 300),
                "cma_sigma0": (0.1, 3.0),
            }
        )

    @classmethod
    def from_config(cls) -> "HMSSolver":
        with open(f"config/{cls.__name__}.json", "r") as json_file:
            config = json.load(json_file)
        return cls(config)

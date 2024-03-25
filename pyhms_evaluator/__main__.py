from .evaluator import ProblemEvaluator
from .problem import BBOBProblem
from .solver import CMAESSolver, HMSDESolver, HMSSolver, ILSHADESolver

solvers = [
    ILSHADESolver.from_config(),
    CMAESSolver.from_config(),
    HMSSolver.from_config(),
    HMSDESolver.from_config(),
]

N = 10
problems = [BBOBProblem(id, N) for id in range(1, 25)]
evaluator = ProblemEvaluator(solvers, problems, seed_count=10, max_n_evals=10000)  # type: ignore[arg-type]
results = evaluator()
results.to_csv("results_hms_with_warm_start.csv")

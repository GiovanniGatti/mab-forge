from functools import partial
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, OptimizeResult


def normalize(mu: npt.NDArray[float], std: npt.NDArray[float]) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Normalizes a bandit to the semi-unit circle.

    :param mu: The bandit's means
    :param std: The bandit's standard deviations
    """
    norm_factor = np.max(np.linalg.norm(np.vstack((mu, std)), axis=0))
    return mu / norm_factor, std / norm_factor


def min_regret(
        mu: npt.NDArray[float], std: npt.NDArray[float], T: int, z: float, tol: float = 1e-10, maxiter=200
) -> OptimizeResult:
    """
    Given a bandit, computes the distribution of actions such as to minimize the total exploration regret over T
     time steps.

    :param mu: The bandit's means (the optimal arm MUST be at index 0)
    :param std: The bandit's standard deviations
    :param T: Horizon time steps
    :param z: The confidence interval
    :param tol: Optimization error tolerance
    :param maxiter: Maximum number of iterations during search
    """

    x0 = np.repeat(T / mu.shape[0], mu.shape[0] - 1)

    def regret(ni: npt.NDArray[float]) -> np.float32:
        return np.sum(ni * (mu[0] - mu[1:]))

    def ci_constr(i: int, ni: npt.NDArray[float]) -> np.float32:
        n_star = T - np.sum(ni)
        ci_star = mu[0] - z * std[0] / np.sqrt(n_star)
        ci_i = mu[i] + z * std[i] / np.sqrt(ni[i - 1])
        return ci_star - ci_i

    cons = []
    for i in range(1, mu.shape[0]):
        cons.append({"type": "ineq", "fun": partial(ci_constr, i)})

    bounds = [(2, None) for _ in range(mu.shape[0] - 1)]

    return minimize(regret, x0, bounds=bounds, constraints=cons, method="slsqp", tol=tol, options={"maxiter": maxiter})

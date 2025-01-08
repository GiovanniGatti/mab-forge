from typing import Tuple

import numpy as np
from numpy import typing as npt
from numpy.random import Generator


def new_mab(
        n_arms: int, d_target: float, rng: Generator = np.random.default_rng()
) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Generates a new MAB problem with specified difficulty.

    :param n_arms: The bandit's number of arms (or actions); must be larger than one.
    :param d_target The bandit's target difficult; must be larger or equal to zero.
    :param rng: the numpy pseudorandom generator
    """
    d_arms = rng.uniform(0., 1., size=n_arms - 1)
    d_arms *= d_target / np.sum(d_arms)

    theta = rng.uniform(-np.pi / 2, np.pi / 2)
    r = np.sqrt(rng.uniform(0, 1.))
    sigma_star, mu_star = r * np.cos(theta), r * np.sin(theta)

    index_norm = rng.choice(n_arms)

    if index_norm == 0:
        norm = np.sqrt(sigma_star ** 2 + mu_star ** 2)
        sigma_star /= norm
        mu_star /= norm

    discriminant = np.sqrt(d_arms ** 2 - 4 * d_arms * mu_star + 4)
    lower = np.maximum((d_arms - discriminant) / 2, -1.0)
    higher = np.minimum((d_arms + discriminant) / 2, mu_star)

    mu_arms = rng.uniform(lower, higher)
    mu_arms[index_norm - 1] = lower[index_norm - 1] if index_norm > 0 else lower[0]

    sigma_arms = np.sqrt(d_arms * (mu_star - mu_arms))

    mu = np.append(mu_arms, mu_star)
    sigma = np.append(sigma_arms, sigma_star)

    return mu, sigma

import unittest

import numpy as np

from mab_forge import min_regret, normalize
from mab_forge import new_mab


class TestMinRegret(unittest.TestCase):

    def test_two_arms_and_low_variance(self) -> None:
        result = min_regret(mu=np.array([4., 0.]), std=np.array([1e-2, 1e-2]), T=3_000_000, z=2.58)
        assert result.x == 2
        assert result.fun == 2 * 4.

    def test_three_arms_and_low_variance(self) -> None:
        result = min_regret(mu=np.array([4., 2., 3.]), std=np.array([1e-2, 1e-2, 1e-2]), T=3_000_000, z=2.58)
        assert all(result.x == np.array([2, 2]))
        assert result.fun == 2 * (4. - 2.) + 2 * (4. - 3.)

    def test_two_arms_and_high_variance(self) -> None:
        result = min_regret(mu=np.array([4., 0.]), std=np.array([3., 5.]), T=3_000_000, z=2.58)
        assert np.isclose(result.x, 10.42, atol=1e-2)
        assert result.fun == result.x * 4.
        assert np.isclose(0. + 2.58 * 5. / np.sqrt(result.x), 4. - 2.58 * 3. / np.sqrt(3_000_000 - result.x), atol=1e-3)

    def test_two_arms_high_variance_and_low_horizon(self) -> None:
        result = min_regret(mu=np.array([4., 0.]), std=np.array([3., 5.]), T=100, z=2.58)
        assert np.isclose(result.x, 16.75303929, atol=1e-2)
        assert result.fun == result.x * 4.
        assert np.isclose(0. + 2.58 * 5. / np.sqrt(result.x), 4. - 2.58 * 3. / np.sqrt(100. - result.x), atol=1e-3)

    def test_optimization_fails_if_horizon_too_short(self) -> None:
        # There is no solution for such short horizon and the selected confidence level
        result = min_regret(mu=np.array([4., 0.]), std=np.array([3., 5.]), T=30, z=2.58)
        assert not result.success


class TestNormalization(unittest.TestCase):

    def test_two_arms(self) -> None:
        mu, std = normalize(mu=np.array([0.9, 1.]), std=np.array([0.2, 0.3]))
        assert all(np.isclose(np.sqrt(mu ** 2 + std ** 2), np.array([0.88, 1.]), atol=1e-2))

        _mu, _std = normalize(mu=np.array([9, 10.]), std=np.array([2, 3]))
        assert all(np.isclose(np.sqrt(_mu ** 2 + _std ** 2), np.array([0.88, 1.]), atol=1e-2))

        assert all(np.isclose(_mu, mu, atol=1e-3)) and all(np.isclose(std, _std, atol=1e-3))

    def test_three_arms(self) -> None:
        mu, std = normalize(mu=np.array([0.9, 0.1, 1.]), std=np.array([0.2, 0.9, 0.3]))
        assert all(np.isclose(np.sqrt(mu ** 2 + std ** 2), np.array([0.88, 0.86, 1.]), atol=1e-2))

        _mu, _std = normalize(mu=np.array([9, 1, 10.]), std=np.array([2, 9, 3]))
        assert all(np.isclose(np.sqrt(_mu ** 2 + _std ** 2), np.array([0.88, 0.86, 1.]), atol=1e-2))

        assert all(np.isclose(_mu, mu, atol=1e-3)) and all(np.isclose(std, _std, atol=1e-3))


class TestSampler(unittest.TestCase):

    def test_two_arms(self) -> None:
        rng = np.random.default_rng(0)
        mu, std = new_mab(2, 2.4, rng)

        idx = np.argsort(mu)[::-1]
        mu = mu[idx]
        std = std[idx]

        assert np.isclose(np.sum(std[1:] ** 2 / (mu[0] - mu[1:])), 2.4, atol=0.1)

        result = min_regret(mu=mu, std=std, T=3_000_000, z=2.58)
        assert np.isclose(result.fun / (2.58 ** 2), 2.4, atol=0.15)

    def test_three_arms(self) -> None:
        rng = np.random.default_rng(1)
        mu, std = new_mab(3, 2.4, rng)

        idx = np.argsort(mu)[::-1]
        mu = mu[idx]
        std = std[idx]

        assert np.isclose(np.sum(std[1:] ** 2 / (mu[0] - mu[1:])), 2.4, atol=0.1)

        result = min_regret(mu=mu, std=std, T=90_000_000, z=2.58)
        assert np.isclose(result.fun / (2.58 ** 2), 2.4, atol=0.15)


if __name__ == "__main__":
    unittest.main()

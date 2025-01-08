![Static Badge](https://img.shields.io/badge/python-3.9_%7C_3.10_%7C_3.11_%7C_3.12_%7C_3.13-blue)
![Static Badge](https://img.shields.io/badge/coverage-100%25-green)

# MAB Forge

Designing algorithms for Multi Armed Bandit problems involves working on their numerical assessment. However, the
conditions such algorithms are tested is fundamental for drawing correct conclusions.

This repository provides an algorithm for controlling the biases when generating bandits for evaluation.

# Installation

```bash
pip install mab-forge
```

# Library

To create a new MAB with difficulty level 2.4, simply do

```python
import mab_forge

mu, std = mab_forge.new_mab(n_arms=3, d_target=2.4)
```

To better grap on the intuitions of what the difficulty means, check out our paper and other provided materials.

Besides, we also provide a method for normalizing bandits to their semi-unit circle representation. Please, refer to
the paper and other materials for better understanding normalization.

```python
import numpy as np

import mab_forge

mu = np.array([1., 3., 5.])
std = np.array([0.3, 7., 10.])

norm_mu, norm_std = mab_forge.normalize(mu, std)
```

Lastly, you can compute the minimum exploration regret for any bandit problem given a confidence level and a maximum
number of steps with

```python
import numpy as np

import mab_forge

mu = np.array([1., 3., 5.])
std = np.array([0.3, 7., 10.])

# When using min_regret, you have to provide the optimal arm at the first index.
idx = np.argsort(mu)[::-1]
mu = mu[idx]
std = std[idx]

regret = mab_forge.min_regret(mu, std, T=2000, z=2.96)

print(regret)
# message: Optimization terminated successfully
# success: True
#  status: 0
#     fun: 523.6313564965296
#       x: [ 2.578e+02  2.000e+00]
#     nit: 15
#     jac: [ 2.000e+00  4.000e+00]
#    nfev: 50
#    njev: 14
```

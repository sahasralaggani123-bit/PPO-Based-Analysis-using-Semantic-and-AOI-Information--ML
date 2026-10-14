"""
baselines.py -- three simple non-learning schedulers.

Every scheduler exposes the same interface:
    reset(seed)            -> called at the start of each evaluation episode
    select_action(obs)     -> returns an int in [0, N-1]
and receives the SAME observation vector PPO sees:
    obs = [AoI_1/MAX, I_1, AoI_2/MAX, I_2, ...]
Ties are broken by the lowest device index (np.argmax).
"""
import numpy as np


class RandomScheduler:
    name = "Random"

    def __init__(self, num_devices: int):
        self.n = num_devices
        self.rng = np.random.default_rng(0)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs) -> int:
        return int(self.rng.integers(0, self.n))


class HighestAoIScheduler:
    name = "Highest-AoI"

    def __init__(self, num_devices: int):
        self.n = num_devices

    def reset(self, seed=None):
        pass

    def select_action(self, obs) -> int:
        return int(np.argmax(np.asarray(obs)[0::2]))   # AoI entries


class HighestSemanticScheduler:
    name = "Highest-Semantic"

    def __init__(self, num_devices: int):
        self.n = num_devices

    def reset(self, seed=None):
        pass

    def select_action(self, obs) -> int:
        return int(np.argmax(np.asarray(obs)[1::2]))   # importance entries


def get_baselines(num_devices: int):
    return [RandomScheduler(num_devices),
            HighestAoIScheduler(num_devices),
            HighestSemanticScheduler(num_devices)]

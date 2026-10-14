"""
environment.py -- Gymnasium environment for semantic- and AoI-aware scheduling.

Definitions used everywhere in the project
------------------------------------------
N devices, discrete time slots t = 1..T (T = episode_length).

Device i has
    AoI_i(t)  : age of the latest information of device i at the receiver
    I_i       : semantic importance in [0, 1] (synthetic, constant within an episode)

STATE (what PPO observes), dimension 2N, interleaved:
    [AoI_1/MAX_AOI, I_1, AoI_2/MAX_AOI, I_2, ..., AoI_N/MAX_AOI, I_N]
    (AoI is divided by MAX_AOI only to scale it to [0,1]; meaning is unchanged.)

ACTION: integer a in {0..N-1}; action k means "device k+1 transmits in this slot".
    Exactly one device is scheduled per time slot.

AoI DYNAMICS (after choosing action a):
    AoI_a     <- FRESH_AOI                       (selected device becomes fresh)
    AoI_i     <- min(AoI_i + AOI_INCREMENT, MAX_AOI)   for every i != a

REWARD (exact equation):
    r_t = W_SEMANTIC * I_a  -  W_AOI * ( mean_i AoI_i(t) / MAX_AOI )
  where AoI_i(t) are the values AFTER the update above.
    - W_SEMANTIC * I_a          : semantic benefit of the scheduled device, in [0, W_SEMANTIC]
    - W_AOI * mean AoI / MAX    : freshness cost, in [0, W_AOI]
  => r_t is bounded in [-W_AOI, +W_SEMANTIC].
  Scheduling a device with a large AoI lowers the mean AoI (more reward);
  scheduling an important device raises the semantic term (more reward).

EPISODE: fixed length; the episode is *truncated* (not terminated) after
    episode_length steps.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config


@dataclass
class EnvConfig:
    """All configurable simulation parameters (defaults come from config.py)."""
    num_devices: int = config.NUM_DEVICES
    episode_length: int = config.EPISODE_LENGTH
    fresh_aoi: float = config.FRESH_AOI
    aoi_increment: float = config.AOI_INCREMENT
    max_aoi: float = config.MAX_AOI
    initial_aoi_max: int = config.INITIAL_AOI_MAX
    importance_low: float = config.IMPORTANCE_LOW
    importance_high: float = config.IMPORTANCE_HIGH
    resample_importance_each_episode: bool = config.RESAMPLE_IMPORTANCE_EACH_EPISODE
    fixed_importance_seed: int = config.FIXED_IMPORTANCE_SEED
    w_semantic: float = config.W_SEMANTIC
    w_aoi: float = config.W_AOI


class IoTSchedulingEnv(gym.Env):
    """Single-channel scheduling of N virtual IoT devices (one transmission per slot)."""

    metadata = {"render_modes": []}

    def __init__(self, env_config: EnvConfig = None):
        super().__init__()
        self.cfg = env_config if env_config is not None else EnvConfig()
        n = self.cfg.num_devices

        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2 * n,), dtype=np.float32)

        self.aoi = np.zeros(n, dtype=np.float64)
        self.importance = np.zeros(n, dtype=np.float64)
        self.t = 0

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        obs = np.empty(2 * self.cfg.num_devices, dtype=np.float32)
        obs[0::2] = self.aoi / self.cfg.max_aoi   # AoI_i (scaled)
        obs[1::2] = self.importance               # I_i
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random
        n, c = self.cfg.num_devices, self.cfg
        if c.resample_importance_each_episode:
            rng = self.np_random
        else:
            rng = np.random.default_rng(c.fixed_importance_seed)
        self.importance = rng.uniform(c.importance_low, c.importance_high, size=n)
        self.aoi = self.np_random.integers(0, c.initial_aoi_max + 1, size=n).astype(np.float64)
        self.t = 0
        return self._get_obs(), {}

    def step(self, action):
        c = self.cfg
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action {action}; must be in [0, {c.num_devices - 1}]")
        action = int(action)

        # AoI update: everyone ages, selected device becomes fresh
        new_aoi = np.minimum(self.aoi + c.aoi_increment, c.max_aoi)
        new_aoi[action] = c.fresh_aoi
        self.aoi = new_aoi
        self.t += 1

        # Reward
        selected_importance = float(self.importance[action])
        mean_aoi = float(self.aoi.mean())
        semantic_benefit = c.w_semantic * selected_importance
        aoi_cost = c.w_aoi * (mean_aoi / c.max_aoi)
        reward = semantic_benefit - aoi_cost

        terminated = False
        truncated = self.t >= c.episode_length
        info = {
            "selected_device": action,
            "selected_importance": selected_importance,
            "mean_aoi": mean_aoi,
            "peak_aoi": float(self.aoi.max()),
            "semantic_benefit": semantic_benefit,
            "aoi_cost": aoi_cost,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

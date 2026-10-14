"""
sanity_checks.py -- quick integration checks. Run: python sanity_checks.py
Trains a tiny PPO model into a temp folder (does NOT overwrite models/ppo_model.zip).
"""
import tempfile
import numpy as np
from stable_baselines3 import PPO

import config
from environment import IoTSchedulingEnv, EnvConfig
from baselines import get_baselines


def check(cond, msg):
    print(("[PASS] " if cond else "[FAIL] ") + msg)
    assert cond, msg


env = IoTSchedulingEnv(EnvConfig())
obs, info = env.reset(seed=1)
N = env.cfg.num_devices
check(obs.shape == (2 * N,), f"reset works; state dim = {obs.shape[0]} (2N = {2 * N})")
check(env.observation_space.contains(obs), "observation lies in observation_space")
check(env.action_space.n == N, f"action space is Discrete({N})")

aoi_before = env.aoi.copy()
a = 3
obs2, r, term, trunc, info = env.step(a)
check(env.aoi[a] == env.cfg.fresh_aoi, "selected device becomes fresh")
others = [i for i in range(N) if i != a]
check(np.all(env.aoi[others] == np.minimum(aoi_before[others] + env.cfg.aoi_increment, env.cfg.max_aoi)),
      "non-selected devices age by AOI_INCREMENT (capped)")
check(not np.array_equal(aoi_before, env.aoi), "AoI actually changes")
check(np.isfinite(r), f"reward finite ({r:.4f})")

# Episode ends (truncation) at episode_length
env.reset(seed=2)
n_steps, done = 0, False
while not done:
    _, r, term, trunc, _ = env.step(env.action_space.sample())
    n_steps += 1
    done = term or trunc
check(n_steps == env.cfg.episode_length, f"episode length = {n_steps}")

# Reward sanity examples (importance / AoI chosen by hand)
e = IoTSchedulingEnv(EnvConfig())
e.reset(seed=0)
e.importance = np.array([0.9, 0.1] + [0.5] * (N - 2))
e.aoi = np.array([1.0, 1.0] + [10.0] * (N - 2))
snap = (e.aoi.copy(), e.importance.copy())
_, r_high_imp, *_ = e.step(0)
e.aoi, e.importance = snap[0].copy(), snap[1].copy(); e.t = 0
_, r_low_imp, *_ = e.step(1)
check(r_high_imp > r_low_imp, f"same AoI, higher importance -> higher reward ({r_high_imp:.4f} > {r_low_imp:.4f})")
e.aoi, e.importance = np.array([40.0] + [1.0] * (N - 1)), np.full(N, 0.5); e.t = 0
_, r_stale, *_ = e.step(0)
e.aoi, e.importance = np.array([40.0] + [1.0] * (N - 1)), np.full(N, 0.5); e.t = 0
_, r_fresh, *_ = e.step(1)
check(r_stale > r_fresh, f"same importance, serving the stalest device -> higher reward ({r_stale:.4f} > {r_fresh:.4f})")
lo, hi = -env.cfg.w_aoi, env.cfg.w_semantic
check(lo <= r_stale <= hi and lo <= r_fresh <= hi, f"reward within bounds [{lo}, {hi}]")

# Baselines
for s in get_baselines(N):
    o, _ = env.reset(seed=5)
    s.reset(5)
    act = s.select_action(o)
    check(0 <= act < N, f"baseline {s.name} returns valid action ({act})")

# PPO init / train / save / load
model = PPO("MlpPolicy", IoTSchedulingEnv(EnvConfig()), n_steps=256, batch_size=64,
            seed=0, device="cpu", verbose=0)
model.learn(total_timesteps=512)
with tempfile.TemporaryDirectory() as d:
    model.save(f"{d}/tmp_model")
    loaded = PPO.load(f"{d}/tmp_model", device="cpu")
    act, _ = loaded.predict(obs, deterministic=True)
    check(0 <= int(act) < N, "PPO initialised, trained, saved, loaded, predicts valid action")

print("\nAll sanity checks passed.")

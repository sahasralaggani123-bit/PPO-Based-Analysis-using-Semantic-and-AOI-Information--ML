"""
evaluate.py -- evaluate Random, Highest-AoI, Highest-Semantic and PPO.

Fairness procedure: evaluation episode k is reset with seed EVAL_SEED_BASE + k for
EVERY scheduler, so all methods start from identical initial AoI and importance
values. The dynamics are deterministic given the actions, so the only source of
difference is the scheduling policy (Random uses its own RNG seeded with the same
episode seed). PPO acts deterministically (argmax action).

Metric definitions (per episode, then mean/std over episodes):
  Average AoI        : mean over all T slots and all N devices of AoI after each slot's update
  Peak AoI           : maximum AoI of any device in any slot of the episode
  Semantic Utility   : mean semantic importance of the device scheduled in each slot
  Cumulative reward  : sum of the per-slot rewards over the episode
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import config
from environment import IoTSchedulingEnv, EnvConfig
from baselines import get_baselines


class PPOScheduler:
    name = "PPO"

    def __init__(self, model):
        self.model = model

    def reset(self, seed=None):
        pass

    def select_action(self, obs) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def run_episode(env, scheduler, seed):
    obs, _ = env.reset(seed=seed)
    scheduler.reset(seed)
    mean_aois, peaks, sel_imps, rewards = [], [], [], []
    done = False
    while not done:
        action = scheduler.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        mean_aois.append(info["mean_aoi"])
        peaks.append(info["peak_aoi"])
        sel_imps.append(info["selected_importance"])
        rewards.append(reward)
        done = terminated or truncated
    return {
        "average_aoi": float(np.mean(mean_aois)),
        "peak_aoi": float(np.max(peaks)),
        "semantic_utility": float(np.mean(sel_imps)),
        "cumulative_reward": float(np.sum(rewards)),
    }


def bar_plot(summary, metric, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = summary["method"].tolist()
    means = summary[f"{metric}_mean"].values
    stds = summary[f"{metric}_std"].values
    colors = ["#9e9e9e", "#4c72b0", "#dd8452", "#55a868"][:len(methods)]
    ax.bar(methods, means, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_reward(path, window=20):
    if not config.TRAIN_MONITOR_PATH.exists():
        print("No training monitor file found; skipping training reward plot.")
        return
    df = pd.read_csv(config.TRAIN_MONITOR_PATH, skiprows=1)
    steps = df["l"].cumsum()
    smooth = df["r"].rolling(window, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, df["r"], alpha=0.25, label="Episode reward")
    ax.plot(steps, smooth, label=f"Moving average ({window} episodes)")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode cumulative reward")
    ax.set_title("PPO training reward curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    config.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    model_file = str(config.MODEL_PATH) + ".zip"
    model = PPO.load(model_file, device=config.PPO_DEVICE)

    env = IoTSchedulingEnv(EnvConfig())
    schedulers = get_baselines(env.cfg.num_devices) + [PPOScheduler(model)]

    rows = []
    for sched in schedulers:
        for ep in range(config.EVAL_EPISODES):
            seed = config.EVAL_SEED_BASE + ep
            m = run_episode(env, sched, seed)
            rows.append({"method": sched.name, "episode": ep, "seed": seed, **m})
        print(f"Finished evaluating: {sched.name}")

    per_ep = pd.DataFrame(rows)
    per_ep.to_csv(config.PER_EPISODE_CSV, index=False)

    metrics = ["average_aoi", "peak_aoi", "semantic_utility", "cumulative_reward"]
    order = [s.name for s in schedulers]
    agg = per_ep.groupby("method")[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    summary = agg.loc[order].reset_index()
    summary.insert(1, "episodes", config.EVAL_EPISODES)
    summary.to_csv(config.RESULTS_CSV, index=False)

    g = config.GRAPHS_DIR
    bar_plot(summary, "average_aoi", "Average AoI (time slots)", "Average AoI (lower is better)", g / "average_aoi.png")
    bar_plot(summary, "peak_aoi", "Peak AoI (time slots)", "Peak AoI (lower is better)", g / "peak_aoi.png")
    bar_plot(summary, "semantic_utility", "Mean importance of scheduled device", "Semantic utility (higher is better)", g / "semantic_utility.png")
    bar_plot(summary, "cumulative_reward", "Cumulative episode reward", "Cumulative reward (higher is better)", g / "cumulative_reward.png")
    plot_training_reward(g / "training_reward.png")

    pd.set_option("display.width", 200)
    print("\nResults (mean over episodes):")
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved: {config.RESULTS_CSV}\n       {config.PER_EPISODE_CSV}\n       graphs in {g}")


if __name__ == "__main__":
    main()

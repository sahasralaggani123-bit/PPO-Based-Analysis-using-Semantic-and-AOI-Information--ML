"""
train_ppo.py -- train the PPO scheduler (Stable-Baselines3) and save the model.

Usage:  python train_ppo.py [--timesteps N]
Outputs: models/ppo_model.zip, results/training.monitor.csv (per-episode training reward)
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import config
from environment import IoTSchedulingEnv, EnvConfig


def main(timesteps: int):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = Monitor(IoTSchedulingEnv(EnvConfig()), filename=str(config.TRAIN_MONITOR_PATH))

    model = PPO(
        "MlpPolicy", env,
        learning_rate=config.PPO_LEARNING_RATE,
        n_steps=config.PPO_N_STEPS,
        batch_size=config.PPO_BATCH_SIZE,
        n_epochs=config.PPO_N_EPOCHS,
        gamma=config.PPO_GAMMA,
        gae_lambda=config.PPO_GAE_LAMBDA,
        clip_range=config.PPO_CLIP_RANGE,
        ent_coef=config.PPO_ENT_COEF,
        policy_kwargs=dict(net_arch=config.PPO_NET_ARCH),
        seed=config.SEED,
        device=config.PPO_DEVICE,
        verbose=1,
    )
    print(f"Training PPO for {timesteps} timesteps (seed={config.SEED}) ...")
    model.learn(total_timesteps=timesteps)
    model.save(str(config.MODEL_PATH))
    print(f"Model saved to {config.MODEL_PATH}.zip")
    print(f"Training reward log saved to {config.TRAIN_MONITOR_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=config.TRAIN_TIMESTEPS)
    main(parser.parse_args().timesteps)

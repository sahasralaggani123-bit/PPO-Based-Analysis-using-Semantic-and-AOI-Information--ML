# Methodology

## 1. Scope
Simulation-based study of scheduling in a massive-IoT-style network: N = 10 virtual devices,
one transmission per time slot. No hardware, no real dataset, no NLP/embeddings.

## 2. Model
- Device i: AoI_i (maintained by the simulator) and semantic importance I_i in [0,1] (synthetic).
- Slot dynamics for action a: AoI_a <- 0; AoI_i <- min(AoI_i + 1, 50) for i != a.
- Horizon: 200 slots (truncation). Initial AoI ~ Uniform{0..10}; importance ~ Uniform(0,1), drawn per episode from the episode seed.

## 3. MDP definition
- State (20-dim): [AoI_1/50, I_1, ..., AoI_10/50, I_10].
- Action: Discrete(10), the device given the slot.
- Reward: r_t = 0.5 * I_a - 0.5 * (mean_i AoI_i / 50), bounded in [-0.5, 0.5].
  Weights are in config.py. The formulation is a simple weighted trade-off, not claimed novel.

## 4. Algorithm
Stable-Baselines3 PPO, MlpPolicy [64,64], lr 3e-4, n_steps 2048, batch 64, 10 epochs,
gamma 0.99, GAE lambda 0.95, clip 0.2, entropy coef 0.01, 150,000 timesteps, seed 42, CPU.
Hyperparameters were NOT tuned.

## 5. Evaluation protocol
50 episodes; episode k uses seed 10000+k for all methods (identical initial conditions).
PPO acts deterministically. Metrics per episode: average AoI, peak AoI, semantic utility
(mean importance of scheduled devices), cumulative reward; reported as mean and std over episodes.

## 6. Workflow
config.py -> environment.py -> baselines.py -> train_ppo.py (model + training log) ->
evaluate.py (results.csv, graphs).

## 7. Results from the run performed while building this project
(Produced by actually running `train_ppo.py` and `evaluate.py`; mean +/- std over 50 episodes.)

| Method | Avg AoI | Peak AoI | Semantic utility | Cumulative reward |
|---|---|---|---|---|
| Random | 8.74 +/- 0.61 | 47.52 +/- 4.07 | 0.477 +/- 0.079 | 30.2 +/- 8.0 |
| Highest-AoI | 4.51 +/- 0.02 | 10.66 +/- 1.21 | 0.478 +/- 0.074 | 38.8 +/- 7.4 |
| Highest-Semantic | 40.49 +/- 0.22 | 50.00 +/- 0.00 | 0.906 +/- 0.093 | 9.6 +/- 9.4 |
| PPO | 9.16 +/- 1.95 | 35.80 +/- 6.78 | 0.710 +/- 0.094 | 52.7 +/- 7.8 |

Interpretation (honest):
- PPO obtained the highest cumulative reward, the metric it was trained on, by balancing
  the two objectives: semantic utility (0.71) is far above Highest-AoI (0.48), while average AoI
  (9.2) is far better than Highest-Semantic (40.5).
- PPO is NOT best on freshness: Highest-AoI has clearly lower average AoI (4.5) and peak AoI (10.7),
  and PPO's average AoI is slightly worse than Random's (9.2 vs 8.7) with a lower peak AoI (35.8 vs 47.5).
- Highest-Semantic is best on semantic utility but starves most devices (peak AoI at the 50 cap).
- Caveats: one training seed, untuned hyperparameters, synthetic setting. The ranking by reward
  depends on the chosen weights (0.5/0.5); other weights may change the ordering.
Re-running on another machine may give slightly different numbers.

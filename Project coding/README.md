# PPO-Based Semantic and Age-of-Information-Aware Scheduling for Massive IoT Networks

## 1. Problem statement
In massive IoT networks many devices compete for limited communication resources. At each
time slot the scheduler must decide which device gets the transmission opportunity. This
project studies the trade-off between **information freshness (Age of Information)** and
**semantic importance** in a *simulated* network.

## 2. Objective
Train a PPO scheduler that uses the network resource efficiently, keeps information fresh
and gives appropriate priority to semantically important devices, and compare it
honestly against Random, Highest-AoI and Highest-Semantic-Importance scheduling.
(We use existing PPO and existing AoI concepts; the contribution is the experimental study
of the joint freshness/importance trade-off, not a new algorithm.)

## 3. Age of Information (AoI)
AoI is the age of the latest information of a device available at the receiver, counted in
time slots. Lower = fresher. In this simulation the selected device's AoI resets to
`FRESH_AOI = 0`; every other device's AoI grows by `AOI_INCREMENT = 1`, capped at `MAX_AOI = 50`.

## 4. Semantic importance
A synthetic number in [0, 1] per device (Uniform(0,1), seeded) describing how useful that
device's information is. It is only a simulation variable: **no real semantic dataset, NLP or
embedding model is used.** By default a new importance vector is drawn every episode (from the
episode seed) so that the state's importance entries carry information; set
`RESAMPLE_IMPORTANCE_EACH_EPISODE = False` in `config.py` for one fixed vector.

## 5. PPO methodology
Stable-Baselines3 `PPO` with `MlpPolicy` (two 64-unit hidden layers). The agent observes the
state, picks a device, receives the reward and next state, and improves its policy over many
episodes. Only the scheduler is trained; the virtual devices are not.

## 6. State (dimension 2N = 20)
`[AoI_1/MAX_AOI, I_1, AoI_2/MAX_AOI, I_2, ..., AoI_N/MAX_AOI, I_N]`
(AoI is only scaled to [0,1]; meaning unchanged.)

## 7. Action
Discrete(N): action k schedules device k+1. Exactly one device per slot.

## 8. Reward (exact equation)
```
r_t = W_SEMANTIC * I_a  -  W_AOI * ( mean_i AoI_i(t) / MAX_AOI )
```
`a` = selected device, `AoI_i(t)` = AoI after the slot's update, `W_SEMANTIC = W_AOI = 0.5`.
- First term: semantic benefit of the served device, in [0, W_SEMANTIC].
- Second term: freshness cost (average network staleness), in [0, W_AOI].
- Hence `r_t` is bounded in [-W_AOI, +W_SEMANTIC] = [-0.5, 0.5].
Serving an important device raises the first term; serving a stale device lowers the mean
AoI and therefore the second term. Neither term alone is optimal over a full episode.
Weights are configurable. Reward behaviour is checked in `sanity_checks.py`.

## 9. Environment
`environment.py` -> `IoTSchedulingEnv` (Gymnasium API: `reset(seed=...)` returns
`(obs, info)`; `step(a)` returns `(obs, reward, terminated, truncated, info)`).
Fixed horizon `EPISODE_LENGTH = 200` slots (ends by truncation). Initial AoI of each device
is Uniform{0..10}. Every parameter is in `config.py`.

## 10. Baselines (`baselines.py`)
1. **Random** - uniformly random device.
2. **Highest-AoI** - device with the largest AoI (ties -> lowest index).
3. **Highest-Semantic** - device with the largest importance (ties -> lowest index).
4. **PPO** - trained policy (deterministic action at evaluation).

## 11. Metrics (per episode; then mean and std over `EVAL_EPISODES = 50` episodes)
- **Average AoI**: mean of AoI over all slots and all devices (after each slot's update).
- **Peak AoI**: maximum AoI of any device in any slot of the episode.
- **Semantic utility**: mean importance of the device scheduled in each slot.
- **Cumulative reward**: sum of `r_t` over the episode.
Only quantities that are actually modelled are reported (no throughput, energy, etc.).

**Fair evaluation:** episode k uses seed `EVAL_SEED_BASE + k` for all four methods, so all
start from identical initial AoI and importance values; dynamics are deterministic given the
actions.

## 12. Project structure
```
config.py            parameters          environment.py   Gymnasium env + reward
baselines.py         3 baselines         train_ppo.py     train + save PPO
evaluate.py          run all, CSV, plots sanity_checks.py  integration checks
requirements.txt     README.md           documentation/methodology.md
models/ppo_model.zip
results/results.csv, results/results_per_episode.csv, results/training.monitor.csv
results/graphs/{average_aoi,peak_aoi,semantic_utility,cumulative_reward,training_reward}.png
```

## 13. Installation
```
pip install -r requirements.txt
```

## 14. Run
```
python sanity_checks.py     # optional integration checks (~seconds)
python train_ppo.py         # trains PPO (150k timesteps, a few minutes on a laptop CPU)
python evaluate.py          # evaluates all methods, writes CSV + graphs
```
Options: `python train_ppo.py --timesteps 50000`.

## 15. Where results are saved
`results/results.csv` (summary), `results/results_per_episode.csv` (raw), `results/graphs/*.png`.

## 16. Results
The numbers in `results/results.csv` are produced by running the code above. See
`documentation/methodology.md` (Section 7) for the run we performed and its honest
interpretation. Results can differ slightly across machines/library versions.

## 17. Important assumptions
- Single channel: exactly one transmission per slot; transmissions always succeed.
- Synthetic importance, constant within an episode; simplified deterministic AoI dynamics.
- AoI capped at `MAX_AOI`; metrics use the capped values.
- Single PPO training seed and default hyperparameters (no tuning).

## 18. Limitations
Small N (10), synthetic importance, no channel errors/energy/delay, single training seed,
no hyperparameter search. Conclusions apply to this simulation only.

## 19. Future work
Larger N, multiple channels, unreliable transmissions, time-varying importance, multiple
training seeds with confidence intervals, other reward weightings.

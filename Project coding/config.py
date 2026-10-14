"""
config.py -- central configuration for the whole project.

Every important simulation / training / evaluation parameter lives here so that
environment.py, baselines.py, train_ppo.py and evaluate.py all use ONE consistent
definition of the problem.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"

MODEL_PATH = MODELS_DIR / "ppo_model"                     # SB3 appends ".zip"
TRAIN_MONITOR_PATH = RESULTS_DIR / "training.monitor.csv"  # written by SB3 Monitor
RESULTS_CSV = RESULTS_DIR / "results.csv"                  # summary (mean/std per method)
PER_EPISODE_CSV = RESULTS_DIR / "results_per_episode.csv"  # raw per-episode metrics

# ----------------------------------------------------------------------------
# Environment / simulation parameters
# ----------------------------------------------------------------------------
NUM_DEVICES = 10          # N virtual IoT devices  -> state dim = 2N, action space = N
EPISODE_LENGTH = 200      # number of time slots per episode (fixed horizon)

FRESH_AOI = 0.0           # AoI of the selected device right after its transmission
AOI_INCREMENT = 1.0       # AoI growth of every non-selected device per time slot
MAX_AOI = 50.0            # AoI is capped at this value (keeps state/reward bounded)
INITIAL_AOI_MAX = 10      # initial AoI of each device ~ Uniform{0, ..., INITIAL_AOI_MAX}

IMPORTANCE_LOW = 0.0      # semantic importance ~ Uniform(LOW, HIGH), synthetic
IMPORTANCE_HIGH = 1.0
# True : new random importance vector every episode (drawn from the reset seed)
# False: one fixed importance vector (drawn from FIXED_IMPORTANCE_SEED) for all episodes
RESAMPLE_IMPORTANCE_EACH_EPISODE = True
FIXED_IMPORTANCE_SEED = 0

# ----------------------------------------------------------------------------
# Reward weights
#   reward = W_SEMANTIC * importance_of_selected_device
#            - W_AOI    * (mean AoI after the update) / MAX_AOI
# ----------------------------------------------------------------------------
W_SEMANTIC = 0.5
W_AOI = 0.5

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42                 # global seed for PPO training / environment
EVAL_SEED_BASE = 10_000   # evaluation episode k uses seed EVAL_SEED_BASE + k

# ----------------------------------------------------------------------------
# PPO (Stable-Baselines3) parameters
# ----------------------------------------------------------------------------
TRAIN_TIMESTEPS = 150_000
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.01
PPO_NET_ARCH = [64, 64]   # small MLP for both policy and value networks
PPO_DEVICE = "cpu"

# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
EVAL_EPISODES = 50

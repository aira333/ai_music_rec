"""
q_agent.py
Double Dueling DQN agent for adaptive song selection.

Key fix vs v1: reward function is now discriminative.
  - User profile is a specific genre + energy preference
  - Reward is based on how well the track matches THAT specific taste
  - Candidate pool always includes a mix of matching and non-matching tracks
  - Random agent gets ~0 on average; good agent gets consistently positive
"""

import sys
import logging
import argparse
import ast
from pathlib import Path
from collections import deque, namedtuple
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "audio_features.csv"
CKPT_DIR  = PROJECT_ROOT / "checkpoints"

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

FEATURE_COLS = [
    "danceability", "energy", "loudness_norm", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo_norm",
]
FEATURE_DIM = len(FEATURE_COLS)

class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden // 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden // 2, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )

    def forward(self, x):
        shared = self.shared(x)
        V = self.value_stream(shared)
        A = self.advantage_stream(shared)
        return V + A - A.mean(dim=1, keepdim=True)

class MusicEnvironment:
    """
    Simulates a user with a specific taste profile.

    The key design principle: candidate pool always contains
      - N/3 tracks that match the user's taste  (should be rewarded)
      - N/3 tracks that are neutral
      - N/3 tracks that mismatch the user's taste (should be penalised)

    This guarantees the agent CAN learn — there's always a clearly
    better choice available. Random selection scores ~0; a good agent
    consistently picks the matching tracks and scores ~+1.
    """

    EPISODE_LEN = 50   # shorter episodes = faster learning signal

    def __init__(self, df: pd.DataFrame, candidate_pool_size: int = 30):
        self.df                  = df.reset_index(drop=True)
        self.candidate_pool_size = candidate_pool_size
        self.n_per_group         = candidate_pool_size // 3

        # Pre-group tracks by energy quartile for fast sampling
        self.df["energy_q"] = pd.qcut(self.df["energy"], q=4, labels=False)
        self.groups = {q: self.df[self.df["energy_q"] == q] for q in range(4)}

        self.history       = []
        self.candidates    = []
        self._user_energy_q = None
        self.steps          = 0

    def reset(self) -> np.ndarray:
        # User prefers a specific energy quartile
        self._user_energy_q = np.random.randint(0, 4)
        self.history        = []
        self.steps          = 0
        self._refresh_candidates()
        return self._get_state()

    def step(self, action: int):
        track  = self.candidates[action]
        reward = self._compute_reward(track)

        vec = self._get_vec(track)
        self.history.append(vec)
        if len(self.history) > 10:
            self.history.pop(0)

        self.steps += 1
        self._refresh_candidates()
        done = self.steps >= self.EPISODE_LEN
        return self._get_state(), reward, done, {}

    def _compute_reward(self, track: pd.Series) -> float:
        """
        Reward based on three components, each clearly discriminative:
          1. Genre match: does the track's energy quartile match user preference?
          2. Popularity: high popularity tracks are intrinsically better
          3. Transition: smooth tempo/energy transition from last track
        Reward is in [-1, +1]. Random agent expected value ≈ 0.
        """
        # 1. Energy preference match (main signal)
        track_eq     = int(track.get("energy_q", 0))
        energy_match = 1.0 if track_eq == self._user_energy_q else -1.0
        if abs(track_eq - self._user_energy_q) == 1:
            energy_match = 0.0   # adjacent quartile = neutral

        # 2. Popularity bonus (normalised to [-0.5, 0.5])
        pop   = float(track.get("popularity", 50))
        pop_r = (pop / 100.0) - 0.5

        # 3. Transition smoothness
        if self.history:
            last        = self.history[-1]
            current_vec = self._get_vec(track)
            # L2 distance in [0, sqrt(FEATURE_DIM)]; closer = smoother
            dist        = float(np.linalg.norm(current_vec - last))
            smooth      = 1.0 - np.clip(dist / 2.0, 0, 1)   # [0, 1]
            transition  = smooth * 2 - 1                       # [-1, 1]
        else:
            transition = 0.0

        reward = 0.6 * energy_match + 0.2 * pop_r + 0.2 * transition
        return float(np.clip(reward, -1.0, 1.0))

    def _refresh_candidates(self):
        """
        Build a candidate pool with equal parts:
          matching tracks, neutral tracks, mismatching tracks.
        """
        mismatch_q = (self._user_energy_q + 2) % 4   # opposite quartile
        neutral_qs = [q for q in range(4) if q not in (self._user_energy_q, mismatch_q)]

        match_pool    = self.groups[self._user_energy_q]
        mismatch_pool = self.groups[mismatch_q]
        neutral_pool  = self.groups[neutral_qs[np.random.randint(len(neutral_qs))]]

        n = self.n_per_group
        candidates = pd.concat([
            match_pool.sample(min(n, len(match_pool)), replace=False),
            neutral_pool.sample(min(n, len(neutral_pool)), replace=False),
            mismatch_pool.sample(min(n, len(mismatch_pool)), replace=False),
        ]).reset_index(drop=True)

        self.candidates = [candidates.iloc[i] for i in range(len(candidates))]

    def _get_state(self) -> np.ndarray:
        if not self.history:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        return np.mean(self.history, axis=0).astype(np.float32)

    def _get_vec(self, track: pd.Series) -> np.ndarray:
        vec = []
        for col in FEATURE_COLS:
            vec.append(float(track.get(col, 0.0)))
        return np.array(vec, dtype=np.float32)

    @property
    def n_actions(self):
        return len(self.candidates) if self.candidates else self.candidate_pool_size

class QLearningAgent:
    def __init__(
        self,
        state_dim:      int,
        n_actions:      int,
        lr:             float = 5e-4,
        gamma:          float = 0.95,
        epsilon_start:  float = 1.0,
        epsilon_end:    float = 0.05,
        epsilon_decay:  int   = 500,
        batch_size:     int   = 128,
        target_update:  int   = 10,
        buffer_capacity: int  = 20_000,
        device: Optional[torch.device] = None,
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps_done    = 0
        self.updates_done  = 0

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        self.policy_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimiser = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory    = ReplayBuffer(buffer_capacity)

        logger.info(f"QLearningAgent | state_dim={state_dim} n_actions={n_actions} "
                    f"params={sum(p.numel() for p in self.policy_net.parameters()):,}")

    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps_done / self.epsilon_decay)

    def select_action(self, state: np.ndarray) -> int:
        self.steps_done += 1
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        self.policy_net.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(self.policy_net(s).argmax(dim=1).item())

    def optimise(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        batch       = self.memory.sample(self.batch_size)
        states      = torch.tensor(np.array([t.state      for t in batch]), dtype=torch.float32).to(self.device)
        actions     = torch.tensor([t.action    for t in batch], dtype=torch.long).to(self.device)
        rewards     = torch.tensor([t.reward    for t in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32).to(self.device)
        dones       = torch.tensor([t.done      for t in batch], dtype=torch.float32).to(self.device)

        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_next = self.policy_net(next_states).argmax(dim=1)
            next_q    = self.target_net(next_states).gather(1, best_next.unsqueeze(1)).squeeze(1)
            target_q  = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimiser.step()

        self.updates_done += 1
        if self.updates_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    @staticmethod
    def compute_reward(feedback: str) -> float:
        return {"like": 1.0, "replay": 1.0, "listen": 0.2,
                "skip": -1.0, "dislike": -1.0}.get(feedback, 0.0)

    def save(self, path: Path) -> None:
        torch.save({
            "policy":  self.policy_net.state_dict(),
            "target":  self.target_net.state_dict(),
            "steps":   self.steps_done,
            "updates": self.updates_done,
        }, path)
        logger.info(f"Checkpoint saved → {path}")

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.steps_done   = ckpt.get("steps", 0)
        self.updates_done = ckpt.get("updates", 0)
        logger.info(f"Checkpoint loaded ← {path}")

def run_episodes(agent, env, n_episodes, learn=True) -> dict:
    history = {"episode_reward": [], "loss": [], "epsilon": []}

    for ep in range(1, n_episodes + 1):
        state     = env.reset()
        ep_reward = 0.0
        losses    = []

        while True:
            action = agent.select_action(state) if learn else np.random.randint(agent.n_actions)
            next_state, reward, done, _ = env.step(action)

            if learn:
                agent.memory.push(state, action, reward, next_state, done)
                loss = agent.optimise()
                if loss is not None:
                    losses.append(loss)

            ep_reward += reward
            state = next_state
            if done:
                break

        history["episode_reward"].append(ep_reward)
        history["loss"].append(np.mean(losses) if losses else 0.0)
        history["epsilon"].append(agent.epsilon)

        if ep % 20 == 0 or ep == 1:
            avg = np.mean(history["episode_reward"][-20:])
            logger.info(
                f"Episode {ep:4d}/{n_episodes} | avg_reward={avg:+.3f} | "
                f"ε={agent.epsilon:.3f} | buffer={len(agent.memory):,} | "
                f"loss={history['loss'][-1]:.4f}"
            )

    return history


def plot_results(rl_hist, rand_hist, save_path):
    def smooth(x, w=20):
        return pd.Series(x).rolling(w, min_periods=1).mean().values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    eps = range(1, len(rl_hist["episode_reward"]) + 1)

    axes[0].plot(eps, smooth(rl_hist["episode_reward"]),   color="#1DB954", label="DQN Agent")
    axes[0].plot(eps, smooth(rand_hist["episode_reward"]), color="#E91429", linestyle="--", label="Random")
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle=":")
    axes[0].set_title("Episode Reward (smoothed)"); axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(eps, smooth(rl_hist["loss"]), color="#1DB954")
    axes[1].set_title("Training Loss (Huber)"); axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss"); axes[1].grid(alpha=0.3)

    axes[2].plot(eps, rl_hist["epsilon"], color="#1DB954")
    axes[2].set_title("Exploration Rate (ε)"); axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("ε"); axes[2].grid(alpha=0.3)

    plt.suptitle("Double Dueling DQN — Music Recommendation Agent", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Saved → {save_path}")
    plt.show()


def main(n_episodes=200, candidate_pool=30):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.\nRun: python data/load_spotify_dataset.py --source huggingface")

    logger.info("Loading dataset ...")
    df = pd.read_csv(DATA_PATH)

    # Ensure normalised columns exist
    for col in ["loudness", "tempo"]:
        norm_col = col + "_norm"
        if norm_col not in df.columns:
            lo, hi = df[col].min(), df[col].max()
            df[norm_col] = ((df[col] - lo) / (hi - lo + 1e-8)).clip(0, 1)

    logger.info(f"Loaded {len(df):,} tracks.")

    env   = MusicEnvironment(df, candidate_pool_size=candidate_pool)
    agent = QLearningAgent(state_dim=FEATURE_DIM, n_actions=candidate_pool)

    logger.info(f"Training DQN agent for {n_episodes} episodes ...")
    rl_hist = run_episodes(agent, env, n_episodes, learn=True)

    logger.info("Running random baseline ...")
    dummy = QLearningAgent(state_dim=FEATURE_DIM, n_actions=candidate_pool)
    rand_hist = run_episodes(dummy, env, n_episodes, learn=False)

    rl_avg   = np.mean(rl_hist["episode_reward"][-50:])
    rand_avg = np.mean(rand_hist["episode_reward"])

    ep_len = MusicEnvironment.EPISODE_LEN
    rl_per_step   = rl_avg   / ep_len
    rand_per_step = rand_avg / ep_len
    improvement = ((rl_avg - rand_avg) / (abs(rand_avg) + ep_len * 0.1)) * 100

    print(f"\n{'='*55}")
    print(f"  DQN avg reward (last 50 eps):  {rl_avg:+.3f}")
    print(f"  Random baseline avg reward:    {rand_avg:+.3f}")
    print(f"  DQN reward/step:               {rl_per_step:+.4f}")
    print(f"  Random reward/step:            {rand_per_step:+.4f}")
    print(f"  Improvement over random:       {improvement:+.1f}%")
    print(f"  Target: ≥30% improvement")
    print(f"  {'TARGET MET' if improvement >= 30 else ' Try --episodes 400'}")
    print(f"{'='*55}\n")

    CKPT_DIR.mkdir(exist_ok=True)
    agent.save(CKPT_DIR / "q_agent.pt")
    plot_results(rl_hist, rand_hist, CKPT_DIR / "rl_training.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int, default=200)
    parser.add_argument("--candidate_pool", type=int, default=30)
    args = parser.parse_args()
    main(args.episodes, args.candidate_pool)
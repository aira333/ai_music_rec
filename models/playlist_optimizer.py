"""
playlist_optimizer.py 
Hill climbing playlist optimizer with four improvements over v1:

  1. Or-opt moves  — relocate a single track to its best position in the
                     playlist (complements 2-opt swaps, escapes different
                     local optima)

  2. Smarter restarts  — instead of fully random initial orderings, half
                         the restarts use a greedy nearest-neighbour seed
                         (start from a random track, always append the
                         cheapest unvisited neighbour)

  3. Dynamic feature weights  — weights are derived from user feedback.
                                 Features where liked vs disliked tracks
                                 differ most are weighted more heavily,
                                 so the optimizer emphasises what the user
                                 actually cares about.

  4. Richer visualisations  — 6-panel figure: convergence, per-transition
                               cost comparison, energy flow, tempo flow,
                               valence flow, and a feature-weight bar chart.

Usage:
  python models/playlist_optimizer.py
  python models/playlist_optimizer.py --playlist_size 20 --restarts 6
"""

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "audio_features.csv"
CKPT_DIR  = PROJECT_ROOT / "checkpoints"

DEFAULT_WEIGHTS = {
    "energy":       0.35,
    "tempo_norm":   0.30,
    "valence":      0.20,
    "danceability": 0.15,
}
TRANSITION_COLS = list(DEFAULT_WEIGHTS.keys())



def compute_dynamic_weights(feedback_log=None, base_weights=DEFAULT_WEIGHTS):
    """
    Derive per-feature weights from user feedback history.
    Features where liked vs disliked tracks differ most get higher weight.
    Falls back to base_weights when feedback is insufficient (<4 entries).
    """
    if not feedback_log or len(feedback_log) < 4:
        logger.info("Using default weights (insufficient feedback data).")
        return np.array(list(base_weights.values()), dtype=np.float32)

    df       = pd.DataFrame(feedback_log)
    liked    = df[df["feedback"] == "like"]
    disliked = df[df["feedback"].isin(["dislike", "skip"])]

    if len(liked) < 2 or len(disliked) < 2:
        logger.info("Using default weights (not enough liked/disliked tracks).")
        return np.array(list(base_weights.values()), dtype=np.float32)

    raw = {}
    for feat in TRANSITION_COLS:
        if feat not in df.columns:
            raw[feat] = base_weights[feat]
            continue
        raw[feat] = abs(liked[feat].mean() - disliked[feat].mean()) + 1e-4

    total = sum(raw.values())
    w_vec = np.array([raw[f] / total for f in TRANSITION_COLS], dtype=np.float32)

    logger.info("Dynamic weights from feedback:")
    for feat, w in zip(TRANSITION_COLS, w_vec):
        logger.info(f"  {feat:<15} {w:.3f}")
    return w_vec



def transition_cost(a, b, w):
    diff = a - b
    return float(np.sqrt(np.dot(w, diff ** 2)))

def playlist_cost(tracks, w):
    return sum(transition_cost(tracks[i], tracks[i+1], w) for i in range(len(tracks)-1))

def worst_transition(tracks, w):
    worst_i, worst_c = 0, 0.0
    for i in range(len(tracks)-1):
        c = transition_cost(tracks[i], tracks[i+1], w)
        if c > worst_c:
            worst_c = c; worst_i = i
    return worst_i, worst_c



def greedy_seed(tracks, w, start=None):
    """
    Build an ordering by always appending the cheapest unvisited track.
    Gives a ~15-30% cheaper starting point than random for hill climbing.
    """
    n       = len(tracks)
    visited = [False] * n
    start   = start if start is not None else np.random.randint(n)
    order   = [start]
    visited[start] = True

    for _ in range(n - 1):
        last = order[-1]
        best_j, best_c = -1, float("inf")
        for j in range(n):
            if not visited[j]:
                c = transition_cost(tracks[last], tracks[j], w)
                if c < best_c:
                    best_c = c; best_j = j
        order.append(best_j)
        visited[best_j] = True

    return tracks[order]

 
def two_opt_pass(tracks, w):
    order    = list(range(len(tracks)))
    best     = playlist_cost(tracks[order], w)
    improved = False

    for i in range(len(order)):
        for j in range(i + 2, len(order)):
            order[i], order[j] = order[j], order[i]
            new_cost = playlist_cost(tracks[order], w)
            if new_cost < best:
                best = new_cost; improved = True
            else:
                order[i], order[j] = order[j], order[i]

    return tracks[order], improved


# ── 1. Or-opt pass ────────────────────────────────────────────────────────────
def or_opt_pass(tracks, w):
    """
    Try removing each track from its current position and inserting it
    at every other position. Finds improvements that 2-opt misses —
    particularly useful for a single track that is badly out of place.
    """
    order    = list(range(len(tracks)))
    best     = playlist_cost(tracks[order], w)
    improved = False

    for i in range(len(order)):
        track_idx = order[i]
        remaining = order[:i] + order[i+1:]

        for j in range(len(remaining) + 1):
            candidate = remaining[:j] + [track_idx] + remaining[j:]
            new_cost  = playlist_cost(tracks[candidate], w)
            if new_cost < best - 1e-9:
                order = candidate; best = new_cost; improved = True
                break   # restart after any improvement

    return tracks[order], improved


# ── Combined hill climbing (2-opt + or-opt) ───────────────────────────────────
def hill_climb(tracks, w, max_iter=1000):
    """Alternate 2-opt and or-opt passes until no improvement is found."""
    current      = tracks.copy()
    cost_history = [playlist_cost(current, w)]
    improved     = True
    iteration    = 0

    while improved and iteration < max_iter:
        improved  = False
        iteration += 1

        current, imp1 = two_opt_pass(current, w)
        if imp1:
            improved = True
            cost_history.append(playlist_cost(current, w))

        current, imp2 = or_opt_pass(current, w)
        if imp2:
            improved = True
            cost_history.append(playlist_cost(current, w))

    return current, cost_history


# ── 2. Multi-restart with greedy seeds ───────────────────────────────────────
def hill_climb_with_restarts(tracks, w, restarts=6, max_iter=500):
    """
    Half restarts use greedy nearest-neighbour seeds for a better starting
    point; the other half use random shuffles for diversity.
    """
    best_tracks  = None
    best_cost    = float("inf")
    best_history = []
    n_greedy     = restarts // 2

    for r in range(restarts):
        if r < n_greedy:
            seeded = greedy_seed(tracks, w, start=np.random.randint(len(tracks)))
            label  = "greedy"
        else:
            seeded = tracks[np.random.permutation(len(tracks))]
            label  = "random"

        result, history = hill_climb(seeded, w, max_iter=max_iter)
        final_cost      = history[-1]
        logger.info(f"  Restart {r+1}/{restarts} [{label}] | cost={final_cost:.4f} | iters={len(history)-1}")

        if final_cost < best_cost:
            best_cost = final_cost; best_tracks = result; best_history = history

    return best_tracks, best_history, best_cost


# ── Feature extraction ────────────────────────────────────────────────────────
def get_transition_features(df):
    df = df.copy()
    if "tempo_norm" not in df.columns:
        lo, hi = df["tempo"].min(), df["tempo"].max()
        df["tempo_norm"] = ((df["tempo"] - lo) / (hi - lo + 1e-8)).clip(0, 1)
    return df[TRANSITION_COLS].fillna(0.0).values.astype(np.float32)


# ── 4. Richer visualisations ──────────────────────────────────────────────────
def plot_results(random_tracks, optimised_tracks, cost_history, w, save_path):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    rand_costs = [transition_cost(random_tracks[i],    random_tracks[i+1],    w) for i in range(len(random_tracks)-1)]
    opt_costs  = [transition_cost(optimised_tracks[i], optimised_tracks[i+1], w) for i in range(len(optimised_tracks)-1)]

    # 1. Convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(cost_history, color="black", linewidth=2)
    ax1.set_title("Convergence (2-opt + or-opt)", fontweight="bold")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Total Cost"); ax1.grid(alpha=0.3)

    # 2. Per-transition cost
    ax2 = fig.add_subplot(gs[0, 1])
    x = range(len(rand_costs))
    ax2.bar(x, rand_costs, color="lightgray", edgecolor="gray",  label="Random",    alpha=0.85)
    ax2.bar(x, opt_costs,  color="black",     edgecolor="black", label="Optimised", alpha=0.6)
    ax2.set_title("Per-Transition Cost", fontweight="bold")
    ax2.set_xlabel("Transition #"); ax2.set_ylabel("Cost")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")

    # 3. Dynamic feature weights
    ax3 = fig.add_subplot(gs[0, 2])
    labels = [c.replace("_norm", "") for c in TRANSITION_COLS]
    shades = ["black", "dimgray", "gray", "lightgray"]
    bars   = ax3.bar(labels, w, color=shades, edgecolor="black")
    ax3.set_title("Feature Weights (dynamic)", fontweight="bold")
    ax3.set_ylabel("Weight")
    ax3.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    for bar, val in zip(bars, w):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax3.grid(alpha=0.3, axis="y")

    # 4–6. Feature flow (energy, tempo, valence)
    for col_i, (feat_idx, label) in enumerate(zip(range(3), ["Energy", "Tempo (norm)", "Valence"])):
        ax = fig.add_subplot(gs[1, col_i])
        ax.plot(random_tracks[:, feat_idx],    color="lightgray", marker="o", markersize=4,
                linewidth=1.5, linestyle="--", label="Random")
        ax.plot(optimised_tracks[:, feat_idx], color="black",     marker="o", markersize=4,
                linewidth=1.5, label="Optimised")
        ax.set_title(f"{label} Flow", fontweight="bold")
        ax.set_xlabel("Track #"); ax.set_ylabel(label)
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Playlist Optimizer v2 — Hill Climbing (2-opt + Or-opt)", fontweight="bold", fontsize=13)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot → {save_path}")
    plt.show()


def print_playlist(tracks_arr, df_sample, label, w):
    print(f"\n── {label} ─────────────────────────────────────────────")
    print(f"  {'#':<3} {'Track':<30} {'Genre':<14} {'Energy':>7} {'Tempo':>7}  {'→ Cost':>8}")
    print(f"  {'─'*76}")
    rows = list(df_sample.iterrows())
    for idx, (_, row) in enumerate(rows):
        cost_str = ""
        if idx < len(rows) - 1:
            cost_str = f"{transition_cost(tracks_arr[idx], tracks_arr[idx+1], w):>8.4f}"
        print(f"  {idx+1:<3} {str(row.get('track_name','?'))[:28]:<30} "
              f"{str(row.get('track_genre','?'))[:13]:<14} "
              f"{float(row.get('energy',0)):>7.3f} "
              f"{float(row.get('tempo',0)):>7.1f}  {cost_str}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(playlist_size=15, restarts=6, feedback_log=None):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.\nRun: python data/load_spotify_dataset.py --source huggingface")

    logger.info("Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    if "tempo_norm" not in df.columns:
        lo, hi = df["tempo"].min(), df["tempo"].max()
        df["tempo_norm"] = ((df["tempo"] - lo) / (hi - lo + 1e-8)).clip(0, 1)

    genres   = df["track_genre"].value_counts().head(10).index.tolist()
    chosen   = np.random.choice(genres, size=min(3, len(genres)), replace=False)
    pool     = df[df["track_genre"].isin(chosen)].sample(min(playlist_size * 4, 200), random_state=42)
    playlist = pool.sample(playlist_size, random_state=7).reset_index(drop=True)
    logger.info(f"Playlist: {playlist_size} tracks from genres: {list(chosen)}")

    tracks_mat = get_transition_features(playlist)
    w          = compute_dynamic_weights(feedback_log)

    rng_order     = np.random.permutation(playlist_size)
    random_tracks = tracks_mat[rng_order]
    random_cost   = playlist_cost(random_tracks, w)
    random_df     = playlist.iloc[rng_order].reset_index(drop=True)

    logger.info(f"Running hill climbing v2 ({restarts} restarts, 2-opt + or-opt) ...")
    opt_tracks, cost_history, opt_cost = hill_climb_with_restarts(tracks_mat, w, restarts=restarts)

    opt_order = []
    for opt_row in opt_tracks:
        for idx, orig_row in enumerate(tracks_mat):
            if np.allclose(opt_row, orig_row) and idx not in opt_order:
                opt_order.append(idx); break
    opt_df = playlist.iloc[opt_order].reset_index(drop=True)

    reduction  = (random_cost - opt_cost) / (random_cost + 1e-8) * 100
    worst_r_i, worst_r_c = worst_transition(random_tracks, w)
    worst_o_i, worst_o_c = worst_transition(opt_tracks, w)

    print(f"\n{'='*55}")
    print(f"  Playlist size:              {playlist_size} tracks")
    print(f"  Random total cost:          {random_cost:.4f}")
    print(f"  Optimised total cost:       {opt_cost:.4f}")
    print(f"  Transition cost reduction:  {reduction:.1f}%")
    print(f"  Worst random transition:    {worst_r_c:.4f} (after track {worst_r_i+1})")
    print(f"  Worst optimised transition: {worst_o_c:.4f} (after track {worst_o_i+1})")
    print(f"  {'✅ TARGET MET (>=40%)' if reduction >= 40 else '🔄 Try --restarts 10'}")
    print(f"{'='*55}")

    print_playlist(random_tracks, random_df,  "Random Order",    w)
    print_playlist(opt_tracks,    opt_df,     "Optimised Order", w)

    CKPT_DIR.mkdir(exist_ok=True)
    opt_df[["track_name","artists","track_genre","energy","tempo","valence","danceability"]].to_csv(
        CKPT_DIR / "optimised_playlist.csv", index=False
    )
    plot_results(random_tracks, opt_tracks, cost_history, w, CKPT_DIR / "playlist_optimizer.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--playlist_size", type=int, default=15)
    parser.add_argument("--restarts",      type=int, default=6)
    args = parser.parse_args()
    main(args.playlist_size, args.restarts)
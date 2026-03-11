"""
explore_dataset.py
Exploratory data analysis on the Spotify Million Playlist Dataset subset.
Loads track-level audio features, reports statistics, and saves cleaned CSV.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
RAW_DIR  = DATA_DIR / "raw"
OUT_DIR  = DATA_DIR / "processed"
OUT_DIR.mkdir(exist_ok=True)

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_mpd_slice(path: Path) -> pd.DataFrame:
    """
    Load one MPD JSON slice and flatten to a track-level DataFrame.
    Each row = one (playlist_id, track) pair, deduplicated on track_uri.
    """
    with open(path) as f:
        data = json.load(f)

    rows = []
    for playlist in data["playlists"]:
        pid = playlist["pid"]
        for track in playlist["tracks"]:
            rows.append({"playlist_id": pid, **track})

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="track_uri")
    return df


def load_audio_features_csv(path: Path) -> pd.DataFrame:
    """Load pre-fetched audio features CSV (output of fetch_features.py)."""
    df = pd.read_csv(path)
    df = df.dropna(subset=AUDIO_FEATURES)
    return df


# ── EDA helpers ───────────────────────────────────────────────────────────────
def feature_distributions(df: pd.DataFrame, save: bool = True) -> None:
    """Plot histograms for all numeric audio features."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for i, feat in enumerate(AUDIO_FEATURES):
        if feat not in df.columns:
            continue
        axes[i].hist(df[feat].dropna(), bins=40, color="#1DB954", edgecolor="black", linewidth=0.3)
        axes[i].set_title(feat, fontsize=11)
        axes[i].set_xlabel("")
        axes[i].tick_params(labelsize=8)

    fig.suptitle("Audio Feature Distributions — Spotify Dataset Subset", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        out = OUT_DIR / "feature_distributions.png"
        plt.savefig(out, dpi=150)
        print(f"[EDA] Saved distribution plot → {out}")
    plt.show()


def correlation_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """Pearson correlation heatmap across audio features."""
    corr = df[AUDIO_FEATURES].corr()

    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        annot_kws={"size": 8},
    )
    plt.title("Audio Feature Correlations", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        out = OUT_DIR / "feature_correlations.png"
        plt.savefig(out, dpi=150)
        print(f"[EDA] Saved correlation heatmap → {out}")
    plt.show()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return summary statistics for audio features."""
    stats = df[AUDIO_FEATURES].describe().T
    stats["missing"] = df[AUDIO_FEATURES].isnull().sum()
    stats["missing_%"] = (stats["missing"] / len(df) * 100).round(2)
    print("\n── Feature Summary ──────────────────────────────────────────────")
    print(stats.to_string())
    print(f"\nTotal tracks: {len(df):,}")
    return stats


def genre_distribution(df: pd.DataFrame, top_n: int = 20, save: bool = True) -> None:
    """Bar chart of top N genres (if 'genre' column exists)."""
    if "genre" not in df.columns:
        print("[EDA] No 'genre' column found — skipping genre distribution.")
        return

    counts = df["genre"].value_counts().head(top_n)
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar", color="#1DB954", edgecolor="black", linewidth=0.4)
    plt.title(f"Top {top_n} Genres", fontsize=13, fontweight="bold")
    plt.xlabel("Genre")
    plt.ylabel("Track Count")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    if save:
        out = OUT_DIR / "genre_distribution.png"
        plt.savefig(out, dpi=150)
        print(f"[EDA] Saved genre distribution → {out}")
    plt.show()


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalise the feature DataFrame.
    - Drop rows missing core audio features
    - Clip loudness to [-60, 0] dB
    - Min-max scale all features to [0, 1]
    - Add binary 'is_explicit' int column if missing
    """
    df = df.dropna(subset=AUDIO_FEATURES).copy()

    # Clip loudness
    df["loudness"] = df["loudness"].clip(-60, 0)

    # Min-max normalisation (per-feature)
    for feat in AUDIO_FEATURES:
        col = df[feat]
        lo, hi = col.min(), col.max()
        df[feat + "_norm"] = (col - lo) / (hi - lo + 1e-8)

    if "explicit" in df.columns:
        df["is_explicit"] = df["explicit"].astype(int)
    else:
        df["is_explicit"] = 0

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Try loading pre-fetched features CSV first
    features_csv = OUT_DIR / "audio_features.csv"

    if features_csv.exists():
        print(f"[EDA] Loading features from {features_csv}")
        df = load_audio_features_csv(features_csv)
    else:
        # Fallback: generate synthetic data for development
        print("[EDA] No dataset found — generating synthetic sample for development.")
        rng = np.random.default_rng(42)
        n = 10_000
        df = pd.DataFrame({
            "track_uri":         [f"spotify:track:{i:07d}" for i in range(n)],
            "track_name":        [f"Track {i}" for i in range(n)],
            "artist_name":       [f"Artist {i % 500}" for i in range(n)],
            "danceability":      rng.beta(4, 2, n),
            "energy":            rng.beta(3, 2, n),
            "loudness":          rng.uniform(-20, -2, n),
            "speechiness":       rng.beta(1, 8, n),
            "acousticness":      rng.beta(2, 5, n),
            "instrumentalness":  rng.beta(1, 10, n),
            "liveness":          rng.beta(2, 8, n),
            "valence":           rng.beta(3, 3, n),
            "tempo":             rng.normal(120, 25, n).clip(40, 220),
            "genre":             rng.choice(
                ["pop", "hip-hop", "rock", "r&b", "electronic",
                 "jazz", "classical", "country", "latin", "metal"],
                n
            ),
        })
        df.to_csv(features_csv, index=False)
        print(f"[EDA] Synthetic dataset saved → {features_csv}")

    # Run EDA
    summarize(df)
    df_clean = preprocess(df)
    df_clean.to_csv(OUT_DIR / "audio_features_clean.csv", index=False)
    print(f"[EDA] Cleaned dataset → {OUT_DIR / 'audio_features_clean.csv'} ({len(df_clean):,} rows)")

    feature_distributions(df_clean)
    correlation_heatmap(df_clean)
    genre_distribution(df_clean)


if __name__ == "__main__":
    main()

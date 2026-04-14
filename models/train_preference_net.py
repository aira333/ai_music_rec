"""
train_preference_net.py
Trains PreferenceNet on the real Spotify dataset using popularity as a
proxy label for user preference.

Label strategy (v2 — wider gap for cleaner signal):
  - popularity >= 65  → label 1 (liked)
  - popularity == 0   → label 0 (disliked)
  - everything else   → dropped

Using pop=0 as negatives is much cleaner than pop<=30 because tracks
with exactly 0 plays are definitively unpopular, not just less popular.
This gives sharper decision boundaries for the model to learn from.

Additional features:
  - One-hot encoded genre (top 20 genres) appended to the feature vector
  - Gives the model genre context without needing user history

Usage:
  python models/train_preference_net.py
  python models/train_preference_net.py --epochs 120 --lr 0.0003
"""

import argparse
import sys
import logging
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from preference_net import PreferenceNet, PreferenceTrainer, PreferenceDataset
from preference_net import AUDIO_DIM, CONTEXT_DIM

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH  = PROJECT_ROOT / "data" / "processed" / "audio_features.csv"
CKPT_DIR   = PROJECT_ROOT / "checkpoints"
TOP_GENRES = 20


def add_genre_features(df: pd.DataFrame, top_n: int = TOP_GENRES) -> tuple:
    """One-hot encode top N genres and append to feature_vec."""
    if "track_genre" not in df.columns:
        logger.warning("No track_genre column — skipping genre features.")
        return df, AUDIO_DIM + CONTEXT_DIM

    top_genres   = df["track_genre"].value_counts().head(top_n).index.tolist()
    genre_to_idx = {g: i for i, g in enumerate(top_genres)}

    def genre_vec(genre):
        vec = [0.0] * top_n
        if genre in genre_to_idx:
            vec[genre_to_idx[genre]] = 1.0
        return vec

    def build_full_vec(row):
        audio = row["feature_vec"]
        if isinstance(audio, str):
            audio = ast.literal_eval(audio)
        return audio + [0.0] * CONTEXT_DIM + genre_vec(row.get("track_genre", ""))

    df = df.copy()
    df["feature_vec"] = df.apply(build_full_vec, axis=1)
    new_dim = AUDIO_DIM + CONTEXT_DIM + top_n
    logger.info(f"Feature dims: {AUDIO_DIM} audio + {CONTEXT_DIM} context + {top_n} genre = {new_dim}")
    return df, new_dim


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """pop>=65 → liked (1), pop==0 → disliked (0), rest dropped."""
    if "popularity" not in df.columns:
        raise ValueError("Missing 'popularity' — re-run load_spotify_dataset.py")

    pos = df[df["popularity"] >= 65].copy(); pos["label"] = 1
    neg = df[df["popularity"] == 0].copy();  neg["label"] = 0

    n_min    = min(len(pos), len(neg))
    combined = pd.concat([
        pos.sample(n_min, random_state=42),
        neg.sample(n_min, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Labels: {n_min:,} pos (pop≥65) + {n_min:,} neg (pop=0) = {len(combined):,} total")
    return combined


def plot_history(history: dict, save_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1DB954")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="#E91429")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training & Validation Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["val_acc"], color="#1DB954")
    ax2.axhline(0.7, color="gray", linestyle="--", label="70% target")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Saved training curves → {save_path}")
    plt.show()


def main(epochs: int = 120, lr: float = 3e-4, batch_size: int = 256):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.\nRun: python data/load_spotify_dataset.py --source huggingface")

    logger.info("Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df):,} tracks.")

    df_labelled          = make_labels(df)
    df_labelled, in_dim  = add_genre_features(df_labelled, top_n=TOP_GENRES)
    dataset              = PreferenceDataset(df_labelled, use_context=False)
    logger.info(f"Dataset: {len(dataset):,} samples | input_dim={in_dim}")

    model = PreferenceNet(
        input_dim=in_dim,
        hidden_dims=(512, 256, 128, 64),
        dropout=0.35,
    )
    trainer = PreferenceTrainer(model, lr=lr, weight_decay=1e-4, checkpoint_dir=CKPT_DIR)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history   = trainer.fit(dataset, epochs=epochs, batch_size=batch_size, val_split=0.15, patience=15)
    best_acc  = max(history["val_acc"])
    final_acc = history["val_acc"][-1]

    print(f"\n{'='*50}")
    print(f"  Best val accuracy:  {best_acc:.3f} ({best_acc*100:.1f}%)")
    print(f"  Final val accuracy: {final_acc:.3f} ({final_acc*100:.1f}%)")
    print(f"  Target:             70.0%")
    print(f"  {'TARGET MET' if best_acc >= 0.70 else 'Try: --lr 0.0001 --epochs 150'}")
    print(f"{'='*50}\n")

    CKPT_DIR.mkdir(exist_ok=True)
    plot_history(history, CKPT_DIR / "training_curves.png")

    print("Sample predictions:")
    for _, row in df_labelled.sample(10, random_state=99).iterrows():
        vec   = row["feature_vec"]
        if isinstance(vec, str): vec = ast.literal_eval(vec)
        score = trainer.predict(np.array(vec, dtype=np.float32))
        true  = int(row["label"])
        print(f"  {'r' if (score>0.5)==bool(true) else 'w'} {'yes' if score>0.5 else 'no'} "
              f"{score:.3f} | pop={int(row.get('popularity',-1)):3d} | "
              f"{str(row.get('track_genre','?')):<14} | {str(row.get('track_name','?'))[:28]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=120)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int,   default=256)
    args = parser.parse_args()
    main(args.epochs, args.lr, args.batch_size)
"""
load_spotify_dataset.py
Loads the maharshipandya Spotify Tracks Dataset (114k tracks, 125 genres)
and saves it as data/processed/audio_features.csv, ready for the rest of the pipeline.

Two ways to get the data — pick whichever is easier:

──────────────────────────────────────────────────────────────────
OPTION A: HuggingFace (easiest, no account needed)
──────────────────────────────────────────────────────────────────
  pip install datasets
  python data/load_spotify_dataset.py --source huggingface

──────────────────────────────────────────────────────────────────
OPTION B: Kaggle (manual download, also free)
──────────────────────────────────────────────────────────────────
  1. Go to https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
  2. Click Download → saves "archive.zip"
  3. Unzip → you get "dataset.csv"
  4. Copy dataset.csv into:  data/raw/dataset.csv
  5. python data/load_spotify_dataset.py --source kaggle

──────────────────────────────────────────────────────────────────
After either option, run:
  python data/explore_dataset.py
  python models/preference_net.py
──────────────────────────────────────────────────────────────────

Dataset info:
  - 114,000 tracks across 125 genres
  - Columns: track_id, artists, album_name, track_name, popularity,
             duration_ms, explicit, danceability, energy, key, loudness,
             mode, speechiness, acousticness, instrumentalness,
             liveness, valence, tempo, time_signature, track_genre
  - Source: Spotify Web API, collected by maharshipandya
  - HuggingFace: https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
  - Kaggle:      https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
"""

import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR  = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR  = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

KEEP_COLS = ["track_id", "track_name", "artists", "track_genre", "popularity"] + AUDIO_FEATURES


# ── Option A: HuggingFace ─────────────────────────────────────────────────────
def load_from_huggingface() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info("Downloading from HuggingFace (maharshipandya/spotify-tracks-dataset)...")
    ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
    df = ds.to_pandas()
    logger.info(f"Downloaded {len(df):,} rows.")
    return df


# ── Option B: Kaggle CSV ──────────────────────────────────────────────────────
def load_from_kaggle() -> pd.DataFrame:
    csv_path = RAW_DIR / "dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\nFile not found: {csv_path}\n\n"
            "Steps to fix:\n"
            "  1. Go to https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset\n"
            "  2. Click Download\n"
            "  3. Unzip the archive\n"
            "  4. Copy dataset.csv → data/raw/dataset.csv\n"
            "  5. Re-run this script\n"
        )
    logger.info(f"Loading from {csv_path} ...")
    df = pd.read_csv(csv_path, index_col=0)
    logger.info(f"Loaded {len(df):,} rows.")
    return df


# ── Shared cleaning ───────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Rename if needed (HuggingFace uses 'artists' as a string list repr)
    if "artists" not in df.columns and "artist_name" in df.columns:
        df = df.rename(columns={"artist_name": "artists"})

    # Drop missing audio features
    df = df.dropna(subset=AUDIO_FEATURES)

    # Drop duplicate tracks (same track_id)
    df = df.drop_duplicates(subset="track_id")

    # Clip loudness
    df["loudness"] = df["loudness"].clip(-60, 0)

    # Keep only columns we need
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    # Normalised feature columns (used by preference_net and q_agent)
    for feat in AUDIO_FEATURES:
        lo = df[feat].min()
        hi = df[feat].max()
        df[feat + "_norm"] = ((df[feat] - lo) / (hi - lo + 1e-8)).clip(0, 1).astype("float32")

    # feature_vec: list of 9 normalised values — what the neural net ingests
    norm_cols = [f + "_norm" for f in AUDIO_FEATURES]
    df["feature_vec"] = df[norm_cols].values.tolist()

    logger.info(f"After cleaning: {len(df):,} tracks | {df['track_genre'].nunique()} genres")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main(source: str = "huggingface"):
    if source == "huggingface":
        df_raw = load_from_huggingface()
    elif source == "kaggle":
        df_raw = load_from_kaggle()
    else:
        raise ValueError("--source must be 'huggingface' or 'kaggle'")

    df = clean(df_raw)

    out = OUT_DIR / "audio_features.csv"
    df.to_csv(out, index=False)
    logger.info(f"\n✅ Saved → {out}")

    # Quick preview
    print("\nSample rows:")
    print(df[["track_name", "artists", "track_genre"] + AUDIO_FEATURES[:4]].head(8).to_string(index=False))
    print(f"\nGenre distribution (top 10):")
    print(df["track_genre"].value_counts().head(10).to_string())
    print(f"\nNext step:  python data/explore_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["huggingface", "kaggle"],
        default="huggingface",
        help="Where to load the dataset from (default: huggingface)",
    )
    args = parser.parse_args()
    main(args.source)

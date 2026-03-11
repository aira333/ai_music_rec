import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR  = Path("data/raw")
OUT_DIR  = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

KEEP_COLS = ["track_id", "track_name", "artists", "track_genre", "popularity"] + AUDIO_FEATURES


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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if "artists" not in df.columns and "artist_name" in df.columns:
        df = df.rename(columns={"artist_name": "artists"})
  
    df = df.dropna(subset=AUDIO_FEATURES)

    df = df.drop_duplicates(subset="track_id")

    df["loudness"] = df["loudness"].clip(-60, 0)

    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    for feat in AUDIO_FEATURES:
        lo = df[feat].min()
        hi = df[feat].max()
        df[feat + "_norm"] = ((df[feat] - lo) / (hi - lo + 1e-8)).clip(0, 1).astype("float32")

    norm_cols = [f + "_norm" for f in AUDIO_FEATURES]
    df["feature_vec"] = df[norm_cols].values.tolist()

    logger.info(f"After cleaning: {len(df):,} tracks | {df['track_genre'].nunique()} genres")
    return df


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

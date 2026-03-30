"""
feature_extractor.py
Extracts and caches audio features for tracks.

Two backends are supported:
  1. Spotify Web API (spotipy)  — for tracks with known Spotify URIs
  2. librosa                    — for local audio files (.mp3 / .wav)

The module outputs a standardised 9-dim feature vector:
  [danceability, energy, loudness_norm, speechiness,
   acousticness, instrumentalness, liveness, valence, tempo_norm]
"""

import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

FEATURE_DIM = len(FEATURE_NAMES)

# Normalisation bounds (from Spotify API documentation + empirical ranges)
_NORM_BOUNDS = {
    "danceability":      (0.0,   1.0),
    "energy":            (0.0,   1.0),
    "loudness":          (-60.0, 0.0),
    "speechiness":       (0.0,   1.0),
    "acousticness":      (0.0,   1.0),
    "instrumentalness":  (0.0,   1.0),
    "liveness":          (0.0,   1.0),
    "valence":           (0.0,   1.0),
    "tempo":             (40.0,  250.0),
}


# ── Normalisation ─────────────────────────────────────────────────────────────
def normalise_features(raw: dict) -> np.ndarray:
    """
    Map raw Spotify/librosa feature dict → normalised [0, 1] numpy vector.

    Args:
        raw: dict with keys matching FEATURE_NAMES.

    Returns:
        np.ndarray of shape (FEATURE_DIM,), dtype float32.
    """
    vec = np.empty(FEATURE_DIM, dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        lo, hi = _NORM_BOUNDS[name]
        val = float(raw.get(name, (lo + hi) / 2))
        vec[i] = np.clip((val - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return vec


# ── Spotify backend ───────────────────────────────────────────────────────────
class SpotifyFeatureExtractor:
    """
    Batch-fetches audio features from the Spotify Web API.
    Requires SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in env / .env file.
    Rate-limits requests to ~30 reqs/s.
    """

    BATCH_SIZE = 100  # Spotify allows up to 100 IDs per request

    def __init__(self):
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
        except ImportError:
            raise ImportError("Install spotipy: pip install spotipy")

        client_id     = os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise EnvironmentError(
                "Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in your .env file."
            )

        auth = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        )
        self.sp = spotipy.Spotify(auth_manager=auth)
        logger.info("SpotifyFeatureExtractor initialised.")

    def fetch_batch(self, track_ids: list[str]) -> list[Optional[dict]]:
        """
        Fetch audio features for up to 100 track IDs.

        Args:
            track_ids: list of Spotify track IDs (not full URIs).

        Returns:
            List of raw feature dicts (None for tracks that failed).
        """
        results = []
        for i in range(0, len(track_ids), self.BATCH_SIZE):
            chunk = track_ids[i : i + self.BATCH_SIZE]
            try:
                features = self.sp.audio_features(chunk)  # returns list of dicts or None
                results.extend(features or [None] * len(chunk))
            except Exception as e:
                logger.warning(f"Spotify API error for chunk {i}: {e}")
                results.extend([None] * len(chunk))
            time.sleep(0.1)  # polite rate limiting
        return results

    def extract_dataframe(self, track_ids: list[str]) -> pd.DataFrame:
        """
        Fetch and normalise features for a list of track IDs.

        Returns:
            DataFrame with columns = FEATURE_NAMES + ['track_id', 'feature_vec']
        """
        raw_list = self.fetch_batch(track_ids)
        rows = []
        for tid, raw in zip(track_ids, raw_list):
            if raw is None:
                continue
            row = {name: raw.get(name) for name in FEATURE_NAMES}
            row["track_id"]    = tid
            row["feature_vec"] = normalise_features(raw).tolist()
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Fetched features for {len(df)}/{len(track_ids)} tracks.")
        return df


# ── librosa backend ───────────────────────────────────────────────────────────
class LibrosaFeatureExtractor:
    """
    Extracts audio features locally from .mp3 / .wav files using librosa.
    Maps to the same 9-dim feature space as the Spotify backend.

    Note: librosa does not produce exactly the same features as Spotify's
    proprietary algorithms, but provides strong approximations for:
      energy, tempo, acousticness (spectral flatness), liveness (RMSE),
      and valence (mode + chroma).
    Fields not reliably estimated (danceability, speechiness,
    instrumentalness) are set to their prior means.
    """

    def __init__(self, sr: int = 22050, duration: float = 30.0):
        try:
            import librosa  # noqa: F401
        except ImportError:
            raise ImportError("Install librosa: pip install librosa")
        self.sr       = sr
        self.duration = duration

    def extract_file(self, path: Union[str, Path]) -> np.ndarray:
        """
        Extract normalised feature vector from a local audio file.

        Args:
            path: Path to .mp3 or .wav file.

        Returns:
            np.ndarray of shape (FEATURE_DIM,).
        """
        import librosa

        y, sr = librosa.load(str(path), sr=self.sr, duration=self.duration, mono=True)

        # ── Tempo ──────────────────────────────────────────────────────────
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)

        # ── Energy (RMS) ───────────────────────────────────────────────────
        rms    = float(np.mean(librosa.feature.rms(y=y)))
        energy = np.clip(rms * 10, 0.0, 1.0)  # rough normalisation

        # ── Loudness (dBFS) ────────────────────────────────────────────────
        loudness = float(librosa.amplitude_to_db(np.array([rms])).item())
        loudness = np.clip(loudness, -60.0, 0.0)

        # ── Acousticness (inverse spectral flatness) ───────────────────────
        flatness     = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        acousticness = float(np.clip(1.0 - flatness * 10, 0.0, 1.0))

        # ── Liveness (high-frequency energy ratio) ─────────────────────────
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        liveness = float(np.clip(centroid / 8000.0, 0.0, 1.0))

        # ── Valence proxy (major/minor mode via chroma) ────────────────────
        chroma  = librosa.feature.chroma_cqt(y=y, sr=sr)
        major   = float(np.mean(chroma[[0, 4, 7], :]))   # C-E-G intervals
        minor   = float(np.mean(chroma[[0, 3, 7], :]))   # C-Eb-G intervals
        valence = float(np.clip((major - minor + 0.1) / 0.2, 0.0, 1.0))

        raw = {
            "danceability":     0.5,   # not reliably estimated
            "energy":           energy,
            "loudness":         loudness,
            "speechiness":      0.1,   # not reliably estimated
            "acousticness":     acousticness,
            "instrumentalness": 0.5,   # not reliably estimated
            "liveness":         liveness,
            "valence":          valence,
            "tempo":            tempo,
        }
        return normalise_features(raw)

    def extract_directory(self, directory: Union[str, Path]) -> pd.DataFrame:
        """
        Batch-extract features for all audio files in a directory.

        Returns:
            DataFrame with columns = FEATURE_NAMES + ['file', 'track_id', 'feature_vec']
        """
        directory = Path(directory)
        files     = list(directory.glob("*.mp3")) + list(directory.glob("*.wav"))
        rows      = []

        for fp in files:
            try:
                vec = self.extract_file(fp)
                row = {name: float(vec[i]) for i, name in enumerate(FEATURE_NAMES)}
                row["file"]        = fp.name
                row["track_id"]    = hashlib.md5(fp.name.encode()).hexdigest()[:16]
                row["feature_vec"] = vec.tolist()
                rows.append(row)
                logger.info(f"Extracted: {fp.name}")
            except Exception as e:
                logger.warning(f"Failed to process {fp.name}: {e}")

        df = pd.DataFrame(rows)
        logger.info(f"Extracted {len(df)}/{len(files)} files.")
        return df


# ── Feature cache ─────────────────────────────────────────────────────────────
class FeatureCache:
    """
    Simple CSV-backed cache so we don't re-fetch features we already have.

    Usage:
        cache = FeatureCache("data/processed/audio_features.csv")
        missing = cache.get_missing(track_ids)
        # fetch 'missing' then:
        cache.update(new_df)
        vec = cache.lookup("track_id_xyz")
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if self.path.exists():
            self._df = pd.read_csv(self.path)
        else:
            self._df = pd.DataFrame(columns=["track_id"] + FEATURE_NAMES + ["feature_vec"])

    def get_missing(self, track_ids: list[str]) -> list[str]:
        cached = set(self._df["track_id"].values)
        return [t for t in track_ids if t not in cached]

    def update(self, new_df: pd.DataFrame) -> None:
        self._df = pd.concat([self._df, new_df], ignore_index=True)
        self._df = self._df.drop_duplicates(subset="track_id")
        self._df.to_csv(self.path, index=False)
        logger.info(f"Cache updated → {self.path} ({len(self._df)} tracks)")

    def lookup(self, track_id: str) -> Optional[np.ndarray]:
        row = self._df[self._df["track_id"] == track_id]
        if row.empty:
            return None
        import ast
        vec = row.iloc[0]["feature_vec"]
        if isinstance(vec, str):
            vec = ast.literal_eval(vec)
        return np.array(vec, dtype=np.float32)

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=== Feature Extractor Demo ===\n")

    # Synthetic demo — no API keys needed
    rng = np.random.default_rng(0)
    synthetic_features = [
        {name: float(rng.uniform(*_NORM_BOUNDS[name])) for name in FEATURE_NAMES}
        for _ in range(5)
    ]

    print("Raw (unnormalised) features for 5 synthetic tracks:")
    for i, raw in enumerate(synthetic_features):
        vec = normalise_features(raw)
        print(f"  Track {i}: {np.round(vec, 3)}")

    # Cache demo
    cache = FeatureCache(Path("data/processed/demo_cache.csv"))
    rows  = []
    for i, raw in enumerate(synthetic_features):
        row              = {name: raw[name] for name in FEATURE_NAMES}
        row["track_id"]  = f"demo_{i:04d}"
        row["feature_vec"] = normalise_features(raw).tolist()
        rows.append(row)

    cache.update(pd.DataFrame(rows))
    vec = cache.lookup("demo_0002")
    print(f"\nCache lookup demo_0002 → {np.round(vec, 3)}")
    print("\nDone. See data/processed/demo_cache.csv")

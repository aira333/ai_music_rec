# AI Music Recommendation System
**CS 5100: Foundations of Artificial Intelligence | Spring 2026**
*Anjali Saumithri Devi Inturi*

## Project Overview
An AI-powered music recommendation system using reinforcement learning (Q-Learning) and neural networks (PyTorch) to learn and adapt to individual user preferences in real-time.

## Architecture
- `data/` — Dataset loading, preprocessing, and exploration scripts
- `models/` — Neural network, Q-Learning agent, playlist optimizer
- `utils/` — Audio feature extraction, metrics, logging helpers
- `notebooks/` — EDA and experiment notebooks
- `checkpoints/` — Saved model weights
- `logs/` — Training logs

## Quickstart
```bash
pip install -r requirements.txt
python data/explore_dataset.py          # EDA on Spotify dataset
python utils/feature_extractor.py       # Extract audio features
python models/train_preference_net.py   # Train user preference model
```

## Components
| Module | Status | Description |
|--------|--------|-------------|
| Dataset Exploration | ✅ Done | EDA on Spotify Million Playlist subset |
| Feature Extraction | ✅ Done | 9 audio features via spotipy/librosa |
| Preference Network | 🔄 In Progress | PyTorch MLP for preference prediction |
| Q-Learning Agent | 🔜 Next | RL agent for adaptive song selection |
| Playlist Optimizer | 🔜 Planned | Hill climbing for smooth transitions |
| Streamlit UI | 🔜 Planned | Interactive feedback interface |

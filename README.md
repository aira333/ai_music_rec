# AI Music Recommendation System
**CS 5100: Foundations of Artificial Intelligence | Spring 2026**
*Anjali Saumithri Devi Inturi*

## Project Overview
An AI-powered music recommendation system that learns and adapts to individual user preferences in real-time using reinforcement learning and neural networks. The system observes user feedback (likes, skips, replays) to continuously improve recommendations, and arranges liked tracks into smooth playlists using a hill climbing optimizer.

## Results

| Component | Target | Result |
|---|---|---|
| PreferenceNet (Neural Network) | ≥70% val accuracy | ✅ **70.0%** |
| Q-Learning Agent (DQN) | ≥30% improvement over random | ✅ **+376.7%** |
| Playlist Optimizer (Hill Climbing) | ≥40% transition cost reduction | ✅ **41.9%** |

## Architecture

```
data/
  load_spotify_dataset.py   — Download 114k real Spotify tracks (HuggingFace / Kaggle)
  explore_dataset.py        — EDA: distributions, correlations, genre breakdown

utils/
  feature_extractor.py      — 9-dim normalised audio feature vectors (Spotipy + librosa)

models/
  preference_net.py         — PyTorch MLP: predicts user preference score [0, 1]
  train_preference_net.py   — Training script with popularity-based labels + genre encoding
  q_agent.py                — Double Dueling DQN: adaptive track selection
  playlist_optimizer.py     — Hill climbing (2-opt + or-opt): smooth playlist ordering

checkpoints/
  best_model.pt             — Trained PreferenceNet weights
  q_agent.pt                — Trained DQN agent weights
  training_curves.png       — PreferenceNet loss + accuracy curves
  rl_training.png           — DQN reward vs random baseline
  playlist_optimizer.png    — Optimisation convergence + transition cost comparison

app.py                      — Streamlit web UI
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Download real Spotify dataset (114k tracks, 113 genres)
python data/load_spotify_dataset.py --source huggingface

# 2. Exploratory data analysis
python data/explore_dataset.py

# 3. Train preference network
python models/train_preference_net.py

# 4. Train Q-Learning agent
python models/q_agent.py

# 5. Run playlist optimizer
python models/playlist_optimizer.py

# 6. Launch Streamlit app
streamlit run app.py
```

## Dataset
[maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) — 114,000 Spotify tracks across 125 genres with full audio features (danceability, energy, tempo, valence, etc.) collected via the Spotify Web API.

## Model Details

### PreferenceNet
- Architecture: MLP with layers 512 → 256 → 128 → 64, BatchNorm + Dropout (0.35)
- Input: 9 audio features + 9 context (rolling mean) + 20 genre one-hot = 38 dims
- Labels: popularity ≥ 65 → liked, popularity = 0 → disliked (cleaner signal than soft threshold)
- Training: AdamW, lr=3e-4, early stopping, patience=15

### Q-Learning Agent (Double Dueling DQN)
- Architecture: Dueling QNetwork with shared layers + separate value/advantage streams
- State: rolling mean of last 10 played tracks (9-dim feature vector)
- Action: select one of 30 candidate tracks
- Reward: energy quartile match (0.6) + popularity bonus (0.2) + transition smoothness (0.2)
- Candidate pool always contains ⅓ matching, ⅓ neutral, ⅓ mismatching tracks

### Playlist Optimizer
- Algorithm: Hill climbing combining 2-opt swaps and or-opt relocations
- Restarts: 6 total — half greedy nearest-neighbour seeds, half random shuffles
- Cost metric: weighted L2 distance (energy 0.35, tempo 0.30, valence 0.20, danceability 0.15)
- Dynamic weights: adjustable based on user feedback patterns

## Ethical Considerations
- **Filter bubbles**: The system balances exploitation (user preferences) with exploration (new genres) via ε-greedy action selection
- **Artist fairness**: Candidate pools are sampled uniformly across genres to avoid popularity bias
- **Over-personalisation**: Rolling history window (10 tracks) prevents the model from over-fitting to short-term mood
"""
app.py
Streamlit web interface for the AI Music Recommendation System.

Run:
  streamlit run app.py

Features:
  - Load real Spotify dataset
  - Set your taste profile (energy, genre preference)
  - Get AI-recommended tracks from the DQN agent
  - Give feedback (like / skip / dislike) to update recommendations
  - Optimise your playlist order with hill climbing
  - View live charts of your feedback history
"""

import sys
import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from q_agent import QLearningAgent, MusicEnvironment, FEATURE_DIM, FEATURE_COLS
from playlist_optimizer import get_transition_features, hill_climb_with_restarts, playlist_cost

logging.basicConfig(level=logging.WARNING)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "audio_features.csv"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "q_agent.pt"

st.set_page_config(
    page_title="AI Music Recommender",
    page_icon="🎵",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0d0d0d; }
  h1, h2, h3 { color: #1DB954; }
  .stButton > button {
    border-radius: 20px;
    font-weight: 600;
  }
  .track-card {
    background: #1a1a1a;
    border-left: 4px solid #1DB954;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
  }
  .metric-val { font-size: 1.4rem; font-weight: 700; color: #1DB954; }
</style>
""", unsafe_allow_html=True)


# ── Data & model loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "tempo_norm" not in df.columns:
        lo, hi = df["tempo"].min(), df["tempo"].max()
        df["tempo_norm"] = ((df["tempo"] - lo) / (hi - lo + 1e-8)).clip(0, 1)
    if "loudness_norm" not in df.columns:
        lo, hi = df["loudness"].min(), df["loudness"].max()
        df["loudness_norm"] = ((df["loudness"] - lo) / (hi - lo + 1e-8)).clip(0, 1)
    return df

@st.cache_resource
def load_agent():
    agent = QLearningAgent(state_dim=FEATURE_DIM, n_actions=30)
    if CKPT_PATH.exists():
        agent.load(CKPT_PATH)
        agent.epsilon_start = 0.05   # mostly exploit in UI mode
    return agent

def get_track_vec(track: pd.Series) -> np.ndarray:
    vec = []
    for col in FEATURE_COLS:
        vec.append(float(track.get(col, 0.0)))
    return np.array(vec, dtype=np.float32)


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "history":          [],    # list of feature vecs of played tracks
        "feedback_log":     [],    # list of {"track", "feedback", "score"}
        "playlist":         [],    # accumulated liked tracks
        "current_track":    None,
        "candidates":       [],
        "page":             "recommend",
        "energy_pref":      2,     # 0-3 quartile
        "initialized":      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Helper: get next recommendation ──────────────────────────────────────────
def get_recommendation(df, agent, energy_q: int):
    """Sample candidates (⅓ match, ⅓ neutral, ⅓ mismatch) and pick best."""
    df_copy = df.copy()
    df_copy["energy_q"] = pd.qcut(df_copy["energy"], q=4, labels=False)

    mismatch_q = (energy_q + 2) % 4
    neutral_q  = (energy_q + 1) % 4

    n = 10
    try:
        match    = df_copy[df_copy["energy_q"] == energy_q].sample(n)
        neutral  = df_copy[df_copy["energy_q"] == neutral_q].sample(n)
        mismatch = df_copy[df_copy["energy_q"] == mismatch_q].sample(n)
        pool     = pd.concat([match, neutral, mismatch]).reset_index(drop=True)
    except Exception:
        pool = df_copy.sample(30).reset_index(drop=True)

    state = (np.mean(st.session_state["history"], axis=0).astype(np.float32)
             if st.session_state["history"]
             else np.zeros(FEATURE_DIM, dtype=np.float32))

    # Score each candidate with the agent
    scores = []
    for i in range(len(pool)):
        vec = get_track_vec(pool.iloc[i])
        with __import__("torch").no_grad():
            import torch
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals = agent.policy_net(s)
            scores.append(float(q_vals[0, i % agent.n_actions].item()))

    best_idx = int(np.argmax(scores))
    return pool.iloc[best_idx], pool, scores


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(df):
    with st.sidebar:
        st.markdown("## 🎵 AI Music Rec")
        st.markdown("---")

        st.markdown("### Your Taste Profile")
        energy_label = st.select_slider(
            "Energy Preference",
            options=["🧘 Calm", "😌 Relaxed", "😊 Upbeat", "🔥 High Energy"],
            value=["🧘 Calm", "😌 Relaxed", "😊 Upbeat", "🔥 High Energy"][st.session_state["energy_pref"]],
        )
        energy_map = {"🧘 Calm": 0, "😌 Relaxed": 1, "😊 Upbeat": 2, "🔥 High Energy": 3}
        st.session_state["energy_pref"] = energy_map[energy_label]

        genres = ["Any"] + sorted(df["track_genre"].dropna().unique().tolist())
        st.selectbox("Favourite Genre", genres, key="genre_filter")

        st.markdown("---")
        st.markdown("### Navigation")
        if st.button("🎧 Recommendations", use_container_width=True):
            st.session_state["page"] = "recommend"
        if st.button("📋 My Playlist", use_container_width=True):
            st.session_state["page"] = "playlist"
        if st.button("📊 My Stats", use_container_width=True):
            st.session_state["page"] = "stats"

        st.markdown("---")
        st.markdown("### Session Stats")
        n_liked   = sum(1 for x in st.session_state["feedback_log"] if x["feedback"] == "like")
        n_skipped = sum(1 for x in st.session_state["feedback_log"] if x["feedback"] == "skip")
        n_total   = len(st.session_state["feedback_log"])
        st.metric("Tracks Heard",  n_total)
        st.metric("Liked",         n_liked)
        st.metric("Skipped",       n_skipped)

        if st.button("🔄 Reset Session", use_container_width=True):
            for k in ["history","feedback_log","playlist","current_track","candidates"]:
                st.session_state[k] = [] if isinstance(st.session_state[k], list) else None
            st.rerun()


# ── Page: Recommendations ─────────────────────────────────────────────────────
def page_recommend(df, agent):
    st.title("🎧 Your Next Track")
    st.markdown("The AI learns your taste with every like and skip.")

    # Filter by genre if set
    genre = st.session_state.get("genre_filter", "Any")
    df_filtered = df[df["track_genre"] == genre] if genre != "Any" else df

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("▶️  Get Recommendation", type="primary", use_container_width=True) or \
           st.session_state["current_track"] is None:
            track, pool, scores = get_recommendation(
                df_filtered, agent, st.session_state["energy_pref"]
            )
            st.session_state["current_track"] = track
            st.session_state["candidates"]    = pool
            st.session_state["_scores"]       = scores

        track = st.session_state["current_track"]
        if track is not None:
            st.markdown(f"""
            <div class="track-card">
              <h2 style="margin:0; color:#fff">{track.get('track_name','Unknown')}</h2>
              <p style="color:#aaa; margin:4px 0">{track.get('artists','Unknown Artist')}</p>
              <p style="color:#1DB954; margin:0"><b>{track.get('track_genre','').replace('-',' ').title()}</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Feature bars
            st.markdown("**Audio Features**")
            features = {
                "Energy":       float(track.get("energy", 0)),
                "Danceability": float(track.get("danceability", 0)),
                "Valence":      float(track.get("valence", 0)),
                "Acousticness": float(track.get("acousticness", 0)),
            }
            for name, val in features.items():
                st.progress(val, text=f"{name}: {val:.2f}")

            # Extra info
            c1, c2, c3 = st.columns(3)
            c1.metric("Tempo",      f"{float(track.get('tempo', 0)):.0f} BPM")
            c2.metric("Popularity", f"{float(track.get('popularity', 0)):.0f}/100")
            c3.metric("Loudness",   f"{float(track.get('loudness', 0)):.1f} dB")

            # Feedback buttons
            st.markdown("**How was it?**")
            fb1, fb2, fb3, fb4 = st.columns(4)
            feedback = None

            if fb1.button("❤️ Like",    use_container_width=True): feedback = "like"
            if fb2.button("⏭️ Skip",    use_container_width=True): feedback = "skip"
            if fb3.button("👎 Dislike", use_container_width=True): feedback = "dislike"
            if fb4.button("🔁 Replay",  use_container_width=True): feedback = "replay"

            if feedback:
                vec = get_track_vec(track)
                st.session_state["history"].append(vec)
                if len(st.session_state["history"]) > 10:
                    st.session_state["history"].pop(0)

                st.session_state["feedback_log"].append({
                    "track":    track.get("track_name", "?"),
                    "genre":    track.get("track_genre", "?"),
                    "feedback": feedback,
                    "energy":   float(track.get("energy", 0)),
                })

                if feedback in ("like", "replay"):
                    st.session_state["playlist"].append(track)
                    st.success(f"Added **{track.get('track_name','?')}** to your playlist!")

                st.session_state["current_track"] = None
                st.rerun()

    with col2:
        st.markdown("### 🔍 Why this track?")
        if st.session_state["history"]:
            avg_energy = np.mean([h[FEATURE_COLS.index("energy")] for h in st.session_state["history"]])
            pref_label = ["Calm", "Relaxed", "Upbeat", "High Energy"][st.session_state["energy_pref"]]
            st.info(
                f"**Your profile:** {pref_label} energy\n\n"
                f"**Avg listened energy:** {avg_energy:.2f}\n\n"
                f"**Tracks in memory:** {len(st.session_state['history'])}/10\n\n"
                f"The DQN agent selected this track based on your listening history "
                f"and energy preference."
            )
        else:
            st.info("Start liking and skipping tracks — the AI will learn your taste and improve its recommendations.")

        # Mini feedback history
        if st.session_state["feedback_log"]:
            st.markdown("### Recent Feedback")
            for item in reversed(st.session_state["feedback_log"][-5:]):
                icon = {"like":"❤️","skip":"⏭️","dislike":"👎","replay":"🔁"}.get(item["feedback"],"")
                st.markdown(f"{icon} {item['track'][:25]}")


# ── Page: Playlist ────────────────────────────────────────────────────────────
def page_playlist(df):
    st.title("📋 My Playlist")

    playlist = st.session_state["playlist"]
    if not playlist:
        st.info("Like some tracks first — they'll appear here.")
        return

    pl_df = pd.DataFrame(playlist)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### {len(pl_df)} tracks")

        if st.button("✨ Optimise Order (Hill Climbing)", type="primary"):
            if len(pl_df) < 4:
                st.warning("Add at least 4 tracks first.")
            else:
                with st.spinner("Optimising playlist order..."):
                    # Ensure normalised columns
                    if "tempo_norm" not in pl_df.columns:
                        lo, hi = df["tempo"].min(), df["tempo"].max()
                        pl_df["tempo_norm"] = ((pl_df["tempo"] - lo) / (hi - lo + 1e-8)).clip(0, 1)

                    tracks_mat = get_transition_features(pl_df)
                    rand_cost  = playlist_cost(tracks_mat[np.random.permutation(len(tracks_mat))])
                    opt_tracks, history, opt_cost = hill_climb_with_restarts(tracks_mat, restarts=3)
                    reduction  = (rand_cost - opt_cost) / (rand_cost + 1e-8) * 100

                    # Reconstruct order
                    opt_order = []
                    for opt_row in opt_tracks:
                        for idx, orig_row in enumerate(tracks_mat):
                            if np.allclose(opt_row, orig_row) and idx not in opt_order:
                                opt_order.append(idx)
                                break
                    pl_df = pl_df.iloc[opt_order].reset_index(drop=True)
                    st.session_state["playlist"] = [pl_df.iloc[i] for i in range(len(pl_df))]

                st.success(f"✅ Playlist optimised! Transition cost reduced by **{reduction:.1f}%**")
                st.rerun()

        for i, (_, row) in enumerate(pl_df.iterrows()):
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                f"**{i+1}. {row.get('track_name','?')}** — "
                f"*{row.get('artists','?')}* · "
                f"{str(row.get('track_genre','')).replace('-',' ').title()}"
            )
            c2.markdown(f"`E:{float(row.get('energy',0)):.2f}`")

    with col2:
        if len(pl_df) >= 2:
            st.markdown("### Energy Flow")
            fig, ax = plt.subplots(figsize=(4, 3))
            energies = [float(r.get("energy", 0)) for _, r in pl_df.iterrows()]
            ax.plot(energies, color="#1DB954", marker="o", markersize=5, linewidth=2)
            ax.fill_between(range(len(energies)), energies, alpha=0.15, color="#1DB954")
            ax.set_xlabel("Track #"); ax.set_ylabel("Energy")
            ax.set_ylim(0, 1); ax.grid(alpha=0.3)
            fig.patch.set_facecolor("#0d0d0d")
            ax.set_facecolor("#0d0d0d")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
            st.pyplot(fig)
            plt.close(fig)


# ── Page: Stats ───────────────────────────────────────────────────────────────
def page_stats():
    st.title("📊 My Listening Stats")
    log = st.session_state["feedback_log"]

    if not log:
        st.info("No data yet — start getting recommendations!")
        return

    log_df = pd.DataFrame(log)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tracks", len(log_df))
    col2.metric("Liked",  int((log_df["feedback"] == "like").sum()))
    col3.metric("Skipped", int((log_df["feedback"] == "skip").sum()))

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Feedback Breakdown")
        counts = log_df["feedback"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        colors  = {"like":"#1DB954","skip":"#E91429","dislike":"#777","replay":"#1ed760"}
        ax.pie(
            counts.values,
            labels=[f"{k} ({v})" for k, v in counts.items()],
            colors=[colors.get(k, "#555") for k in counts.index],
            autopct="%1.0f%%", startangle=90,
        )
        fig.patch.set_facecolor("#0d0d0d")
        st.pyplot(fig); plt.close(fig)

    with col_b:
        st.markdown("### Energy Over Time")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(log_df["energy"].values, color="#1DB954", linewidth=2)
        fb_colors = {"like":"#1DB954","skip":"#E91429","dislike":"#777","replay":"#1ed760"}
        for i, row in log_df.iterrows():
            ax.scatter(i, row["energy"], color=fb_colors.get(row["feedback"],"gray"),
                       s=60, zorder=5)
        ax.set_xlabel("Track #"); ax.set_ylabel("Energy")
        ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        fig.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        st.pyplot(fig); plt.close(fig)

    if len(log_df) >= 3:
        st.markdown("### Genre Breakdown")
        genre_counts = log_df["genre"].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(8, 3))
        genre_counts.plot(kind="bar", ax=ax, color="#1DB954", edgecolor="black")
        ax.set_xlabel("Genre"); ax.set_ylabel("Count"); ax.set_xticklabels(
            [g.replace("-"," ").title() for g in genre_counts.index], rotation=30, ha="right"
        )
        ax.grid(alpha=0.3, axis="y")
        fig.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        st.pyplot(fig); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`.\nRun: `python data/load_spotify_dataset.py --source huggingface`")
        return

    df    = load_data()
    agent = load_agent()

    render_sidebar(df)

    page = st.session_state["page"]
    if page == "recommend":
        page_recommend(df, agent)
    elif page == "playlist":
        page_playlist(df)
    elif page == "stats":
        page_stats()


if __name__ == "__main__":
    main()
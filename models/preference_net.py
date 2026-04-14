"""
preference_net.py
PyTorch neural network that predicts user preference scores for tracks.

Architecture:
  Input:  [audio_features (9-dim) + context_features (k-dim)]
  Hidden: 3 × fully-connected layers with BatchNorm + Dropout
  Output: scalar preference score in [0, 1]

Temporal context (optional) is encoded by prepending a rolling
mean of the last T liked/disliked tracks' feature vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


AUDIO_DIM   = 9    # feature_extractor.FEATURE_DIM
CONTEXT_DIM = 9    # rolling mean of last T tracks (same dim as audio)
INPUT_DIM   = AUDIO_DIM + CONTEXT_DIM   # 18 when context is used


class PreferenceDataset(Dataset):
    """
    Wraps a DataFrame of (feature_vec, label) pairs.

    Args:
        df:        DataFrame with 'feature_vec' (list[float]) and 'label' (0/1) columns.
        context:   Optional dict mapping track_id → context_vec (np.ndarray).
        use_context: Whether to concatenate context vectors with audio features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        context: Optional[dict] = None,
        use_context: bool = True,
    ):
        self.use_context = use_context
        self.context     = context or {}

        self.features = []
        self.labels   = []

        for _, row in df.iterrows():
            import ast
            vec = row["feature_vec"]
            if isinstance(vec, str):
                vec = ast.literal_eval(vec)
            vec = np.array(vec, dtype=np.float32)

            if use_context:
                ctx = self.context.get(row.get("track_id", ""), np.zeros(CONTEXT_DIM, dtype=np.float32))
                vec = np.concatenate([vec, ctx])

            self.features.append(torch.tensor(vec))
            self.labels.append(torch.tensor(float(row["label"]), dtype=torch.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PreferenceNet(nn.Module):
    """
    MLP preference predictor.

    Args:
        input_dim:   Dimension of the input feature vector.
        hidden_dims: Sequence of hidden layer sizes.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        input_dim:   int   = INPUT_DIM,
        hidden_dims: tuple = (128, 64, 32),
        dropout:     float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size,) — preference scores in [0, 1]
        """
        return torch.sigmoid(self.net(x)).squeeze(-1)


class TemporalContext:
    """
    Maintains a rolling buffer of the last T tracks and returns
    their mean feature vector as a context embedding.
    """

    def __init__(self, window: int = 10, feature_dim: int = AUDIO_DIM):
        self.window      = window
        self.feature_dim = feature_dim
        self.buffer      = []   # list of np.ndarray

    def update(self, feature_vec: np.ndarray) -> None:
        self.buffer.append(feature_vec.copy())
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

    def get_context(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros(self.feature_dim, dtype=np.float32)
        return np.mean(self.buffer, axis=0).astype(np.float32)

    def reset(self) -> None:
        self.buffer.clear()

class PreferenceTrainer:
    """
    Full training loop with train/val split, early stopping, and checkpoint saving.
    """

    def __init__(
        self,
        model:        PreferenceNet,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        device:       Optional[torch.device] = None,
        checkpoint_dir: Path = Path("checkpoints"),
    ):
        self.model   = model
        self.device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimiser = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode="min", patience=5, factor=0.5
        )
        self.criterion = nn.BCELoss()

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.history        = {"train_loss": [], "val_loss": [], "val_acc": []}

    def fit(
        self,
        dataset:      PreferenceDataset,
        epochs:       int   = 50,
        batch_size:   int   = 64,
        val_split:    float = 0.15,
        patience:     int   = 10,
    ) -> dict:
        """
        Train with early stopping. Returns history dict.
        """
        n_val   = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

        best_val_loss  = float("inf")
        patience_count = 0
        best_state     = None

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_acc = self._eval_epoch(val_loader)

            self.scheduler.step(val_loss)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.3f}"
                )

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                best_state     = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.save_checkpoint("best_model.pt")
            else:
                patience_count += 1
                if patience_count >= patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        # Restore best weights
        if best_state:
            self.model.load_state_dict(best_state)

        return self.history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimiser.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()
            total_loss += loss.item() * len(y)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            total_loss += self.criterion(pred, y).item() * len(y)
            correct    += ((pred > 0.5) == y.bool()).sum().item()
        n = len(loader.dataset)
        return total_loss / n, correct / n

    
    @torch.no_grad()
    def predict(self, feature_vec: np.ndarray) -> float:
        """Return preference score ∈ [0, 1] for a single track."""
        self.model.eval()
        x = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        return float(self.model(x).item())


    def save_checkpoint(self, name: str = "checkpoint.pt") -> Path:
        path = self.checkpoint_dir / name
        torch.save({
            "model_state": self.model.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "history": self.history,
        }, path)
        return path

    def load_checkpoint(self, name: str = "best_model.pt") -> None:
        path = self.checkpoint_dir / name
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", self.history)
        logger.info(f"Loaded checkpoint from {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    torch.manual_seed(42)

    print("=== PreferenceNet Smoke Test ===\n")

    rng = np.random.default_rng(42)
    n   = 1_000
    vecs   = rng.random((n, AUDIO_DIM)).astype(np.float32)
    ctx    = rng.random((n, CONTEXT_DIM)).astype(np.float32)
    inputs = np.concatenate([vecs, ctx], axis=1)
    labels = (rng.random(n) > 0.5).astype(int)

    df = pd.DataFrame({
        "feature_vec": [list(np.concatenate([v, c])) for v, c in zip(vecs, ctx)],
        "label":       labels,
    })

    ds      = PreferenceDataset(df, use_context=False)
    model   = PreferenceNet(input_dim=INPUT_DIM)
    trainer = PreferenceTrainer(model, lr=1e-3, checkpoint_dir=Path("checkpoints"))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset size:     {len(ds)} samples\n")

    history = trainer.fit(ds, epochs=20, batch_size=64, val_split=0.2, patience=5)
    final_acc = history["val_acc"][-1]
    print(f"\nFinal val accuracy: {final_acc:.3f}")

    sample_vec = np.random.rand(INPUT_DIM).astype(np.float32)
    score      = trainer.predict(sample_vec)
    print(f"Preference score for random track: {score:.4f}")

"""
Temporal Progression Model — Research Sandbox Prototype
=========================================================
Predicts 6-month PD symptom progression risk from longitudinal
voice recordings using a Transformer-based architecture.

DISCLAIMER:
"This model is a research prototype requiring longitudinal datasets. 
It is intended for research and screening support purposes only and is 
not a diagnostic medical device."

Architecture:
  Feature Encoder → Positional Encoding → Multi-Head Self-Attention
  → Feed-Forward → Pooling → Regression / Classification Head

Input:  Sequence of clinical feature vectors over time (T, 22)
Output: Progression risk score [0, 1] + trend direction
"""

import logging
import math
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_DIM = 22  # Number of clinical voice features


# ─── Positional Encoding ──────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─── Transformer Progression Model ───────────────────────────────────────────

class VoiceProgressionTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 50,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_pool = nn.Linear(d_model, 1)

        self.progression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.trend_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        attn_weights = torch.softmax(self.attn_pool(h), dim=1)
        context = (h * attn_weights).sum(dim=1)

        risk = self.progression_head(context)
        trend = self.trend_head(context)

        return risk, trend


# ─── Synthetic Longitudinal Dataset ──────────────────────────────────────────

class LongitudinalDataset(Dataset):
    def __init__(
        self, n_patients: int = 500,
        min_seq_len: int = 3, max_seq_len: int = 12,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.sequences = []
        self.progression_risks = []
        self.trend_labels = []

        n_pd = int(n_patients * 0.7)
        n_healthy = n_patients - n_pd

        for i in range(n_pd):
            seq_len = self.rng.integers(min_seq_len, max_seq_len + 1)
            seq = self._generate_pd_sequence(seq_len)
            self.sequences.append(seq)
            risk = float(self.rng.uniform(0.45, 0.95))
            self.progression_risks.append(risk)
            self.trend_labels.append(2 if risk > 0.65 else 1)

        for i in range(n_healthy):
            seq_len = self.rng.integers(min_seq_len, max_seq_len + 1)
            seq = self._generate_healthy_sequence(seq_len)
            self.sequences.append(seq)
            risk = float(self.rng.uniform(0.0, 0.25))
            self.progression_risks.append(risk)
            self.trend_labels.append(0 if risk < 0.1 else 1)

        self.max_seq_len = max_seq_len

    def _base_pd_features(self) -> np.ndarray:
        return np.array([
            self.rng.normal(154, 15),    # fo_mean
            self.rng.normal(180, 20),    # fo_max
            self.rng.normal(120, 15),    # fo_min
            self.rng.exponential(0.008), # jitter_local
            self.rng.exponential(5e-5),  # jitter_abs
            self.rng.exponential(0.004), # jitter_rap
            self.rng.exponential(0.004), # jitter_ppq5
            self.rng.exponential(0.012), # jitter_ddp
            self.rng.exponential(0.035), # shimmer_local
            self.rng.exponential(0.30),  # shimmer_db
            self.rng.exponential(0.018), # shimmer_apq3
            self.rng.exponential(0.020), # shimmer_apq5
            self.rng.exponential(0.030), # shimmer_apq11
            self.rng.exponential(0.055), # shimmer_dda
            self.rng.exponential(0.030), # nhr
            self.rng.normal(20.0, 4.0),  # hnr
            self.rng.normal(0.50, 0.07), # rpde
            self.rng.normal(0.72, 0.05), # dfa
            self.rng.normal(-5.6, 0.7),  # spread1
            self.rng.normal(0.23, 0.04), # spread2
            self.rng.normal(2.4, 0.25),  # d2
            self.rng.normal(0.21, 0.04), # ppe
        ], dtype=np.float32)

    def _generate_pd_sequence(self, seq_len: int) -> np.ndarray:
        base = self._base_pd_features()
        seq = []
        drift_rate = self.rng.uniform(0.01, 0.05)
        for t in range(seq_len):
            noise = self.rng.normal(0, 0.02, size=base.shape)
            drift = np.zeros_like(base)
            drift[3:8] += drift_rate * t * 0.5
            drift[8:14] += drift_rate * t * 0.3
            drift[16] += drift_rate * t * 0.2
            drift[21] += drift_rate * t * 0.2
            drift[15] -= drift_rate * t * 0.3
            seq.append(base + drift + noise)
        return np.stack(seq, axis=0)

    def _generate_healthy_sequence(self, seq_len: int) -> np.ndarray:
        base = np.array([
            self.rng.normal(188, 12), self.rng.normal(220, 15),
            self.rng.normal(155, 12), self.rng.exponential(0.003),
            self.rng.exponential(2e-5), self.rng.exponential(0.0015),
            self.rng.exponential(0.0015), self.rng.exponential(0.0045),
            self.rng.exponential(0.015), self.rng.exponential(0.15),
            self.rng.exponential(0.008), self.rng.exponential(0.01),
            self.rng.exponential(0.015), self.rng.exponential(0.024),
            self.rng.exponential(0.01), self.rng.normal(27.0, 3.0),
            self.rng.normal(0.43, 0.05), self.rng.normal(0.67, 0.04),
            self.rng.normal(-6.8, 0.5), self.rng.normal(0.14, 0.03),
            self.rng.normal(2.0, 0.2), self.rng.normal(0.095, 0.025),
        ], dtype=np.float32)

        seq = []
        for _ in range(seq_len):
            noise = self.rng.normal(0, 0.01, size=base.shape)
            seq.append(base + noise)
        return np.stack(seq, axis=0)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        risk = self.progression_risks[idx]
        trend = self.trend_labels[idx]

        T = seq.shape[0]
        pad_len = self.max_seq_len - T
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)

        if pad_len > 0:
            padding = np.zeros((pad_len, FEATURE_DIM), dtype=np.float32)
            seq = np.concatenate([seq, padding], axis=0)
            mask[T:] = True

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(risk, dtype=torch.float32),
            torch.tensor(trend, dtype=torch.long),
            mask,
        )


# ─── Training ─────────────────────────────────────────────────────────────────

def train_temporal_model(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Dict:
    logger.info("=" * 60)
    logger.info(f"Training Temporal Progression Transformer | Device: {DEVICE}")

    train_ds = LongitudinalDataset(n_patients=600, seed=42)
    val_ds = LongitudinalDataset(n_patients=150, seed=99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VoiceProgressionTransformer(
        feature_dim=FEATURE_DIM,
        d_model=128,
        n_heads=4,
        n_layers=3,
        dim_ff=256,
        dropout=0.2,
        max_seq_len=train_ds.max_seq_len,
    ).to(DEVICE)

    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    regression_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for seq, risk, trend, mask in train_loader:
            seq, risk, trend, mask = seq.to(DEVICE), risk.to(DEVICE), trend.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            risk_pred, trend_pred = model(seq, src_key_padding_mask=mask)

            loss_reg = regression_criterion(risk_pred.squeeze(), risk)
            loss_cls = classification_criterion(trend_pred, trend)
            loss = loss_reg + 0.3 * loss_cls
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_maes = []
        with torch.no_grad():
            for seq, risk, trend, mask in val_loader:
                seq, risk, trend, mask = seq.to(DEVICE), risk.to(DEVICE), trend.to(DEVICE), mask.to(DEVICE)
                risk_pred, trend_pred = model(seq, src_key_padding_mask=mask)
                loss = regression_criterion(risk_pred.squeeze(), risk)
                val_loss += loss.item()
                val_maes.extend(torch.abs(risk_pred.squeeze() - risk).cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_mae = np.mean(val_maes)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f} | "
                f"Time: {time.time()-t0:.1f}s"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_DIR, "temporal_progression.pt")
            )

    logger.info(f"  ✅ Temporal model saved | Best Val Loss: {best_val_loss:.4f}")

    config = {
        "feature_dim": FEATURE_DIM,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 3,
        "dim_ff": 256,
        "dropout": 0.2,
        "max_seq_len": train_ds.max_seq_len,
    }
    with open(os.path.join(MODEL_DIR, "temporal_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return {"best_val_loss": best_val_loss}


def predict_progression(
    feature_sequence: List[np.ndarray],
    model: VoiceProgressionTransformer = None,
) -> Dict:
    T = len(feature_sequence)
    if T < 3:
        return {
            "error": "Sequence too short. Progression forecasting requires at least 3 longitudinal sessions.",
            "status": "disabled",
            "disclaimer": "Research prototype requiring longitudinal datasets."
        }

    if model is None:
        config_path = os.path.join(MODEL_DIR, "temporal_config.json")
        weights_path = os.path.join(MODEL_DIR, "temporal_progression.pt")
        if not os.path.exists(weights_path):
            return {
                "error": "Temporal model not trained yet.",
                "disclaimer": "Research prototype requiring longitudinal datasets."
            }

        with open(config_path) as f:
            config = json.load(f)

        model = VoiceProgressionTransformer(**config).to(DEVICE)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    model.eval()
    max_len = 50
    seq = np.stack(feature_sequence, axis=0)[:max_len]
    pad_len = max_len - T

    if pad_len > 0:
        pad = np.zeros((pad_len, FEATURE_DIM), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)

    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[T:] = True

    with torch.no_grad():
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        m = mask.unsqueeze(0).to(DEVICE)
        risk, trend_logits = model(x, src_key_padding_mask=m)

    risk_score = float(risk.squeeze().cpu())
    trend_idx = int(torch.argmax(trend_logits, dim=1).cpu())
    trend_label = ["Improving", "Stable", "Worsening"][trend_idx]
    trend_conf = float(torch.softmax(trend_logits, dim=1).squeeze()[trend_idx].cpu())

    return {
        "progression_risk": round(risk_score, 4),
        "trend": trend_label,
        "trend_confidence": round(trend_conf, 4),
        "n_recordings_used": T,
        "disclaimer": "Research prototype requiring longitudinal datasets."
    }


if __name__ == "__main__":
    train_temporal_model(epochs=50, batch_size=32, lr=1e-3)

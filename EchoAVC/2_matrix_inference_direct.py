import os
import glob
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# ===========================
# CONFIG
# ===========================
EMB_T = 30
EMB_D = 774


# ===========================
# Utils
# ===========================
def log(x):
    print(x, flush=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


# ===========================
# Matrix listing
# ===========================
def list_study_matrices_flexible(root_dir, only_matrix1=False):
    """
    Supports both structures:

    A) root/study/matrix*.npy
    B) root/period/study/matrix*.npy

    Returns a DataFrame with columns:
      period, study, study_key, matrix_path, matrix_name
    """
    rows = []

    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"Directory does not exist: {root_dir}")

    level1 = sorted(os.listdir(root_dir))
    for name1 in level1:
        path1 = os.path.join(root_dir, name1)
        if not os.path.isdir(path1):
            continue

        mats1 = sorted(glob.glob(os.path.join(path1, "matrix*.npy")))
        if only_matrix1:
            mats1 = [m for m in mats1 if os.path.basename(m) == "matrix001.npy" or os.path.basename(m) == "matrix1.npy"]

        # Case A: root/study/matrix*.npy
        if mats1:
            for m in mats1:
                rows.append({
                    "period": "",
                    "study": name1,
                    "study_key": name1,
                    "matrix_path": m,
                    "matrix_name": os.path.basename(m),
                })
            continue

        # Case B: root/period/study/matrix*.npy
        for name2 in sorted(os.listdir(path1)):
            path2 = os.path.join(path1, name2)
            if not os.path.isdir(path2):
                continue

            mats2 = sorted(glob.glob(os.path.join(path2, "matrix*.npy")))
            if only_matrix1:
                mats2 = [m for m in mats2 if os.path.basename(m) == "matrix001.npy" or os.path.basename(m) == "matrix1.npy"]

            for m in mats2:
                rows.append({
                    "period": name1,
                    "study": name2,
                    "study_key": f"{name1}\\{name2}",
                    "matrix_path": m,
                    "matrix_name": os.path.basename(m),
                })

    df = pd.DataFrame(rows, columns=["period", "study", "study_key", "matrix_path", "matrix_name"])
    return df


# ===========================
# Dataset
# ===========================
class MatrixInferenceDataset(Dataset):
    def __init__(self, df, expected_shape=(EMB_T, EMB_D)):
        self.df = df.reset_index(drop=True)
        self.expected_shape = expected_shape

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = row["matrix_path"]
        x = np.load(path).astype(np.float32)

        if x.shape != self.expected_shape:
            raise ValueError(
                f"Unexpected shape in {path}: {x.shape}, expected {self.expected_shape}"
            )

        return (
            torch.from_numpy(x),
            row["study"],
            row["study_key"],
            row["matrix_path"],
            row["matrix_name"],
        )


# ===========================
# Model
# ===========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=EMB_D, max_len=EMB_T):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_term = div_term[:pe[:, 1::2].shape[1]]
        pe[:, 1::2] = torch.cos(position * cos_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        t = x.size(1)
        return x + self.pe[:t].unsqueeze(0)


class FrameDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if (not self.training) or self.p <= 0:
            return x
        b, t, d = x.shape
        mask = torch.bernoulli(torch.full((b, t, 1), 1 - self.p, device=x.device))
        return x * mask


class AttentionPool1D(nn.Module):
    def __init__(self, d_model=EMB_D, hidden=128, dropout=0.0):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, h):
        scores = self.attn(h).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        z = torch.bmm(weights.unsqueeze(1), h)
        return z.squeeze(1), weights


class TinyTemporalTransformer(nn.Module):
    def __init__(self, d_in=EMB_D, nhead=2, n_layers=4, ff=768, dropout=0.3, use_posenc=True, frame_dropout=0.1):
        super().__init__()
        self.posenc = PositionalEncoding(d_model=d_in, max_len=EMB_T) if use_posenc else nn.Identity()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_in,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.frame_drop = FrameDropout(p=frame_dropout)
        self.norm = nn.LayerNorm(d_in)

    def forward(self, x):
        h = self.posenc(x)
        h = self.frame_drop(h)
        h = self.enc(h)
        h = self.norm(h)
        return h


class ModelDiagnostic(nn.Module):
    def __init__(
        self,
        d_in=EMB_D,
        nhead=2,
        n_layers=4,
        ff=512,
        dropout=0.3,
        frame_dropout=0.1,
        attn_hidden=64,
        head_hidden=64,
        head_dropout=0.4,
    ):
        super().__init__()
        self.backbone = TinyTemporalTransformer(
            d_in=d_in,
            nhead=nhead,
            n_layers=n_layers,
            ff=ff,
            dropout=dropout,
            use_posenc=True,
            frame_dropout=frame_dropout,
        )
        self.pool = AttentionPool1D(d_model=d_in, hidden=attn_hidden, dropout=dropout)

        self.h_reg = nn.Sequential(
            nn.Linear(d_in, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )
        self.h_b0 = nn.Sequential(
            nn.Linear(d_in, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        h = self.backbone(x)
        z, _ = self.pool(h)
        y_reg = self.h_reg(z).squeeze(-1)
        y_b0 = self.h_b0(z).squeeze(-1)
        return y_reg, y_b0


# ===========================
# Inference
# ===========================
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()

    rows = []

    for X, study, study_key, matrix_path, matrix_name in tqdm(loader, total=len(loader), desc="Inference"):
        X = X.to(device)

        o_reg, o_b0 = model(X)

        pred_reg = o_reg.detach().cpu().numpy()
        pred_b0 = sigmoid_np(o_b0.detach().cpu().numpy())

        batch_size = len(pred_reg)
        for i in range(batch_size):
            rows.append({
                "study": study[i],
                "study_key": study_key[i],
                "matrix_path": matrix_path[i],
                "matrix_name": matrix_name[i],
                "EchoAVC_PRES": float(pred_b0[i]),
                "EchoAVC": float(max(0.0, pred_reg[i] * 5000.0)),
            })

    return pd.DataFrame(rows)


def aggregate_by_study(df_pred):
    if df_pred.empty:
        return pd.DataFrame(columns=["study", "EchoAVC_PRES", "EchoAVC", "n_matrices"])

    df_out = (
        df_pred.groupby("study", as_index=False)
        .agg(
            EchoAVC_PRES=("EchoAVC_PRES", "mean"),
            EchoAVC=("EchoAVC", "mean"),
            n_matrices=("matrix_path", "count"),
        )
    )

    df_out["EchoAVC_PRES"] = df_out["EchoAVC_PRES"].round(6)
    df_out["EchoAVC"] = df_out["EchoAVC"].round(2)

    return df_out


# ===========================
# Main
# ===========================
def main():
    out_csv = r'EchoAVC/results/EchoAVC_predictions.csv'
    model_path = r'EchoAVC/results/aggregator_model.pt'
    matrix_root = r'EchoAVC/matrix_out'
    seed = 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    mats_df = list_study_matrices_flexible(matrix_root, only_matrix1=False)

    if mats_df.empty:
        raise RuntimeError(f"No matrices were found in {matrix_root}")

    log(f"Matrices found: {len(mats_df)}")
    log(f"Unique studies: {mats_df['study'].nunique()}")

    ds = MatrixInferenceDataset(mats_df, expected_shape=(EMB_T, EMB_D))
    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    model = ModelDiagnostic(d_in=EMB_D).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    df_pred_matrix = run_inference(model, loader, device)
    df_pred_study = aggregate_by_study(df_pred_matrix)

    if out_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("results_inference", ts)
        ensure_dir(out_dir)
        out_csv = os.path.join(out_dir, "EchoAVC_predictions.csv")
        out_csv_matrix = os.path.join(out_dir, "EchoAVC_predictions_by_matrix.csv")
    else:
        out_csv = out_csv
        out_dir = os.path.dirname(out_csv) or "."
        ensure_dir(out_dir)
        base, ext = os.path.splitext(out_csv)
        out_csv_matrix = f"{base}_by_matrix{ext if ext else '.csv'}"

    df_pred_study.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_pred_matrix.to_csv(out_csv_matrix, index=False, encoding="utf-8-sig")

    log(f"Study-level CSV saved to: {out_csv}")
    log(f"Matrix-level CSV saved to: {out_csv_matrix}")
    log("Done.")


if __name__ == "__main__":
    main()

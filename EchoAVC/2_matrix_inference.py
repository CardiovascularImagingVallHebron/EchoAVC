import os, re, glob, argparse, random
from datetime import datetime
from collections import defaultdict
from xml.parsers.expat import model
from tqdm.auto import tqdm
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, r2_score
)


ROOT_MATRIX = r"data\matrix_pretrain"

CSV_RESULTS = r"content\info_csv.csv"

EMB_T = 30
EMB_D = 774

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def log(x): print(x, flush=True)

def list_study_matrices_bicator(root_dir, nmatrix=None):
    """
    Escanea ROOT_MATRIX (BICATOR): <period>\<study>\matrix*.npy
    study_key = "<period>\\<study>"
    """
    rows = []
    for period in sorted(os.listdir(root_dir)):
        p_dir = os.path.join(root_dir, period)
        if not os.path.isdir(p_dir): 
            continue
        for study in sorted(os.listdir(p_dir)):
            s_dir = os.path.join(p_dir, study)
            if not os.path.isdir(s_dir): 
                continue
            if nmatrix == 1:
                mats = sorted(glob.glob(os.path.join(s_dir, "matrix1.npy")))
            else:
                mats = sorted(glob.glob(os.path.join(s_dir, "matrix*.npy")))
            # mats = sorted(glob.glob(os.path.join(s_dir, "matrix1.npy")))
            # mats = sorted(glob.glob(os.path.join(s_dir, "matrix[12345].npy")))
            for m in mats:
                rows.append({
                    "period": period,
                    "study": study,
                    "study_key": f"{period}\\{study}",   # clave compuesta
                    "matrix_path": m,
                    "matrix_name": os.path.basename(m),
                    "source": "BICATOR",
                })
    return pd.DataFrame(rows)

def list_study_matrices_saltire(root_dir, nmatrix=None):
    """
    Escanea ROOT_MATRIX_SALT (SALTIRE): <study>\matrix*.npy
    study_key = "<study>"   (solo el paciente/estudio)
    """
    rows = []
    if not os.path.isdir(root_dir):
        return pd.DataFrame(rows)
    for study in sorted(os.listdir(root_dir)):
        s_dir = os.path.join(root_dir, study)
        if not os.path.isdir(s_dir):
            continue
        if nmatrix == 1:
            mats = sorted(glob.glob(os.path.join(s_dir, "matrix1.npy")))
        else:
            mats = sorted(glob.glob(os.path.join(s_dir, "matrix*.npy")))
        # mats = sorted(glob.glob(os.path.join(s_dir, "matrix1.npy")))
        # mats = sorted(glob.glob(os.path.join(s_dir, "matrix[12345].npy")))
        for m in mats:
            rows.append({
                "period": "SALTIRE",                  # marcamos el “period” con etiqueta fija
                "study": study,
                "study_key": f"{study}",              # clave = solo estudio (sin period)
                "matrix_path": m,
                "matrix_name": os.path.basename(m),
                "source": "SALTIRE",
            })
    return pd.DataFrame(rows)

def list_all_matrices(root):
    df_b = list_study_matrices_bicator(root)
    if df_b.empty:
        return pd.DataFrame(columns=["period","study","study_key","matrix_path","matrix_name","source"])
    return df_b

def list_study_matrices(root_dir):
    """
    Devuelve DataFrame con columnas:
    period, study, study_key, matrix_path, matrix_name
    Donde study_key = period\study (clave para cruce con 'study' de CSVs)
    """
    rows = []
    for period in sorted(os.listdir(root_dir)):
        p_dir = os.path.join(root_dir, period)
        if not os.path.isdir(p_dir): continue
        for study in sorted(os.listdir(p_dir)):
            s_dir = os.path.join(p_dir, study)
            if not os.path.isdir(s_dir): continue
            mats = sorted(glob.glob(os.path.join(s_dir, "matrix*.npy")))
            for m in mats:
                rows.append({
                    "period": period,
                    "study": study,
                    "study_key": f"{period}\\{study}",
                    "matrix_path": m,
                    "matrix_name": os.path.basename(m)
                })
    return pd.DataFrame(rows)

def ensure_dir(path): os.makedirs(path, exist_ok=True)

def bce_loss_from_logits(logits, targets, pos_weight=None):
    if pos_weight is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

def accuracy_from_logits_binary(logits, targets, thresh=0.5):
    preds = (torch.sigmoid(logits) > thresh).float()
    return (preds == targets).float().mean().item()

def cm_and_acc_binary(logits_np, targets_np, thresh=0.5):
    preds = (1/(1+np.exp(-logits_np)) > thresh).astype(int)
    cm = confusion_matrix(targets_np.astype(int), preds)
    acc = accuracy_score(targets_np.astype(int), preds)
    return cm, acc

def cm_and_acc_multiclass(logits_np, targets_np):
    preds = logits_np.argmax(axis=1)
    cm = confusion_matrix(targets_np.astype(int), preds)
    acc = accuracy_score(targets_np.astype(int), preds)
    return cm, acc

def plot_cm(cm, labels, title, save_path):
    acc = np.trace(cm) / np.sum(cm)
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_percent = np.nan_to_num(cm_percent) * 100
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=f"{title}\nAccuracy: {acc:.2%}",
           ylabel='True', xlabel='Pred')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_percent[i, j] > 50 else "black"
            txt = f"{cm_percent[i,j]:.1f}%\n({cm[i,j]})"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9, fontweight="bold")
    fig.tight_layout(); plt.savefig(save_path); plt.close()

# ---------------------------
# Dataset
# ---------------------------
class MatrixDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        X = np.load(row["matrix_path"]).astype(np.float32)  # (30,768)
        # Targets
        y_reg = row["y_reg"]
        y_b0 = row["y_bin0"]
        study_key = row["study_key"]  # <<--- NUEVO

        return (torch.from_numpy(X),
                torch.tensor([y_reg], dtype=torch.float32).squeeze(0),
                torch.tensor([y_b0], dtype=torch.float32).squeeze(0),
                study_key)  # <<--- NUEVO

# ---------------------------

class ConvTemporalBackbone(nn.Module):
    """
    Entrada:  x (B, T, D)  con D=768
    Conv1d a lo largo del eje temporal (tras permutar a (B, D, T)).
    Sale un embedding fijo (B, d_model) usando global pooling, independiente de T.
    """
    def __init__(self, seq_len=30, d_in=768, d_model=128, hidden1=1024, hidden2=256, dropout=0.5):
        super().__init__()
        # Conv temporal: in_channels = features
        self.conv1 = nn.Conv1d(in_channels=d_in, out_channels=hidden1, kernel_size=2, padding='same')
        self.bn1   = nn.BatchNorm1d(hidden1)

        # Conv dilatada para aumentar receptive field temporal
        self.conv2 = nn.Conv1d(in_channels=hidden1, out_channels=hidden2, kernel_size=2, dilation=2, padding='same')
        self.bn2   = nn.BatchNorm1d(hidden2)

        self.flatten = nn.Flatten()

        # Proyección a d_model tras global pooling
        self.fc1 = nn.Linear(hidden2*seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))         # (B, H1, T)
        x = F.relu(self.bn2(self.conv2(x)))         # (B, H2, T)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return x

class PositionalEncoding(nn.Module):
    """Clásico sinusoidal; soporta d_model par o impar."""
    def __init__(self, d_model=EMB_D, max_len=EMB_T):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        # senos para 0,2,4,...
        pe[:, 0::2] = torch.sin(position * div_term)
        # cosenos para 1,3,5,... (recorta si d_model es impar)
        cos_term = div_term[:pe[:, 1::2].shape[1]]
        pe[:, 1::2] = torch.cos(position * cos_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class FrameDropout(nn.Module):
    """Apaga frames completos con prob p durante train (MIL-style regularization)."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        B, T, D = x.shape
        # Máscara por batch-frame (B, T, 1)
        mask = torch.bernoulli(torch.full((B, T, 1), 1 - self.p, device=x.device))
        return x * mask

class AttentionPool1D(nn.Module):
    """Atención escalar por frame -> pooling ponderado sobre T."""
    def __init__(self, d_model=EMB_D, hidden=128, dropout=0.0):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, h):  # h: (B, T, D)
        scores = self.attn(h).squeeze(-1)          # (B, T)
        weights = F.softmax(scores, dim=1)         # (B, T)
        z = torch.bmm(weights.unsqueeze(1), h)     # (B, 1, D)
        return z.squeeze(1), weights               # (B, D), (B, T)

# ---------------------------
# Backbone + Heads
# ---------------------------
class TinyTemporalTransformer(nn.Module):
    """
    TransformerEncoder muy ligero, d_model = D para no proyectar.
    """
    def __init__(self, d_in=EMB_D, nhead=2, n_layers=4, ff=768, dropout=0.3, use_posenc=True, frame_dropout=0.1):
        super().__init__()
        self.use_posenc = use_posenc
        self.posenc = PositionalEncoding(d_model=d_in, max_len=EMB_T) if use_posenc else nn.Identity()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_in, nhead=nhead, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.frame_drop = FrameDropout(p=frame_dropout)
        self.norm = nn.LayerNorm(d_in)

    def forward(self, x):  # x: (B, T, D)
        h = self.posenc(x)
        h = self.frame_drop(h)           # (B, T, D)
        h = self.enc(h)                  # (B, T, D)
        h = self.norm(h)                 # (B, T, D)
        return h

class ModelDiagnostic(nn.Module):
    """
    Modelo second-stage:
      - Input:  (B, T=30, D=768)
      - Output: y_reg (B,), y_b0 (B,)
    """
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
        head_dropout=0.4
    ):
        super().__init__()
        self.backbone = TinyTemporalTransformer(
            d_in=d_in, nhead=nhead, n_layers=n_layers, ff=ff,
            dropout=dropout, use_posenc=True, frame_dropout=frame_dropout
        )
        self.pool = AttentionPool1D(d_model=d_in, hidden=attn_hidden, dropout=dropout)

        # Dos heads: regresión y binaria (sigmoid en training loop)
        self.h_reg = nn.Sequential(
            nn.Linear(d_in, head_hidden), nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1)
        )
        self.h_b0 = nn.Sequential(
            nn.Linear(d_in, head_hidden), nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, x):
        """
        x: Tensor (B, T, D)
        returns: y_reg (B,), y_b0 (B,)
        """
        h = self.backbone(x)                     # (B, T, D)
        z, _ = self.pool(h)                      # (B, D)
        y_reg = self.h_reg(z).squeeze(-1)        # (B,)
        y_b0  = self.h_b0(z).squeeze(-1)         # (B,)
        return y_reg, y_b0


   
# ---------------------------
# Preparación de datos + splits
# ---------------------------
def build_dataframe_and_splits():
    # Inventario matrices
    # mats_df = list_study_matrices(ROOT_MATRIX)
    mats_df = list_all_matrices(ROOT_MATRIX)

    if mats_df.empty:
        raise RuntimeError("No se encontraron matrices en el árbol especificado.")

    # CSVs externos
    res_csv = pd.read_csv(CSV_RESULTS)       # debe tener columnas: 'study', 'split', 'AVCS_CT', 'AVCS'
    # Normaliza claves
    for c in ["study"]:
        if c not in res_csv.columns: raise ValueError("CSV de resultados debe tener columna 'study'")

    # Merge labels por study_key
    res_csv["study"] = res_csv["study"].astype(str)
    mats_df["study_key"] = mats_df["study_key"].astype(str)

    # Une labels a nivel estudio
    mats_df['study_key'] = mats_df['study_key'].str.lower()

    df = mats_df.merge(res_csv[["study","split","AVCS_CT"]], 
                       left_on="study_key", right_on="study", how="left")

    # Filtra estudios con labels existentes
    df = df[~df["AVCS_CT"].isna()].copy()

    # Targets
    df["y_reg"] = (df["AVCS_CT"] / 5000.0).clip(0, None)  # regresión escalada

    df["y_bin0"]    = (df["AVCS_CT"] > 0).astype(int)

    # Define splits:
    df["split_set"] = "train"  # por defecto


    # 2) test: en results con split==1 o 2
    test_mask = df["split"].isin([1,2])
    df.loc[(df["split_set"] != "val") & test_mask, "split_set"] = "test"

    # Report
    log(df["split_set"].value_counts())

    return df


@torch.no_grad()
def evaluate(model, loader, device, pos_weights=None, lambdas=None):
    """
    Evalúa métricas + (opcionalmente) loss de validación si se pasan pos_weights y lambdas.
    Devuelve: dict con r2, mean_auc, head_aucs, arrays de preds/targets y 'loss' (float o nan)
    """
    model.eval()

    y_true_reg, y_pred_reg = [], []
    YT_bin = {"b0":[]}
    YP_bin = {"b0":[]}
    study_keys = []

    tot_loss = 0.0
    n_batches = 0
    compute_loss = (pos_weights is not None) and (lambdas is not None)

    for X, y_reg, y_b0, sk in tqdm(loader, total=len(loader), desc="eval", leave=False):
        X = X.to(device)
        y_reg = y_reg.to(device); y_b0 = y_b0.to(device)

        o_reg, o_b0 = model(X)

        # Acumular para métricas
        y_true_reg.extend(y_reg.cpu().numpy().tolist())
        y_pred_reg.extend(o_reg.cpu().numpy().tolist())
        study_keys.extend(list(sk))
        for k, o, y in [("b0",o_b0,y_b0)]:
            YT_bin[k].extend(y.cpu().numpy().tolist())
            YP_bin[k].extend(torch.sigmoid(o).cpu().numpy().tolist())

        # (Opcional) calcular loss de validación con las mismas ponderaciones
        if compute_loss:
            l_reg = F.mse_loss(o_reg, y_reg)
            l_b0  = bce_loss_from_logits(o_b0, y_b0, pos_weights.get("b0"))
            loss = (lambdas["reg"]*l_reg + lambdas["b0"]*l_b0)
            loss = (lambdas["reg"]*l_reg + 
                    lambdas["b0"]*l_b0)
            tot_loss += loss.item()
            n_batches += 1

    r2 = r2_score(y_true_reg, y_pred_reg) if len(y_true_reg)>1 else float("nan")

    aucs = []
    head_aucs = {}
    for k in ["b0"]:
        yt = np.array(YT_bin[k]); yp = np.array(YP_bin[k])
        try: 
            auc = roc_auc_score(yt, yp); head_aucs[k]=auc; aucs.append(auc)
        except: 
            head_aucs[k]=np.nan
   
    mean_auc = np.nanmean(aucs) if len(aucs)>0 else float("nan")

    val_loss = (tot_loss / max(1, n_batches)) if compute_loss else float("nan")

    return {
        "r2": r2,
        "mean_auc": mean_auc,
        "head_aucs": head_aucs,
        "y_reg_true": np.array(y_true_reg),
        "y_reg_pred": np.array(y_pred_reg),
        "YT_bin": {k: np.array(v) for k,v in YT_bin.items()},
        "YP_bin": {k: np.array(v) for k,v in YP_bin.items()},
        "study_keys": np.array(study_keys),
        "loss": val_loss,  # <<--- NUEVO
    }



# ---------------------------
# Plots requeridos
# ---------------------------
def plot_training_curves(history, out_dir):
    # Loss
    plt.figure()
    plt.plot(history["loss_tr"], label="Train")
    plt.plot(history["loss_va"], label="Valid")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (Train vs Valid)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves_loss.png")); plt.close()

    # AUROC
    plt.figure()
    plt.plot(history["auc_tr"], label="Train mean AUC")
    plt.plot(history["auc_va"], label="Valid mean AUC")
    plt.xlabel("Epoch"); plt.ylabel("Mean AUROC"); plt.title("AUROC (Train vs Valid)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves_auc.png")); plt.close()

    # R2
    plt.figure()
    plt.plot(history["r2_tr"], label="Train R2")
    plt.plot(history["r2_va"], label="Valid R2")
    plt.xlabel("Epoch"); plt.ylabel("R2"); plt.title("R2 (Train vs Valid)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves_r2.png")); plt.close()

def scatter_reg(y_true, y_pred, title, save_path):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'k--', linewidth=1)
    r2 = r2_score(y_true, y_pred) if len(y_true)>1 else float("nan")
    plt.title(f"{title} (R2={r2:.3f})"); plt.xlabel("True (AVCS_CT/5000)"); plt.ylabel("Pred")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def boxplot_binary(yt, yp, title, save_path):
    dfp = pd.DataFrame({"label": yt.astype(int), "score": yp})
    plt.figure(figsize=(5,4))
    sns.boxplot(data=dfp, x="label", y="score")
    try: auc = roc_auc_score(yt, yp)
    except: auc = float("nan")
    plt.title(f"{title}  (AUROC={auc:.3f})")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def boxplot_multiclass(yt, yp_prob, title, save_path):
    # mostremos distribución de prob. de la clase verdadera
    true_prob = yp_prob[np.arange(len(yt)), yt.astype(int)]
    dfp = pd.DataFrame({"true_class": yt.astype(int), "prob_true": true_prob})
    plt.figure(figsize=(5,4))
    sns.boxplot(data=dfp, x="true_class", y="prob_true")
    try: auc = roc_auc_score(yt, yp_prob, multi_class="ovr", average="macro")
    except: auc = float("nan")
    plt.title(f"{title}  (AUROC={auc:.3f})")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def aggregate_test_by_study(eval_dict):
    sk = eval_dict["study_keys"]
    # --- Regresión ---
    df_reg = pd.DataFrame({
        "study": sk,
        "y_true": eval_dict["y_reg_true"],
        "y_pred": eval_dict["y_reg_pred"],
    })
    g_reg = df_reg.groupby("study", as_index=False).mean()

    # --- Binarios ---
    bin_agg = {}
    for k in ["b0"]:
        dfb = pd.DataFrame({"study": sk, "y": eval_dict["YT_bin"][k], "p": eval_dict["YP_bin"][k]})
        gb  = dfb.groupby("study", as_index=False).mean()  # y se mantiene (es constante por study), p se promedia
        bin_agg[k] = gb

    # --- Multiclase (promedio de probabilidades por clase) ---
    mc_agg = {}

    return g_reg, bin_agg, mc_agg

def plot_test_grouped_by_study(test_eval, out_dir):
    # Agrega
    g_reg, bin_agg, mc_agg = aggregate_test_by_study(test_eval)

    # --- Scatter regresión por study ---
    scatter_reg(g_reg["y_true"].values, g_reg["y_pred"].values,
                "Regression TEST (by study)",
                os.path.join(out_dir, "scatter_reg_test_by_study.png"))

    # --- Binarios: boxplot + CM (por study) ---
    for k, nice in [("b0", "CT>0")]:
        gb = bin_agg[k]

        # Boxplot por clase con AUROC en título
        boxplot_binary(gb["y"].values, gb["p"].values,
                       f"TEST Binaria {nice} (by study)",
                       os.path.join(out_dir, f"box_test_bin_{k}_by_study.png"))

        # CM (umbral 0.5)
        logits = np.log(gb["p"].values / np.clip(1 - gb["p"].values, 1e-8, 1))
        cm, acc = cm_and_acc_binary(logits, gb["y"].values)
        plot_cm(cm, [0, 1],
                f"TEST CM Binaria {nice} (by study)",
                os.path.join(out_dir, f"cm_test_bin_{k}_by_study.png"))

        # ---------- Scatter extra: escala ×5000, y_pred=0 si p<0.5, recta roja y MAE ----------
        # Aseguramos el merge por 'study' para alinear las filas
        cols_needed = ["study", "y_true", "y_pred"]
        if not all(c in g_reg.columns for c in cols_needed):
            raise ValueError(f"g_reg debe contener columnas {cols_needed}")

        if "study" not in gb.columns or "p" not in gb.columns:
            raise ValueError("gb debe contener columnas 'study' y 'p'")

        dfm = g_reg.merge(gb[["study", "p"]], on="study", how="left")
        y_true = dfm["y_true"].to_numpy().astype(float)
        y_pred = dfm["y_pred"].to_numpy().astype(float)
        p_b0   = dfm["p"].to_numpy().astype(float)

        # Ajuste: predicción a 0 si p(b0) < 0.5
        y_pred_adj = y_pred.copy()
        y_pred_adj[p_b0 < 0.5] = 0.0
        y_pred_adj = np.clip(y_pred_adj, 0, None)  # no negativo

        # Escala ×5000
        scale = 5000.0
        x = y_true * scale
        y = y_pred_adj * scale

        # MAE
        mae = float(np.mean(np.abs(x - y)))
        r2 = r2_score(x, y) if len(y) > 1 else float("nan")
        r = float(np.corrcoef(x, y)[0, 1])
        # Recta de regresión (y = a*x + b)
        if x.size >= 2:
            a, b = np.polyfit(x, y, 1)
        else:
            a, b = 1.0, 0.0  # fallback estable

        # Plot
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(x, y, alpha=0.7)
        xline = np.linspace(np.min(x), np.max(x), 100) if x.size else np.array([0, 1])
        plt.plot(xline, a * xline + b, '-', color='red', linewidth=2, label=f'y={a:.2f}x+{b:.2f}')
        # (opcional) referencia y=x
        plt.plot(xline, xline, '--', linewidth=1, alpha=0.5, label='y = x')

        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.title(f"Regression TEST (by study)  |  MAE={mae:.0f} R2={r2:.3f} R={r:.3f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"scatter_reg_test_by_study_scaled_{k}_thr05.png"), dpi=150)
        plt.close()

def plot_grouped_by_study(eval_dict, out_dir, split_name="TEST"):
    # Agrega
    g_reg, bin_agg, mc_agg = aggregate_test_by_study(eval_dict)

    # --- Scatter regresión por study ---
    scatter_reg(
        g_reg["y_true"].values, g_reg["y_pred"].values,
        f"Regression {split_name} (by study)",
        os.path.join(out_dir, f"scatter_reg_{split_name.lower()}_by_study.png")
    )

    # --- Binarios: boxplot + CM (por study) ---
    for k, nice in [("b0", "CT>0")]:
        gb = bin_agg[k]

        # Boxplot por clase con AUROC en título
        boxplot_binary(
            gb["y"].values, gb["p"].values,
            f"{split_name} Binaria {nice} (by study)",
            os.path.join(out_dir, f"box_{split_name.lower()}_bin_{k}_by_study.png")
        )

        # CM (umbral 0.5)
        logits = np.log(gb["p"].values / np.clip(1 - gb["p"].values, 1e-8, 1))
        cm, acc = cm_and_acc_binary(logits, gb["y"].values)
        plot_cm(
            cm, [0, 1],
            f"{split_name} CM Binaria {nice} (by study)",
            os.path.join(out_dir, f"cm_{split_name.lower()}_bin_{k}_by_study.png")
        )

        # ---------- Scatter extra: escala ×5000, y_pred=0 si p<0.5, recta y MAE ----------
        cols_needed = ["study", "y_true", "y_pred"]
        if not all(c in g_reg.columns for c in cols_needed):
            raise ValueError(f"g_reg debe contener columnas {cols_needed}")
        if "study" not in gb.columns or "p" not in gb.columns:
            raise ValueError("gb debe contener columnas 'study' y 'p'")

        dfm = g_reg.merge(gb[["study", "p"]], on="study", how="left")
        y_true = dfm["y_true"].to_numpy().astype(float)
        y_pred = dfm["y_pred"].to_numpy().astype(float)
        p_b0   = dfm["p"].to_numpy().astype(float)

        # Ajuste: predicción a 0 si p(b0) < 0.5
        y_pred_adj = y_pred.copy()
        y_pred_adj[p_b0 < 0.5] = 0.0
        y_pred_adj = np.clip(y_pred_adj, 0, None)

        # Escala ×5000
        scale = 5000.0
        x = y_true * scale
        y = y_pred_adj * scale

        mae = float(np.mean(np.abs(x - y)))
        r2  = r2_score(x, y) if len(y) > 1 else float("nan")

        if x.size >= 2:
            a, b = np.polyfit(x, y, 1)
        else:
            a, b = 1.0, 0.0

        plt.figure()
        plt.scatter(x, y, alpha=0.7)
        xline = np.linspace(np.min(x), np.max(x), 100) if x.size else np.array([0, 1])
        plt.plot(xline, a * xline + b, '-', color='red', linewidth=2, label=f'y={a:.2f}x+{b:.2f}')
        plt.plot(xline, xline, '--', linewidth=1, alpha=0.5, label='y = x')
        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.title(f"Regression {split_name} (by study)  |  MAE={mae:.0f} R2={r2:.3f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"scatter_reg_{split_name.lower()}_by_study_scaled_{k}_thr05.png"), dpi=150)
        plt.close()

def summarize_optimizer(optimizer):
    """Convierte la config del optimizador en algo JSON-friendly."""
    try:
        pg = optimizer.param_groups  # lista de grupos
        # Nos quedamos con campos relevantes y serializables
        clean_groups = []
        for g in pg:
            cg = {k: g[k] for k in ["lr","weight_decay"] if k in g}
            # extras comunes en Adam/AdamW
            for k in ["betas","eps","amsgrad","momentum"]:
                if k in g:
                    cg[k] = g[k]
            clean_groups.append(cg)
        return {
            "optimizer_class": optimizer.__class__.__name__,
            "param_groups": clean_groups,
        }
    except Exception as e:
        return {"optimizer_class": optimizer.__class__.__name__, "error": str(e)}

def summarize_model(model: nn.Module):
    """Resumen legible del modelo y sus capas."""
    try:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # listado compacto de submódulos de primer nivel
        layers = []
        for name, mod in model.named_children():
            layers.append({
                "name": name,
                "class": mod.__class__.__name__,
                "repr": str(mod)[:1000]  # recorte por si es muy largo
            })
        return {
            "model_class": model.__class__.__name__,
            "n_parameters": n_params,
            "n_trainable": n_trainable,
            "top_layers": layers,
            "full_repr_head": str(model)[:2000]  # resumen
        }
    except Exception as e:
        return {"model_class": model.__class__.__name__, "error": str(e)}

def save_run_json(path, args, history, optimizer, model, best_epoch, best_score, files_used=None):
    import json
    payload = {
        "args": vars(args),
        "history": {k: list(map(float, v)) for k, v in history.items()},
        "best_epoch": int(best_epoch),
        "best_score_mean_auc_r2": float(best_score),
        "optimizer": summarize_optimizer(optimizer),
        "model": summarize_model(model),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    # <<< NUEVO: adjuntar ficheros usados >>>
    if files_used is not None:
        payload["files_used"] = files_used
        # (opcional) resumimos los conteos para consulta rápida
        try:
            payload["files_used_counts"] = {
                "matrices": {
                    "train": len(files_used.get("matrices", {}).get("train", [])),
                    "val":   len(files_used.get("matrices", {}).get("val", [])),
                    "test":  len(files_used.get("matrices", {}).get("test", [])),
                    "total": sum(len(files_used.get("matrices", {}).get(k, [])) for k in ["train","val","test"])
                }
            }
        except Exception:
            pass

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------
# Main runnable (ipynb/.py)
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", type=str, default="results")
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--lambda_b", type=float, default=0.5)
    parser.add_argument("--lambda_mc", type=float, default=1.0)
    args = parser.parse_args(args=[])  # <-- en ipynb; quita args=[] si lo corres como script

    set_seed(args.seed)
    out_dir = os.path.join(args.out_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    ensure_dir(out_dir)

    # Data + splits
    df = build_dataframe_and_splits()

    # Datasets
    df_test  = df[df["split_set"]=="test"].copy()
    
    # --- NUEVO: recopilar ficheros usados ---
    files_used = {
        "roots": {
            "ROOT_MATRIX": ROOT_MATRIX
        },
        "csvs": {
            "CSV_RESULTS": CSV_RESULTS
        },
        "matrices": {
            "test":  df_test["matrix_path"].astype(str).tolist(),
        }
    }

    test_ds  = MatrixDataset(df_test)

    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelDiagnostic(d_in=EMB_D).to(device)

    best_path = os.path.join(r'results\best_model.pt')

    # Carga best y eval en VAL y TEST para plots requeridos
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_eval = evaluate(model, tqdm(test_loader, total=len(test_loader), desc="TEST", leave=False), device)

    plot_test_grouped_by_study(test_eval, out_dir)
    # -------- Boxplots (clasificaciones) + AUROC en título --------
    # Binarias
    for key, nice in [("b0", "CT>0")]:
        boxplot_binary(test_eval["YT_bin"][key], test_eval["YP_bin"][key], f"TEST Binaria {nice}", os.path.join(out_dir, f"box_test_bin_{key}.png"))

        cm, acc = cm_and_acc_binary(
            logits_np=np.log(test_eval["YP_bin"][key] / np.clip(1-test_eval["YP_bin"][key],1e-8,1)), 
            targets_np=test_eval["YT_bin"][key]
        )
        plot_cm(cm, [0,1], f"TEST CM Binaria {nice}", os.path.join(out_dir, f"cm_test_bin_{key}.png"))

    # Guardar resumen
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\nTEST metrics:\n")
        f.write(f"  R2: {test_eval['r2']:.4f}\n  Mean AUC: {test_eval['mean_auc']:.4f}\n  Head AUCs: {test_eval['head_aucs']}\n")
   
    # ========= CSV de resultados por estudio =========
    # (usa medias por estudio; convierte regresión de [0,1] a la escala AVCS_CT)
    g_reg, bin_agg, _ = aggregate_test_by_study(test_eval)
    gb = bin_agg["b0"][["study", "p"]].rename(columns={"p": "EchoAVC_PRES"})

    df_csv = g_reg[["study", "y_true", "y_pred"]].copy()
    df_csv["AVCS_CT"] = (df_csv["y_true"] * 5000.0).round(2)   # valor real
    df_csv["EchoAVC"] = (df_csv["y_pred"] * 5000.0).clip(0, None).round(2)  # predicción continua

    # Une la probabilidad binaria por estudio (CT>0)
    df_csv = df_csv.merge(gb, on="study", how="left")

    # Reordena y guarda con los nombres solicitados
    df_csv = df_csv[["study", "AVCS_CT", "EchoAVC_PRES", "EchoAVC"]]
    csv_out = os.path.join(out_dir, "EchoAVC_predictions.csv")
    df_csv.to_csv(csv_out, index=False, encoding="utf-8-sig")
    log(f"💾 CSV guardado en: {csv_out}")

    log(f"\n✅ Listo. Resultados en: {out_dir}")

if __name__ == "__main__":
    main()

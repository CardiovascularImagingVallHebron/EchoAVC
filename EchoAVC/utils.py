import ast
import glob
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLIP_LEN = 32
CENTER_CROP_SIZE = 224
MIN_SIDE = 224
FRAME_STRIDE_SHORT = 2
FRAME_STRIDE_LONG = 5
FRAME_STRIDE_MATRIX = 5

QUAL_DIM = 3

EMB_T = 30
EMB_D = 774

VIEW_LABELS = {
    "PLAX": 0.0,
    "PSAX": 1.0,
    "3CH": 2.0,
}


class Task():
    """Echocardiography interpretation task object."""

    def __init__(self, task_name, task_type, class_names, mean=np.nan):
        self.task_name = task_name
        self.task_type = task_type
        self.class_names = class_names  # ndarray
        self.class_indices = np.arange(class_names.size)
        self.mean = mean


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


def merge_task_dicts(d):
    merged_dict = {}

    for dictionary in d:
        for key, value in dictionary.items():
            if key in merged_dict:
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key in merged_dict[key]:
                            merged_dict[key][sub_key] += sub_value
                        else:
                            merged_dict[key][sub_key] = sub_value
                else:
                    merged_dict[key] += value
            else:
                merged_dict[key] = value

    return merged_dict


def log(msg: str):
    print(msg, flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def ensure_min_size(img: np.ndarray, min_side: int = MIN_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    short = min(h, w)
    if short >= min_side:
        return img
    scale = min_side / short
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop(img: np.ndarray, size: int = CENTER_CROP_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    if h < size or w < size:
        raise ValueError(f"Cannot do a {size}x{size} center crop on image {h}x{w}")
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def load_video_frames(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ensure_min_size(frame, MIN_SIDE)
            frame = center_crop(frame, CENTER_CROP_SIZE)
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Video has no frames: {path}")
    return frames


def build_clip(frames: List[np.ndarray], start: int, clip_len: int = CLIP_LEN) -> np.ndarray:
    n = len(frames)
    idxs = [(start + i) % n for i in range(clip_len)]
    clip = np.stack([frames[i] for i in idxs], axis=0)
    clip = clip.astype(np.float32) / 255.0
    clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
    clip = np.transpose(clip, (3, 0, 1, 2))
    clip = np.expand_dims(clip, 0)
    return clip


def save_embedding(out_path: str, emb: np.ndarray):
    ensure_dir(os.path.dirname(out_path))
    np.save(out_path, emb)


def load_embedding(path: str) -> np.ndarray:
    x = np.load(path)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x


def parse_filename(path_or_name: str) -> Tuple[str, int]:
    """
    Extracts:
      - base video name
      - frame_idx
    from names like:
      foo_frame0005.npy
      foo_frame0010
      /a/b/foo_frame0005.npy
    If it does not find the _frameXXXX suffix => frame_idx = -1
    """
    name = os.path.basename(path_or_name)
    stem = os.path.splitext(name)[0]
    marker = "_frame"
    pos = stem.rfind(marker)
    if pos == -1:
        return stem, -1

    base = stem[:pos]
    suffix = stem[pos + len(marker):]
    if suffix.isdigit():
        return base, int(suffix)
    return stem, -1


def normalize_study(x) -> str:
    return str(x).strip()


def normalize_video_name(x) -> str:
    """Normalizes the video name to a basename without extension."""
    s = str(x).strip()
    s = os.path.basename(s)
    s = os.path.splitext(s)[0]
    return s


def canon_view(v) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip().upper()

    replacements = {
        "PARASTERNAL LONG AXIS": "PLAX",
        "LONG AXIS": "PLAX",
        "PSLAX": "PLAX",
        "PLA": "PLAX",
        "PARASTERNAL SHORT AXIS": "PSAX",
        "SHORT AXIS": "PSAX",
        "SAX": "PSAX",
        "3CH": "3CH",
        "APICAL 3 CHAMBER": "3CH",
        "APICAL 3-CHAMBER": "3CH",
        "A3C": "3CH",
    }

    if s in replacements:
        return replacements[s]

    if "PLAX" in s or ("LONG" in s and "AXIS" in s):
        return "PLAX"
    if "PSAX" in s or ("SHORT" in s and "AXIS" in s) or s == "SAX":
        return "PSAX"
    if "3CH" in s or "3 CH" in s or "A3C" in s:
        return "3CH"

    return None


def parse_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def parse_quality(x) -> np.ndarray:
    """
    Accepts:
    - list/string like "[0.1, 0.2, 0.7]"
    - tuple
    - np.ndarray
    - scalar -> padded with zeros except for the first value
    """
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float32).reshape(-1)
    elif isinstance(x, (list, tuple)):
        arr = np.array(x, dtype=np.float32).reshape(-1)
    elif isinstance(x, str):
        s = x.strip()
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple, np.ndarray)):
                arr = np.array(obj, dtype=np.float32).reshape(-1)
            else:
                arr = np.array([float(obj)], dtype=np.float32)
        except Exception:
            parts = [p for p in s.replace(";", ",").split(",") if p.strip()]
            try:
                arr = np.array([float(p) for p in parts], dtype=np.float32).reshape(-1)
            except Exception:
                arr = np.zeros(QUAL_DIM, dtype=np.float32)
    else:
        try:
            arr = np.array([float(x)], dtype=np.float32)
        except Exception:
            arr = np.zeros(QUAL_DIM, dtype=np.float32)

    if arr.size == QUAL_DIM:
        return arr.astype(np.float32)

    out = np.zeros(QUAL_DIM, dtype=np.float32)
    m = min(arr.size, QUAL_DIM)
    out[:m] = arr[:m]
    return out


def find_required_col(df: pd.DataFrame, candidates: List[str], csv_name: str) -> str:
    low_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low_map:
            return low_map[cand.lower()]
    raise ValueError(f"{csv_name} is missing one of these columns: {candidates}")


def load_metadata(csv_quality: str, csv_view: str) -> pd.DataFrame:
    df_q = pd.read_csv(csv_quality)
    df_v = pd.read_csv(csv_view)

    q_study = find_required_col(df_q, ["study"], "csv_quality")
    q_video = find_required_col(df_q, ["video"], "csv_quality")
    q_qual = find_required_col(df_q, ["calidad_pred", "quality_pred"], "csv_quality")

    v_study = find_required_col(df_v, ["study"], "csv_view")
    v_video = find_required_col(df_v, ["video"], "csv_view")
    v_view = find_required_col(df_v, ["view"], "csv_view")
    v_prob = find_required_col(df_v, ["view_prob"], "csv_view")

    q = df_q[[q_study, q_video, q_qual]].copy()
    q.columns = ["study", "video", "quality_pred"]

    v = df_v[[v_study, v_video, v_view, v_prob]].copy()
    v.columns = ["study", "video", "view", "view_prob"]

    for df_ in (q, v):
        df_["study"] = df_["study"].apply(normalize_study)
        df_["video"] = df_["video"].apply(normalize_video_name)

    v["_view"] = v["view"].apply(canon_view)

    merged = pd.merge(q, v[["study", "video", "_view", "view_prob"]], on=["study", "video"], how="inner")

    if merged.empty:
        raise RuntimeError("The merge between csv_quality and csv_view is empty. Check the study/video keys.")

    return merged


def extract_study_embeddings(
    study_dir: str,
    temp_study_dir: str,
    model,
    device: torch.device
) -> int:
    """
    Extract embeddings from all .avi files in a study into:
      temp_study_dir/<video>_frameXXXX.npy
    Returns the number of saved embeddings.
    """
    ensure_dir(temp_study_dir)

    total_embeddings = 0
    avi_files = sorted(Path(study_dir).glob("*.avi"))

    if not avi_files:
        log(f"Warning: no .avi files in {study_dir}")
        return 0

    for avi_path in avi_files:
        in_path = str(avi_path)
        base = avi_path.stem

        try:
            frames = load_video_frames(in_path)
        except Exception as e:
            log(f"[ERROR VIDEO] {in_path}: {e}")
            continue

        n = len(frames)
        jump = FRAME_STRIDE_SHORT if n < CLIP_LEN else FRAME_STRIDE_LONG
        saved_for_video = 0

        log(f"Processing {in_path} | {n} frames | stride={jump}")

        for start in range(0, n, jump):
            out_name = f"{base}_frame{start:04d}.npy"
            out_path = os.path.join(temp_study_dir, out_name)

            if os.path.exists(out_path):
                continue

            try:
                clip_np = build_clip(frames, start, clip_len=CLIP_LEN)
                clip = torch.from_numpy(clip_np).to(device)

                with torch.no_grad():
                    emb = model.forward_features(clip)

                emb_np = emb.detach().cpu().numpy().reshape(-1).astype(np.float32)
                save_embedding(out_path, emb_np)
                saved_for_video += 1
                total_embeddings += 1
            except Exception as e:
                log(f"[ERROR EMB] {in_path} start={start}: {e}")

        log(f"Done: {in_path} -> {saved_for_video} embeddings saved")

    return total_embeddings


def round_robin_pick(
    per_video_items: Dict[str, List[dict]],
    k: int,
    used_global: set
) -> Tuple[List[str], set]:
    """
    Selects k paths while trying to distribute them across videos.
    Prioritizes not reusing paths already used globally.
    If there are not enough, it allows reuse.
    """
    if k <= 0:
        return [], set()

    video_ids = sorted(per_video_items.keys(), key=str)
    if not video_ids:
        return [], set()

    cursors_fresh = {vid: 0 for vid in video_ids}
    fresh_lists = {
        vid: [it["path"] for it in per_video_items[vid] if it["path"] not in used_global]
        for vid in video_ids
    }

    selected = []
    selected_set = set()

    progress = True
    while len(selected) < k and progress:
        progress = False
        for vid in video_ids:
            arr = fresh_lists[vid]
            cur = cursors_fresh[vid]
            if cur < len(arr):
                p = arr[cur]
                cursors_fresh[vid] += 1
                selected.append(p)
                selected_set.add(p)
                progress = True
                if len(selected) >= k:
                    break

    if len(selected) < k:
        all_lists = {vid: [it["path"] for it in per_video_items[vid]] for vid in video_ids}
        cursors_all = {vid: 0 for vid in video_ids}

        progress = True
        while len(selected) < k and progress:
            progress = False
            for vid in video_ids:
                arr = all_lists[vid]
                if not arr:
                    continue
                idx = cursors_all[vid] % len(arr)
                p = arr[idx]
                cursors_all[vid] += 1
                selected.append(p)
                selected_set.add(p)
                progress = True
                if len(selected) >= k:
                    break

    return selected, selected_set


def build_study_matrices(
    study: str,
    temp_study_dir: str,
    out_study_dir: str,
    df_meta_study: pd.DataFrame
) -> int:
    """Build matrices for a study from temporary embeddings."""
    ensure_dir(out_study_dir)

    npy_files = sorted(Path(temp_study_dir).glob("*.npy"))
    if not npy_files:
        log(f"Skipped {study}: no temporary embeddings")
        return 0

    files_by_video = defaultdict(list)
    emb_dim = None

    for p in npy_files:
        vid, fidx = parse_filename(p.name)
        if fidx < 0 or fidx % FRAME_STRIDE_MATRIX != 0:
            continue

        p_str = str(p)
        files_by_video[vid].append((fidx, p_str))

        if emb_dim is None:
            emb_dim = load_embedding(p_str).shape[0]

    if emb_dim is None:
        log(f"Skipped {study}: could not infer embedding dimension")
        return 0

    for vid in files_by_video:
        files_by_video[vid].sort(key=lambda x: x[0])

    df_s = df_meta_study.copy()
    df_s = df_s[df_s["_view"].notna()].copy()

    if df_s.empty:
        log(f"Skipped {study}: no valid view/quality metadata")
        return 0

    views_data = defaultdict(lambda: defaultdict(list))
    bad_quality = 0
    missing_any = 0

    for _, r in df_s.iterrows():
        video_val = normalize_video_name(r["video"])
        vcanon = r["_view"]
        if not vcanon:
            continue

        qpred = parse_quality(r["quality_pred"])
        if qpred.shape[0] != QUAL_DIM:
            bad_quality += 1
            qpred = np.zeros(QUAL_DIM, dtype=np.float32)

        vlabel = VIEW_LABELS.get(vcanon, -1.0)
        vprob = parse_float(r["view_prob"], 0.0)

        if video_val not in files_by_video:
            missing_any += 1
            continue

        for fidx, path in files_by_video[video_val]:
            views_data[vcanon][video_val].append({
                "path": path,
                "frame_idx": fidx,
                "qpred": qpred,
                "vlabel": vlabel,
                "vprob": vprob,
            })

    if missing_any:
        log(f"Warning: {study}: {missing_any} CSV rows do not have associated temporary embeddings")
    if bad_quality:
        log(f"Warning: {study}: {bad_quality} rows with invalid quality")

    present = [
        v for v in ["PSAX", "PLAX", "3CH"]
        if v in views_data and any(len(lst) > 0 for lst in views_data[v].values())
    ]

    if not present:
        log(f"Skipped {study}: no resolvable views with embeddings")
        return 0

    if len(present) == 3:
        quotas = {"PSAX": 10, "PLAX": 10, "3CH": 10}
    elif len(present) == 2:
        quotas = {present[0]: 15, present[1]: 15}
    else:
        quotas = {present[0]: 30}

    rows_per_matrix = sum(quotas.values())
    video_ids = sorted({vid for v in views_data for vid in views_data[v].keys()}, key=str)
    vid_to_idx = {vid: i + 1 for i, vid in enumerate(video_ids)}

    path_to_vidx = {}
    path_to_qpred = {}
    path_to_vlabel = {}
    path_to_vprob = {}

    for v in views_data:
        for vid, items in views_data[v].items():
            idx_num = vid_to_idx[vid]
            for it in items:
                path_to_vidx[it["path"]] = float(idx_num)
                path_to_qpred[it["path"]] = it["qpred"].astype(np.float32, copy=False)
                path_to_vlabel[it["path"]] = float(it["vlabel"])
                path_to_vprob[it["path"]] = float(it["vprob"])

    eligible_count = sum(len(items) for v in present for items in views_data[v].values())
    num_matrices = math.ceil(eligible_count / rows_per_matrix) if eligible_count > 0 else 0

    if num_matrices == 0:
        log(f"Skipped {study}: 0 eligible frames")
        return 0

    emb_dim_out = emb_dim + 1 + QUAL_DIM + 1 + 1
    used_global = {v: set() for v in present}

    log(f"\n== Study: {study} ==")
    log(f"Views: {present} | Quotas: {quotas} | Unique videos: {len(video_ids)} | Eligible frames: {eligible_count} | Matrices: {num_matrices}")

    written = 0

    for m in range(1, num_matrices + 1):
        sel_paths = []

        for v in ["PLAX", "PSAX", "3CH"]:
            if v not in quotas:
                continue
            k = quotas[v]
            chosen, chosen_set = round_robin_pick(views_data.get(v, {}), k, used_global.get(v, set()))
            used_global[v].update(chosen_set)
            sel_paths.extend(chosen)

        if len(sel_paths) != rows_per_matrix:
            log(f"Warning: {study} matrix {m}: incomplete selection ({len(sel_paths)} vs {rows_per_matrix})")
            continue

        try:
            X = np.stack([load_embedding(p) for p in sel_paths], axis=0).astype(np.float32)
            if X.shape != (rows_per_matrix, emb_dim):
                raise ValueError(f"Unexpected embedding shape: {X.shape}, expected {(rows_per_matrix, emb_dim)}")

            vcol = np.array([path_to_vidx.get(p, 0.0) for p in sel_paths], dtype=np.float32)[:, None]
            qmat = np.stack([path_to_qpred.get(p, np.zeros(QUAL_DIM, dtype=np.float32)) for p in sel_paths], axis=0)
            vlabel_col = np.array([path_to_vlabel.get(p, -1.0) for p in sel_paths], dtype=np.float32)[:, None]
            vprob_col = np.array([path_to_vprob.get(p, 0.0) for p in sel_paths], dtype=np.float32)[:, None]

            X_out = np.concatenate([X, vcol, qmat, vlabel_col, vprob_col], axis=1)

            if X_out.shape != (rows_per_matrix, emb_dim_out):
                raise ValueError(f"Unexpected final shape: {X_out.shape}, expected {(rows_per_matrix, emb_dim_out)}")
        except Exception as e:
            log(f"Warning: error in {study} matrix {m}: {e}")
            continue

        out_path = os.path.join(out_study_dir, f"matrix{m:03d}.npy")
        np.save(out_path, X_out)
        written += 1
        log(f"Saved {out_path} shape={X_out.shape}")

    return written


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

    for name1 in sorted(os.listdir(root_dir)):
        path1 = os.path.join(root_dir, name1)
        if not os.path.isdir(path1):
            continue

        mats1 = sorted(glob.glob(os.path.join(path1, "matrix*.npy")))
        if only_matrix1:
            mats1 = [m for m in mats1 if os.path.basename(m) in {"matrix001.npy", "matrix1.npy"}]

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

        for name2 in sorted(os.listdir(path1)):
            path2 = os.path.join(path1, name2)
            if not os.path.isdir(path2):
                continue

            mats2 = sorted(glob.glob(os.path.join(path2, "matrix*.npy")))
            if only_matrix1:
                mats2 = [m for m in mats2 if os.path.basename(m) in {"matrix001.npy", "matrix1.npy"}]

            for m in mats2:
                rows.append({
                    "period": name1,
                    "study": name2,
                    "study_key": f"{name1}\\{name2}",
                    "matrix_path": m,
                    "matrix_name": os.path.basename(m),
                })

    return pd.DataFrame(rows, columns=["period", "study", "study_key", "matrix_path", "matrix_name"])


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

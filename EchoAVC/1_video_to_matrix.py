import os
import sys
import cv2
import ast
import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from utils import Task

import numpy as np
import pandas as pd
import torch

from models import MultiTaskModelBBScoreHead

# =============================
# GENERAL CONFIG
# =============================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLIP_LEN = 32
CENTER_CROP_SIZE = 224
MIN_SIDE = 224
FRAME_STRIDE_SHORT = 2   # if video < 32 frames
FRAME_STRIDE_LONG = 5    # if video >= 32 frames

FRAME_STRIDE_MATRIX = 5  # for building matrices: only use _frame values that are multiples of 5

QUAL_DIM = 3

# Mapping from views to numeric label
VIEW_LABELS = {
    "PLAX": 0.0,
    "PSAX": 1.0,
    "3CH": 2.0,
}

# =============================
# LOGGING
# =============================

def log(msg: str):
    print(msg, flush=True)


# =============================
# HELPERS: VIDEO / EMBEDDINGS
# =============================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
    clip = np.stack([frames[i] for i in idxs], axis=0)  # (T,H,W,C)
    clip = clip.astype(np.float32) / 255.0
    clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
    clip = np.transpose(clip, (3, 0, 1, 2))  # (C,T,H,W)
    clip = np.expand_dims(clip, 0)           # (1,C,T,H,W)
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


# =============================
# HELPERS: CSV NORMALIZATION
# =============================

def normalize_study(x) -> str:
    return str(x).strip()


def normalize_video_name(x) -> str:
    """
    Normalizes the video name to a basename without extension.
    """
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

    # Basic heuristic
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


# =============================
# CSV LOADING AND MERGING
# =============================

def find_required_col(df: pd.DataFrame, candidates: List[str], csv_name: str) -> str:
    low_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low_map:
            return low_map[cand.lower()]
    raise ValueError(f"{csv_name} is missing one of these columns: {candidates}")


def load_metadata(csv_quality: str, csv_view: str) -> pd.DataFrame:
    df_q = pd.read_csv(csv_quality)
    df_v = pd.read_csv(csv_view)

    # Quality CSV
    q_study = find_required_col(df_q, ["study"], "csv_quality")
    q_video = find_required_col(df_q, ["video"], "csv_quality")
    q_qual = find_required_col(df_q, ["calidad_pred", "quality_pred"], "csv_quality")

    # View CSV
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


# =============================
# MODEL
# =============================

def load_model(tasks_path: str, checkpoint_path: str, device: torch.device):
    tasks = np.load(tasks_path, allow_pickle=True)

    task_inclusion = [
        'AoCalcium-presence',
        'AoCalcium-score',
        'AoCalcium-multiclass-low',
        'AoCalcium-cutoff-low',
        'AoCalcium-multiclass-high',
        'AoCalcium-cutoff-high',
    ]
    tasks = [t for t in tasks if t.task_name in task_inclusion]

    log("Loading PanEcho model (backbone_only, clip_len=32) ...")
    model = MultiTaskModelBBScoreHead(
        backbone=torch.hub.load(
            'CarDS-Yale/PanEcho',
            'PanEcho',
            force_reload=False,
            backbone_only=True,
            clip_len=CLIP_LEN
        ),
        tasks=tasks,
        fc_dropout=0.4
    )

    chkpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = chkpt['weights']
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    return model


# =============================
# TEMPORARY EXTRACTION PER STUDY
# =============================

def extract_study_embeddings(
    study_dir: str,
    temp_study_dir: str,
    model,
    device: torch.device
) -> int:
    """
    Extracts embeddings from all .avi files in a study into:
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


# =============================
# ROUND-ROBIN SELECTION
# =============================

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

    # First pass: without reuse
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

    # Second pass: allow reuse if needed
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


# =============================
# MATRIX CONSTRUCTION
# =============================

def build_study_matrices(
    study: str,
    temp_study_dir: str,
    out_study_dir: str,
    df_meta_study: pd.DataFrame
) -> int:
    """
    Builds matrices for a study from temporary embeddings.
    Output: out_study_dir/matrixXXX.npy
    """
    ensure_dir(out_study_dir)

    # Index available embeddings
    npy_files = sorted(Path(temp_study_dir).glob("*.npy"))
    if not npy_files:
        log(f"Skipped {study}: no temporary embeddings")
        return 0

    files_by_video = defaultdict(list)
    emb_dim = None

    for p in npy_files:
        vid, fidx = parse_filename(p.name)
        if fidx < 0:
            continue
        if fidx % FRAME_STRIDE_MATRIX != 0:
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

    # Prepare metadata by study/video
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

    video_ids = sorted(
        {vid for v in views_data for vid in views_data[v].keys()},
        key=str
    )
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

        # Fixed order for consistency
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


# =============================
# MAIN PIPELINE
# =============================

def run(
    src_root: str,
    csv_quality: str,
    csv_view: str,
    dest_root: str,
    tasks_path: str,
    checkpoint_path: str,
    keep_temp: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # Strict validations
    if not os.path.isdir(src_root):
        raise NotADirectoryError(f"src_root does not exist or is not a directory: {src_root}")
    if not os.path.isfile(csv_quality):
        raise FileNotFoundError(f"csv_quality does not exist: {csv_quality}")
    if not os.path.isfile(csv_view):
        raise FileNotFoundError(f"csv_view does not exist: {csv_view}")
    if not os.path.isfile(tasks_path):
        raise FileNotFoundError(f"tasks_path does not exist: {tasks_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")

    ensure_dir(dest_root)

    df_meta = load_metadata(csv_quality, csv_view)
    studies_meta = set(df_meta["study"].unique())

    study_dirs = sorted([p for p in Path(src_root).iterdir() if p.is_dir()])
    if not study_dirs:
        raise RuntimeError(f"No study subdirectories were found in: {src_root}")

    model = load_model(tasks_path, checkpoint_path, device)

    temp_root_obj = tempfile.TemporaryDirectory(prefix="echo_temp_")
    temp_root = temp_root_obj.name
    log(f"Temp root: {temp_root}")

    total_studies = 0
    total_embeddings = 0
    total_matrices = 0

    try:
        for study_path in study_dirs:
            study = study_path.name
            total_studies += 1

            log(f"\n{'='*80}")
            log(f"STUDY: {study}")

            if study not in studies_meta:
                log(f"Warning: study {study} does not appear in the CSVs. Skipping.")
                continue

            temp_study_dir = os.path.join(temp_root, study)
            out_study_dir = os.path.join(dest_root, study)
            ensure_dir(temp_study_dir)
            ensure_dir(out_study_dir)

            # 1) Extract temporary embeddings
            n_emb = extract_study_embeddings(
                study_dir=str(study_path),
                temp_study_dir=temp_study_dir,
                model=model,
                device=device
            )
            total_embeddings += n_emb

            # 2) Build matrices for this study
            df_meta_study = df_meta[df_meta["study"] == study].copy()
            n_mat = build_study_matrices(
                study=study,
                temp_study_dir=temp_study_dir,
                out_study_dir=out_study_dir,
                df_meta_study=df_meta_study
            )
            total_matrices += n_mat

            # 3) Clean temp files for the study if they should not be kept
            if not keep_temp:
                shutil.rmtree(temp_study_dir, ignore_errors=True)
                log(f"Temp deleted: {temp_study_dir}")

        log(f"\n{'='*80}")
        log("SUMMARY")
        log(f"Studies inspected: {total_studies}")
        log(f"Temporary embeddings saved: {total_embeddings}")
        log(f"Final matrices written: {total_matrices}")
        log(f"Final destination: {dest_root}")

        if keep_temp:
            log(f"Temporary files kept in: {temp_root}")
        else:
            log("Temporary files deleted at the end.")

    finally:
        if keep_temp:
            # If temp files should be kept, do not close/delete the TemporaryDirectory yet.
            # We keep it alive during execution, but once the Python process ends it will be lost
            # unless you use a fixed path. For real debugging, it is better to replace it with a fixed directory.
            pass
        else:
            temp_root_obj.cleanup()


# =============================
# CLI
# =============================

if __name__ == "__main__":
    src_root = r'..\VALVE_VIDS'
    csv_quality = r'EchoAVC\data\quality.csv'
    csv_view = r'EchoAVC\data\view.csv'
    dest_root = r'EchoAVC\matrix_out'
    tasks_path = r'EchoAVC\content\tasks_v6.npy'
    checkpoint_path = r'EchoAVC\data\echoavc_feature_extraction.pt'
    keep_temp = 0

    run(
        src_root=src_root,
        csv_quality=csv_quality,
        csv_view=csv_view,
        dest_root=dest_root,
        tasks_path=tasks_path,
        checkpoint_path=checkpoint_path,
        keep_temp=keep_temp,
    )

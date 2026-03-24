import os
import sys
import cv2
import math
import numpy as np
import torch
from typing import List, Tuple
import timm
from models import MultiTaskModelBBScoreHead
from utils import Task

# -----------------------------
# Utils
# -----------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def log(msg: str):
    print(msg, flush=True)


def ensure_min_size(img: np.ndarray, min_side: int = 224) -> np.ndarray:
    h, w = img.shape[:2]
    short = min(h, w)
    if short >= min_side:
        return img
    scale = min_side / short
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop(img: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def load_video_frames(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir el video: {path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ensure_min_size(frame, 224)
            frame = center_crop(frame, 224)
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Video sin frames: {path}")
    return frames


def build_clip(frames: List[np.ndarray], start: int, clip_len: int = 32) -> np.ndarray:
    n = len(frames)
    idxs = [(start + i) % n for i in range(clip_len)]
    clip = np.stack([frames[i] for i in idxs], axis=0)
    clip = clip.astype(np.float32) / 255.0
    clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
    clip = np.transpose(clip, (3, 0, 1, 2))
    clip = np.expand_dims(clip, 0)
    return clip


def save_embedding(out_path: str, emb: np.ndarray):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, emb)


# -----------------------------
# Main
# -----------------------------

def run(root_dir: str, dest_root: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_path = r'content\tasks_v6.npy'
    tasks = np.load(task_path, allow_pickle=True)

    task_inclusion = []
    task_inclusion.append('AoCalcium-presence')
    task_inclusion.append('AoCalcium-score')
    task_inclusion.append('AoCalcium-multiclass-low')
    task_inclusion.append('AoCalcium-cutoff-low')
    task_inclusion.append('AoCalcium-multiclass-high')
    task_inclusion.append('AoCalcium-cutoff-high')
    tasks = [t for t in tasks if t.task_name in task_inclusion]

    log("Cargando modelo PanEcho (backbone_only, clip_len=32) desde torch.hub…")
    model = MultiTaskModelBBScoreHead(
    # model = MultiTaskModelBBComplex(
        backbone=torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho', force_reload=False, backbone_only=True, clip_len=32),
        tasks=tasks,
        fc_dropout=0.4 
        # hidden_units=1536
    )    
    checkpoint_path = r'data\echoavc_feature_extraction.pt'
    chkpt = torch.load(checkpoint_path, map_location='cpu')
    filtered_state_dict = {}

    for k, v in chkpt['weights'].items():
        # Incluir capas base (backbone) o las cabezas permitidas
        filtered_state_dict[k] = v
            
    # Cargar las capas filtradas
    model.load_state_dict(filtered_state_dict, strict=True)

    print(model)
    model.eval().to(device)
    total_videos = 0
    total_embeddings = 0
    root_dirs = [

                 r'VALVE_VIDS', 

                 ]
    
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            # if 'studyID2__1135' not in dirpath:
            #     continue
            for fname in filenames:
                if not fname.lower().endswith('.avi'):
                    continue
                in_path = os.path.join(dirpath, fname)
                base, _ = os.path.splitext(fname)

                rel_dir = os.path.relpath(dirpath, root_dir)
                out_dir = os.path.join(dest_root, rel_dir)

                try:
                    frames = load_video_frames(in_path)
                except Exception as e:
                    log(f"[ERROR] {in_path}: {e}")
                    continue

                n = len(frames)
                total_videos += 1
                log(f"Procesando {in_path} | {n} frames")
                if n < 32:
                    jump = 2
                else:
                    jump = 5
                for start in range(0, n, jump):
                    out_name = f"{base}_frame{start:04d}.npy"
                    out_path = os.path.join(out_dir, out_name)
                    
                    if os.path.exists(out_path):
                        print(f"  - Ya existe {out_path}, saltando...")
                        continue

                    clip_np = build_clip(frames, start, clip_len=32)
                    clip = torch.from_numpy(clip_np).to(device)
                    with torch.no_grad():
                        emb = model.forward_features(clip)
                    emb_np = emb.detach().cpu().numpy().reshape(-1)

                    try:
                        save_embedding(out_path, emb_np)
                    except Exception as e:
                        log(f"[ERROR SAVE] {out_path}: {e}")
                        continue

                    total_embeddings += 1

                log(f"Listo: {in_path} → {n} embeddings guardados en {out_dir}")

    log(f"Resumen: {total_videos} vídeos procesados | {total_embeddings} embeddings guardados")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)

    root = sys.argv[1]
    dest = r"data\row_pretrain_embeddings"
    run(root, dest)

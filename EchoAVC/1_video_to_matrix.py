import os
import shutil
import tempfile
from pathlib import Path

import torch

from models import load_model
from utils import build_study_matrices, ensure_dir, extract_study_embeddings, load_metadata, log, Task


def run(
    src_root: str,
    csv_quality: str,
    csv_view: str,
    dest_root: str,
    checkpoint_path: str,
    keep_temp: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    if not os.path.isdir(src_root):
        raise NotADirectoryError(f"src_root does not exist or is not a directory: {src_root}")
    if not os.path.isfile(csv_quality):
        raise FileNotFoundError(f"csv_quality does not exist: {csv_quality}")
    if not os.path.isfile(csv_view):
        raise FileNotFoundError(f"csv_view does not exist: {csv_view}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")

    ensure_dir(dest_root)
    
    tasks_path = r'EchoAVC\content\tasks_v6.npy'

    df_meta = load_metadata(csv_quality, csv_view)
    studies_meta = set(df_meta["study"].unique())

    study_dirs = sorted([p for p in Path(src_root).iterdir() if p.is_dir()])
    if not study_dirs:
        raise RuntimeError(f"No study subdirectories were found in: {src_root}")

    model = load_model(tasks_path, checkpoint_path, device)
    print(model)
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

            n_emb = extract_study_embeddings(
                study_dir=str(study_path),
                temp_study_dir=temp_study_dir,
                model=model,
                device=device,
            )
            total_embeddings += n_emb

            df_meta_study = df_meta[df_meta["study"] == study].copy()
            n_mat = build_study_matrices(
                study=study,
                temp_study_dir=temp_study_dir,
                out_study_dir=out_study_dir,
                df_meta_study=df_meta_study,
            )
            total_matrices += n_mat

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
        if not keep_temp:
            temp_root_obj.cleanup()


if __name__ == "__main__":
    src_root = r'\\NAS3_Z\all\BKP_PERE\BBDD_Datasets\TEST_ECHOAVC\TEST_V\TEST\VALVE_VIDS'
    csv_quality = r'EchoAVC\data\quality.csv'
    csv_view = r'EchoAVC\data\view.csv'
    dest_root = r'EchoAVC\matrix_out'
    checkpoint_path = r'EchoAVC\data\echoavc_feature_extraction.pt'
    keep_temp = 0

    run(
        src_root=src_root,
        csv_quality=csv_quality,
        csv_view=csv_view,
        dest_root=dest_root,
        checkpoint_path=checkpoint_path,
        keep_temp=keep_temp,
    )

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import ModelDiagnostic, load_model
from utils import (
    EMB_D,
    EMB_T,
    MatrixInferenceDataset,
    aggregate_by_study,
    build_study_matrices,
    ensure_dir,
    extract_study_embeddings,
    load_metadata,
    log,
    run_inference,
    Task,
    set_seed,
)



def list_study_matrices(study: str, out_study_dir: str) -> pd.DataFrame:
    rows = []
    for matrix_path in sorted(Path(out_study_dir).glob("matrix*.npy")):
        rows.append(
            {
                "period": "",
                "study": study,
                "study_key": study,
                "matrix_path": str(matrix_path),
                "matrix_name": matrix_path.name,
            }
        )

    return pd.DataFrame(rows, columns=["period", "study", "study_key", "matrix_path", "matrix_name"])


def save_predictions(df_pred_matrix: pd.DataFrame, out_csv: str):
    df_pred_study = aggregate_by_study(df_pred_matrix)

    out_dir = os.path.dirname(out_csv) or "."
    ensure_dir(out_dir)
    out_csv_matrix = get_matrix_csv_path(out_csv)

    df_pred_study.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_pred_matrix.to_csv(out_csv_matrix, index=False, encoding="utf-8-sig")

    return out_csv_matrix


def get_matrix_csv_path(out_csv: str) -> str:
    base, ext = os.path.splitext(out_csv)
    return f"{base}_by_matrix{ext if ext else '.csv'}"


def load_completed_studies(out_csv: str) -> set:
    if not os.path.isfile(out_csv):
        return set()

    try:
        df_done = pd.read_csv(out_csv)
    except Exception as e:
        log(f"Warning: could not read existing predictions CSV {out_csv}: {e}")
        return set()

    if "study" not in df_done.columns:
        log(f"Warning: existing predictions CSV has no 'study' column: {out_csv}")
        return set()

    done = set(df_done["study"].dropna().astype(str).str.strip())
    log(f"Studies already present in predictions CSV: {len(done)}")
    return done


def load_existing_matrix_predictions(out_csv: str) -> pd.DataFrame:
    out_csv_matrix = get_matrix_csv_path(out_csv)
    if not os.path.isfile(out_csv_matrix):
        return pd.DataFrame(columns=["study", "study_key", "matrix_path", "matrix_name", "EchoAVC_PRES", "EchoAVC"])

    try:
        df_existing = pd.read_csv(out_csv_matrix)
    except Exception as e:
        log(f"Warning: could not read existing matrix predictions CSV {out_csv_matrix}: {e}")
        return pd.DataFrame(columns=["study", "study_key", "matrix_path", "matrix_name", "EchoAVC_PRES", "EchoAVC"])

    log(f"Existing matrix predictions loaded: {len(df_existing)}")
    return df_existing


def run(
    src_root: str,
    csv_quality: str,
    csv_view: str,
    dest_root: str,
    checkpoint_path: str,
    aggregator_model_path: str,
    out_csv: str,
    keep_temp: bool = False,
    seed: int = 42,
):
    set_seed(seed)

    feature_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_device = feature_device
    log(f"Feature device: {feature_device}")
    log(f"Inference device: {infer_device}")

    if not os.path.isdir(src_root):
        raise NotADirectoryError(f"src_root does not exist or is not a directory: {src_root}")
    if not os.path.isfile(csv_quality):
        raise FileNotFoundError(f"csv_quality does not exist: {csv_quality}")
    if not os.path.isfile(csv_view):
        raise FileNotFoundError(f"csv_view does not exist: {csv_view}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")
    if not os.path.isfile(aggregator_model_path):
        raise FileNotFoundError(f"aggregator_model_path does not exist: {aggregator_model_path}")

    ensure_dir(dest_root)

    tasks_path = r"EchoAVC\content\tasks_v6.npy"

    df_meta = load_metadata(csv_quality, csv_view)
    studies_meta = set(df_meta["study"].unique())

    study_dirs = sorted([p for p in Path(src_root).iterdir() if p.is_dir()])
    if not study_dirs:
        raise RuntimeError(f"No study subdirectories were found in: {src_root}")

    feature_model = load_model(tasks_path, checkpoint_path, feature_device)

    infer_model = ModelDiagnostic(d_in=EMB_D).to(infer_device)
    infer_state = torch.load(aggregator_model_path, map_location=infer_device)
    infer_model.load_state_dict(infer_state)
    infer_model.eval()

    temp_root_obj = tempfile.TemporaryDirectory(prefix="echo_temp_")
    temp_root = temp_root_obj.name
    log(f"Temp root: {temp_root}")

    total_studies = 0
    total_embeddings = 0
    total_matrices = 0
    pred_matrix_parts = []
    completed_studies = load_completed_studies(out_csv)
    df_pred_matrix_existing = load_existing_matrix_predictions(out_csv)
    if not df_pred_matrix_existing.empty:
        pred_matrix_parts.append(df_pred_matrix_existing)

    try:
        for study_path in study_dirs:
            study = study_path.name
            total_studies += 1

            log(f"\n{'=' * 80}")
            log(f"STUDY: {study}")

            if study not in studies_meta:
                log(f"Warning: study {study} does not appear in the CSVs. Skipping.")
                continue

            if study in completed_studies:
                log(f"Skipping {study}: already present in {out_csv}")
                continue

            temp_study_dir = os.path.join(temp_root, study)
            out_study_dir = os.path.join(dest_root, study)
            ensure_dir(temp_study_dir)
            ensure_dir(out_study_dir)

            n_emb = extract_study_embeddings(
                study_dir=str(study_path),
                temp_study_dir=temp_study_dir,
                model=feature_model,
                device=feature_device,
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

            mats_df = list_study_matrices(study, out_study_dir)
            if mats_df.empty:
                log(f"Warning: no matrices found for study {study}.")
            else:
                ds = MatrixInferenceDataset(mats_df, expected_shape=(EMB_T, EMB_D))
                loader = DataLoader(
                    ds,
                    batch_size=64,
                    shuffle=False,
                    num_workers=0,
                )

                df_pred_matrix_study = run_inference(infer_model, loader, infer_device)
                pred_matrix_parts.append(df_pred_matrix_study)
                completed_studies.add(study)

                df_pred_matrix_all = pd.concat(pred_matrix_parts, ignore_index=True)
                out_csv_matrix = save_predictions(df_pred_matrix_all, out_csv)
                log(f"Predictions updated: {out_csv}")
                log(f"Matrix predictions updated: {out_csv_matrix}")

            if not keep_temp:
                shutil.rmtree(temp_study_dir, ignore_errors=True)
                log(f"Temp deleted: {temp_study_dir}")

        if pred_matrix_parts:
            df_pred_matrix_all = pd.concat(pred_matrix_parts, ignore_index=True)
        else:
            df_pred_matrix_all = pd.DataFrame(
                columns=["study", "study_key", "matrix_path", "matrix_name", "EchoAVC_PRES", "EchoAVC"]
            )

        out_csv_matrix = save_predictions(df_pred_matrix_all, out_csv)

        log(f"\n{'=' * 80}")
        log("SUMMARY")
        log(f"Studies inspected: {total_studies}")
        log(f"Temporary embeddings saved: {total_embeddings}")
        log(f"Final matrices written: {total_matrices}")
        log(f"Study-level CSV saved to: {out_csv}")
        log(f"Matrix-level CSV saved to: {out_csv_matrix}")
        log(f"Final destination: {dest_root}")

        if keep_temp:
            log(f"Temporary files kept in: {temp_root}")
        else:
            log("Temporary files deleted at the end.")

    finally:
        if not keep_temp:
            temp_root_obj.cleanup()


if __name__ == "__main__":
    src_root = r"E:\DICOM\videos_valve"

    csv_quality = r"EchoAVC\data\quality_leuven.csv"
    csv_view = r"EchoAVC\data\view_leuven.csv"

    dest_root = r"E:\DICOM\matrix_out"
    checkpoint_path = r"EchoAVC\data\echoavc_feature_extraction.pt"
    aggregator_model_path = r"EchoAVC\results\aggregator_model.pt"
    out_csv = r"EchoAVC\results\EchoAVC_predictions.csv"
    keep_temp = 0

    run(
        src_root=src_root,
        csv_quality=csv_quality,
        csv_view=csv_view,
        dest_root=dest_root,
        checkpoint_path=checkpoint_path,
        aggregator_model_path=aggregator_model_path,
        out_csv=out_csv,
        keep_temp=keep_temp,
    )

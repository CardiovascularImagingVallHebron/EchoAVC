import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from models import ModelDiagnostic
from utils import (
    EMB_D,
    EMB_T,
    MatrixInferenceDataset,
    aggregate_by_study,
    ensure_dir,
    list_study_matrices_flexible,
    log,
    run_inference,
    set_seed,
)


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

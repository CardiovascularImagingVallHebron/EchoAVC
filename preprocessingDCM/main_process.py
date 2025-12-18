import os
import multiprocessing
import pandas as pd
from src.utils import extract_frames_from_dicom_direct, new_resize_img, new_orig_img
from src.utils_cone_extract import cone_extract
import pydicom
from tqdm import tqdm
import numpy as np
import cv2
import time

def process_dicom_file(args):
    rel_dir, full_dicom_path, root_out_folder, low_mask_records = args

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    x_dim, y_dim = 256, 256
    fps = 15

    # Filename without extension
    filename = os.path.splitext(os.path.basename(full_dicom_path))[0]

    # Replicate directory structure from DICOM inside AVI
    res_path  = os.path.join(root_out_folder, 'vids_resized', rel_dir)
    crop_path = os.path.join(root_out_folder, 'vids_cropped', rel_dir)
    os.makedirs(res_path, exist_ok=True)
    os.makedirs(crop_path, exist_ok=True)

    vid_res_path  = os.path.join(res_path,  f'{filename}.avi')
    vid_crop_path = os.path.join(crop_path, f'{filename}.avi')

    if os.path.exists(vid_res_path) and os.path.exists(vid_crop_path):
        print(f"Skipped (already exists): {os.path.join(rel_dir, filename + '.dcm')}")
        return

    try:
        ds = pydicom.dcmread(full_dicom_path, force=True)

        if (ds.PhotometricInterpretation not in ['MONOCHROME1', 'MONOCHROME2'] and len(ds.pixel_array.shape) < 4) or \
           (ds.PhotometricInterpretation in ['MONOCHROME1', 'MONOCHROME2'] and len(ds.pixel_array.shape) < 3):
            return

        try:
            frames, frame_ecg_mask = extract_frames_from_dicom_direct(full_dicom_path)
            mask = cone_extract(frames, frame_ecg_mask, 4)
            suma_mask = np.sum(mask)
            total_image = mask.shape[0] * mask.shape[1]
            percentage_mask = (suma_mask / total_image * 100) / 3

            resize_writer = cv2.VideoWriter(vid_res_path, fourcc, fps, (x_dim, y_dim))
            croped_writer = None

            if percentage_mask > 15:
                for i, frame in enumerate(frames):
                    frame = (frame / 255.).astype(np.float32)

                    res_frame, res_mask, _ = new_resize_img(frame * mask, mask, x_dim, y_dim)
                    res_cone = (res_frame * res_mask * 255).astype(np.uint8)
                    if res_cone.ndim == 2:
                        res_cone = cv2.cvtColor(res_cone, cv2.COLOR_GRAY2BGR)
                    else:
                        res_cone = cv2.cvtColor(res_cone, cv2.COLOR_RGB2BGR)
                    resize_writer.write(res_cone)

                    crop_frame, crop_mask = new_orig_img(frame * mask, mask)
                    crop_cone = (crop_frame * crop_mask * 255).astype(np.uint8)
                    if crop_cone.ndim == 2:
                        crop_cone = cv2.cvtColor(crop_cone, cv2.COLOR_GRAY2BGR)
                    else:
                        crop_cone = cv2.cvtColor(crop_cone, cv2.COLOR_RGB2BGR)

                    if i == 0:
                        croped_writer = cv2.VideoWriter(
                            vid_crop_path, fourcc, fps, (crop_cone.shape[1], crop_cone.shape[0])
                        )
                    croped_writer.write(crop_cone)

                print(f'✅ OK: {os.path.join(rel_dir, filename)}')
            else:
                low_mask_records.append((rel_dir, filename))

                for i, frame in enumerate(frames):
                    if frame.ndim == 3:
                        h, w, _ = frame.shape
                    else:
                        h, w = frame.shape

                    min_dim = min(h, w)
                    cx, cy = w // 2, h // 2
                    x1 = max(cx - min_dim // 2, 0)
                    y1 = max(cy - min_dim // 2, 0)
                    x2, y2 = x1 + min_dim, y1 + min_dim

                    cropped = frame[y1:y2, x1:x2]
                    resized = cv2.resize(cropped, (x_dim, y_dim), interpolation=cv2.INTER_AREA)

                    if cropped.ndim == 2:
                        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
                    else:
                        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

                    if resized.ndim == 2:
                        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                    else:
                        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

                    if i == 0:
                        croped_writer = cv2.VideoWriter(
                            vid_crop_path, fourcc, fps, (cropped_bgr.shape[1], cropped_bgr.shape[0])
                        )

                    resize_writer.write(resized_bgr)
                    croped_writer.write(cropped_bgr)

                print(f'⚠️ OK (low mask): {os.path.join(rel_dir, filename)}')

            resize_writer.release()
            if croped_writer is not None:
                croped_writer.release()

        except Exception as e:
            print(f"❌ Error {e} in: {full_dicom_path}")

    except Exception as e:
        print(f"❌ Error processing {full_dicom_path}: {e}")


def process_patient_folders(root_folder, root_out_folder):
    tasks = []
    manager = multiprocessing.Manager()
    low_mask_records = manager.list()

    # Traverse the entire TEST_DICOM tree
    for dirpath, _, filenames in os.walk(root_folder):
        # Relative path with respect to root_folder
        rel_dir = os.path.relpath(dirpath, root_folder)
        # If it is the root, leave an empty string to avoid creating a "." folder
        if rel_dir == '.':
            rel_dir = ''

        for fn in filenames:
            # if fn.lower().endswith('.dcm'):
            full_path = os.path.join(dirpath, fn)
            tasks.append((rel_dir, full_path, root_out_folder, low_mask_records))

    total = len(tasks)
    with tqdm(total=total, desc="Processing DICOM: ") as pbar:
        with multiprocessing.Pool() as pool:
            for _ in pool.imap_unordered(process_dicom_file, tasks):
                pbar.update()

    # Save CSV with low mask
    if len(low_mask_records) > 0:
        df = pd.DataFrame(list(low_mask_records), columns=['relative_folder', 'filename'])
        csv_path = os.path.join(root_out_folder, 'low_mask_files.csv')
        df.to_csv(csv_path, index=False)
        print(f"📄 CSV saved at: {csv_path}")


if __name__ == '__main__':
    start = time.time()
    root_folder = r'..\TEST_DICOM'
    root_out_folder = r'..\TEST_AVI'
    process_patient_folders(root_folder, root_out_folder)
    print('Total time: ', time.time() - start, 's')

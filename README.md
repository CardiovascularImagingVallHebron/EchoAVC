# EchoAVC: Artificial intelligence for aortic valve calcium score quantification by echocardiography

EchoAVC is a framework for accurate, non-invasive detection and quantification of aortic valve calcification (AVC) using echocardiographic videos.

[[Preprint](https://www.medrxiv.org/content/10.64898/2025.12.26.25343075v1)]

---

![EchoAVC Overview](EchoAVC/content/echoavc_twostages.png)

## Repository Structure

- **preprocessingDCM**: Converts echocardiography studies from DICOM (DCM) format to AVI. Requires a source folder (`root_folder`) containing DICOM files and an output folder (`root_out_folder`) where the converted AVI files will be saved. Use the conda environment defined in `prepro.yaml`.
- **EchoAVC**: Identifies and quantifies aortic valve calcium. Feature extraction builds on [CarDS-Yale/PanEcho](https://github.com/CarDS-Yale/PanEcho); to aggregate video-level features at the study level, a transformer-based architecture is used.

## Model Weights and Repositories

- EchoQuality: [Weights](https://huggingface.co/perolope/EchoQuality) & [Repository](https://github.com/CardiovascularImagingVallHebron/EchoQuality)
- AV detection: [Weights](https://huggingface.co/perolope/AVdetector) & [Repository](https://github.com/CardiovascularImagingVallHebron/AoVdetector)
- EchoAVC: [Weights](https://huggingface.co/perolope/EchoAVC) & Repository (Here!)

## Getting Started

### 1. Clone the repository:

```bash
git clone https://github.com/CardiovascularImagingVallHebron/EchoAVC.git
cd EchoAVC
```

### 2. Preprocessing DICOM to AVI

1. Create a conda environment from `preprocessingDCM/prepro.yaml`:
   ```bash
   conda env create -f preprocessingDCM/prepro.yaml
   conda activate prepro
   ```
2. Prepare your input folder (`root_folder`) containing DICOM files.
3. Specify an output folder (`root_out_folder`) where converted AVI files will be saved. Inside this folder, the script will create:
   - `vids_resized`: Videos cropped to the echo cone and resized to 256x256.
   - `vids_cropped`: Videos cropped to the echo cone at original resolution.
4. Run the preprocessing script (see `preprocessingDCM` for specific instructions).

### 3. Aortic Valve Detection

Follow the manual usage of the [AV Detector](https://github.com/CardiovascularImagingVallHebron/AoVdetector) repository.

### 4. Echo Quality Assessment

Follow the manual usage of the [EchoQuality](https://github.com/CardiovascularImagingVallHebron/EchoQuality) repository.

### 5. Echocardiography Aortic Valve Calcification (EchoAVC) Quantification

#### Step 0: Environment installation

For the following steps, create the conda environment from `EchoAVC\content\torchone.yaml`:

```bash
conda env create -f EchoAVC\content\torchone.yaml
conda activate torchone
```

#### Run the complete pipeline directly from videos

Use `EchoAVC/3_video_to_predictions_direct.py` to perform feature extraction, matrix construction, and study-level inference in one run. You do **not** need to run `1_video_to_matrix.py` and then `2_matrix_inference_direct.py` separately. Just run 1* and then 2* if you want to save the video embeddings.

1. Download both model files from the [EchoAVC Hugging Face repository](https://huggingface.co/perolope/EchoAVC):
   - `echoavc_feature_extraction.pt`
   - `aggregator_model.pt`
2. Open `EchoAVC/3_video_to_predictions_direct.py` and set the paths in the `if __name__ == "__main__":` section:
   - `src_root`: directory containing one subfolder per study, with valve-cropped `.avi` videos (`src_root/<study>/*.avi`)
   - `csv_quality`: CSV containing video-level quality predictions
   - `csv_view`: CSV containing video-level view predictions
   - `dest_root`: directory in which the generated study matrices will be saved
   - `checkpoint_path`: path to `echoavc_feature_extraction.pt`
   - `aggregator_model_path`: path to `aggregator_model.pt`
   - `out_csv`: destination for the study-level prediction CSV
   - `keep_temp`: set to `0` to delete temporary embeddings, or `1` to keep them

   Example:

   ```python
   src_root = r"EchoAVC\data\videos_valve"
   csv_quality = r"EchoAVC\data\quality_example.csv"
   csv_view = r"EchoAVC\data\view_example.csv"
   dest_root = r"EchoAVC\data\matrix_out"
   checkpoint_path = r"EchoAVC\models\echoavc_feature_extraction.pt"
   aggregator_model_path = r"EchoAVC\results\aggregator_model.pt"
   out_csv = r"EchoAVC\results\EchoAVC_predictions.csv"
   keep_temp = 0
   ```

3. From the repository root, run:

   ```bash
   python EchoAVC/3_video_to_predictions_direct.py
   ```

The script processes each study end to end and saves results incrementally. If it is restarted with an existing `out_csv`, studies already present in that file are skipped.

**Note:** Example quality and view CSV files are provided in `EchoAVC/data`. Study names and video identifiers in these CSVs must match the input folder and video names.

The generated matrix for each study contains up to 30 rows of 774 features (768 PanEcho embedding features, one video identifier, three quality values, one numeric view label, and one view probability). Matrices are saved as `dest_root/<study>/matrixXXX.npy`.

The script creates:

- **Study-level predictions** → `EchoAVC_predictions.csv` (the path configured in `out_csv`)
- **Matrix-level predictions** → `EchoAVC_predictions_by_matrix.csv`

Each study-level prediction includes:

- `EchoAVC_PRES`: probability of EchoAVC presence
- `EchoAVC`: continuous EchoAVC score prediction

---

### Citation

In case of use this repository, please cite:

Lopez-Gutierrez, Pere, et al. "**Artificial intelligence for aortic valve calcium score quantification by echocardiography**." MedRxiv.

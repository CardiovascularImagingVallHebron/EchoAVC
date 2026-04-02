# EchoAVC: Artificial intelligence for aortic valve calcium score quantification by echocardiography

EchoAVC is a framework for accurate, non-invasive detection and quantification of aortic valve calcification (AVC) using echocardiographic videos.

![EchoAVC Overview](echoavc_twostages.png)

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

### 5. Aortic Valve Calcification (AVC) Quantification

#### Step 0: Environment installation

For the following steps, create the conda environment from `EchoAVC\content\torchone.yaml`:

```bash
conda env create -f EchoAVC\content\torchone.yaml
conda activate torchone
```

#### Step 1: Build Study-Level Matrices from Videos

1. Run `EchoAVC/1_video_to_matrix.py`.
2. Set the following paths inside the script:
   - `src_root`: directory containing study subfolders with valve-cropped `.avi` videos (`src_root/study/*.avi`)
   - `csv_quality`: CSV with video-level quality predictions
   - `csv_view`: CSV with video-level view predictions
   - `dest_root`: output directory where study matrices will be saved
   - `tasks_path`: PanEcho task definition file
   - `checkpoint_path`: **`echoavc_feature_extraction.pt` (download from the provided HuggingFace link)**
3. The script:
   - Extracts PanEcho embeddings from each video
   - Builds fixed-size matrices using metadata from both CSVs

**Note:** Example CSV files are provided in `EchoAVC/data`.

4. Matrix Format

   Each video clip is converted into a **774-dimensional feature vector** composed of:
   - 768 PanEcho embedding features
   - 1 video identifier
   - 3 quality prediction values
   - 1 numeric view label
   - 1 view probability

   Up to **30 rows** are selected per study, producing matrices of shape:

   **`30 × 774`**

   Matrices are saved as `.npy` files in: dest_root/<study>/matrixXXX.npy

   The number of matrices per study is dynamic, depending on the number of available clips.

---

#### Step 2: Study-Level EchoAVC aggregator

1. Run `EchoAVC/2_matrix_inference_direct.py`.
2. Set the following paths inside the script:
   - `matrix_root`: directory containing matrices generated in Step 1
   - `model_path`: **`aggregator_model.pt` (download from the same HuggingFace link)**
   - `out_csv`: output CSV path for study-level predictions
3. The script:
   - Loads all matrices
   - Runs inference using the aggregation model
   - Aggregates predictions at study level

4. Outputs:
   - **Study-level predictions** → `EchoAVC_predictions.csv`
   - **Matrix-level predictions** → `EchoAVC_predictions_by_matrix.csv`

Each study-level prediction includes:

- `EchoAVC_PRES`: probability of EchoAVC presence
- `EchoAVC`: continuous EchoAVC score prediction

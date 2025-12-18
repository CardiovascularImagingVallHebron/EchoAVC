# EchoAVC

EchoAVC is a framework for accurate, non-invasive detection and quantification of aortic valve calcification (AVC), providing diagnostic and prognostic value for aortic stenosis progression and the need for valve replacement. This technique is designed as a scalable tool for early detection and clinical management of aortic valve stenosis.

## Repository Structure

- **preprocessingDCM**: Converts echocardiography studies from DICOM (DCM) format to AVI.
- **AVdetection**: Detects the aortic valve (AV) in echocardiography images.
- **EchoQuality**: Classifies echocardiography quality. The model is fine-tuned from [CarDS-Yale/PanEcho](https://github.com/CarDS-Yale/PanEcho), and part of the code is adapted from that project.
- **EchoAVC**: Identifies and quantifies aortic valve calcium. Feature extraction builds on [CarDS-Yale/PanEcho](https://github.com/CarDS-Yale/PanEcho); to aggregate video-level features at the study level, a transformer-based architecture is used.

## Model Weights

- EchoQuality weights: https://huggingface.co/perolope/EchoQuality  
- AV detection weights: https://huggingface.co/perolope/AVdetector  
- EchoAVC weights: https://huggingface.co/perolope/EchoAVC

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/CardiovascularImagingVallHebron/EchoAVC.git
   cd EchoAVC
   ```
2. Install dependencies (add your environment steps here).
3. Download the model weights from the links above as needed for each module.
4. Follow subfolder-specific instructions for preprocessing, detection, quality assessment, and AVC quantification workflows.

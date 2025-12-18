import os
import torch
import numpy as np
import pandas as pd
from torchvision import models
from PIL import Image
from torchvision.transforms import ToTensor
import csv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2

# Models trained path
model_path = 'results/model_avdetector.pth'

# Paths of the images
base_dirs = [r'..\TEST_AVI\vids_cropped']

# CSV to save results
output_csv = 'inference_test.csv'

# Labels of the views (adjust if necessary)
label_to_view = {1: 'psax_aov', 2: 'plax', 3: '3c'}

# Device (GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Image transformation
transform = ToTensor()

# Function to load the model only once per process (not inside the loop)
def load_model(model_path, device):
    model = models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')

    num_classes = 4
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.rpn.nms_thresh = 0.7
    model.rpn.post_nms_top_n_test = 1
    model.roi_heads.detections_per_img = 1

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = None

def initialize_model(model_path, device):
    global model
    model = load_model(model_path, device)

def process_frame(data, image_np, patient, avi, frame_count):
    try:
        global model
        if model is None:
            raise ValueError("Model not loaded")
        
        # Perform inference
        image = Image.fromarray(image_np).resize((256, 256))
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

        pred_boxes = outputs[0]['boxes'].cpu().numpy()
        pred_labels = outputs[0]['labels'].cpu().numpy()
        pred_scores = outputs[0]['scores'].cpu().numpy()

        if len(pred_boxes) > 0:
            max_idx = np.argmax(pred_scores)
            p1 = pred_boxes[max_idx][:2]
            p2 = pred_boxes[max_idx][2:]
            label = pred_labels[max_idx]
            vista = label_to_view.get(label, 'unknown')
            return [data, patient, avi, frame_count, list(p1), list(p2), vista]
        
        return None
    except Exception as e:
        print(f"Error processing {patient} - {avi} - {frame_count}: {e}")
        return None

# Ensure the code only runs when the main script is executed directly
if __name__ == "__main__":
    # Read the existing CSV if it has already been created and is not empty
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        df_results = pd.read_csv(output_csv)
    else:
        df_results = pd.DataFrame(columns=['data', 'patient', 'file', 'frame', 'p1', 'p2', 'vista'])

    # Open the CSV to write new results
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header only if the file is empty
        if df_results.empty:
            writer.writerow(['data', 'patient', 'file', 'frame', 'p1', 'p2', 'vista'])

        # Limit the number of parallel processes
        max_workers = 10  # Adjust this value according to your needs and system capabilities

        df_results['file_no_ext'] = df_results['file'].str.replace('.avi', '', regex=False)

        # Convert the DataFrame to a set for easy comparison
        processed_videos = set(df_results.apply(lambda row: f"{row['patient']}/{row['file']}", axis=1))

        probs_view = pd.read_csv('probs_mayo_v2.csv')

        probs_view = probs_view[~probs_view['fname'].isin(df_results['file_no_ext'])]

        aov_views = probs_view[probs_view['view'].isin(['PLAX', 'PSAX_AO', 'A3C'])]

        # Create a dictionary for quick access to view and period
        fname_info = aov_views.set_index('fname')[['view', 'study']].to_dict('index')
 
        # Create a process pool for parallelization
        with ProcessPoolExecutor(max_workers=max_workers, initializer=initialize_model, initargs=(model_path, device)) as executor:
            futures = []
            
            # Traverse the subfolders and perform inference in a single loop
            for base_dir in base_dirs:

                print(f"Processing directory: {base_dir}")
                patient_folders = os.listdir(base_dir)
                
                for patient in patient_folders:
                    patient_path = os.path.join(base_dir, patient)
                    if '1135' not in patient_path:
                        continue
                    else:
                        print(f"Processing patient: {patient_path}")
                    if os.path.isdir(patient_path):  # Check if it is a directory
                        avi_files = os.listdir(patient_path)
                        for avi in avi_files:
                            if avi.lower().endswith('.avi'):
                                avi_name = os.path.splitext(avi)[0]  # Name without extension
                                
                                if avi_name in fname_info:
                                    avi_path = os.path.join(patient_path, avi)
                                    view = fname_info[avi_name]['view']
                                    # substudy = fname_info[avi_name]['substudy']  # Assuming the images are .npy files
                                    cap = cv2.VideoCapture(avi_path)
                                    frame_count = 0
                                    processed_count = 0

                                    while True:
                                        ret, frame = cap.read()
                                        if not ret:
                                            break

                                        # Process only 1 out of every 4 frames
                                        if frame_count % 4 == 0:
                                            # Convert the frame to a numpy array
                                            image_np = np.array(frame)
                                            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

                                            # Add the task to the process pool
                                            futures.append(executor.submit(
                                                process_frame, view, image_np, patient, avi, frame_count
                                            ))
                                            processed_count += 1

                                        frame_count += 1

            # Write the results as the tasks complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Writing results"):
                result = future.result()
                if result:
                    writer.writerow(result)

    print(f"Inference completed. Results saved to {output_csv}")

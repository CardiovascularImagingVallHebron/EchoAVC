import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.hub
from dataset import EchoNetAoVCalcium
from models import MultiTaskModelBBViewHead
from utils import  evaluate_echonetdynamic, set_seed

def load_parameters(parameters_path):
    parameters = {}
    with open(parameters_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                parameters[key.strip()] = value.strip()
    return parameters


def main(args):
    # Identify parameters file
    parameters_path = os.path.join('parameters.txt')
    assert os.path.exists(parameters_path), f"Parameters file not found at {parameters_path}"

    # Load parameters
    parameters = load_parameters(parameters_path)
    # Extract relevant parameters
    data_dir = parameters.get('data_dir', '')
    dims = int(parameters.get('dims', 224))
    clip_len = int(parameters.get('clip_len', 16))
    num_clips = int(parameters.get('num_clips', 20))
    sampling_rate = int(parameters.get('sampling_rate', 1))
    normalization = parameters.get('normalization', '')
    fc_dropout = float(parameters.get('fc_dropout', 0.0))
    seed = int(parameters.get('seed', 0))
    split = 'test'
    # split = 'tt-test-st'
    # split = 'test'
    set_seed(seed)
    print(dims, clip_len, sampling_rate, normalization, seed, flush=True)
    
    tasks = np.load(os.path.join('content/tasks_v5_qual.npy'), allow_pickle=True)
    for t in tasks:
        print(t.task_name)
    task_inclusion = []

    taskfold = 'calidad'

    task_inclusion.append('view')
    task_inclusion.append('complete-structures')
    task_inclusion.append('calidad')
    task_inclusion.append('combined-quality')

    tasks = [t for t in tasks if t.task_name in task_inclusion]

    # Load test dataset
    print('Preparing test dataset...')

    test_df = pd.read_excel('content/20260227_mimics_iv_quality.xlsx') #COLOR
    
    test_df['FileName'] = test_df['Period'].astype(str) + '___' + test_df['FileName'].astype(str)

    # Visualize the distribution in each fold
    fold = 0
    foldnum = fold + 1

    # Filter the data ensuring that only the correct patients are used
    fold_val_df = test_df
    
    # Check
    print(f"Fold {fold+1}:", flush=True)
    print(f" - Test Patients: {len(fold_val_df)} ", flush=True)
    print("-" * 60, flush=True)
    test_dataset = EchoNetAoVCalcium(
        data_dir=data_dir,
        data_df=fold_val_df,
        tasks=tasks,
        split=split,
        dims=dims,
        clip_len=clip_len,
        sampling_rate=sampling_rate,
        num_clips=num_clips,
        augment=False,
        normalization=normalization
    )

    print(f"Test dataset size: {len(test_dataset)}", flush=True)

    # Create loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2
    )

    # Load model
    print("Loading model...", flush=True)
    model = MultiTaskModelBBViewHead(
        backbone=torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho', force_reload=False, backbone_only=True, clip_len=clip_len),
        tasks=tasks,
        fc_dropout=fc_dropout 
    )

    filtered_state_dict = {}
    checkpoint_path = os.path.join('echoqual.pt')

    # Load the checkpoint
    chkpt = torch.load(checkpoint_path, map_location='cpu')
    print(checkpoint_path, flush=True)
    filtered_state_dict = {}

    for k, v in chkpt['weights'].items():
        # Include backbone layers or allowed heads
        filtered_state_dict[k] = v
            
    # Load the filtered layers
    model.load_state_dict(filtered_state_dict, strict=True)

    model = model.cuda()

    # Define loss functions
    loss_fxns = {}
    for task in tasks:
        if task.task_type == 'regression':
            loss_fxns[task.task_name] = torch.nn.MSELoss()
        elif task.task_type == 'binary_classification':
            loss_fxns[task.task_name] = torch.nn.BCEWithLogitsLoss()
        elif task.task_type == 'multi-class_classification':
            loss_fxns[task.task_name] = torch.nn.CrossEntropyLoss()

    # Evaluate the model
    evaluate_echonetdynamic(
        model=model,
        tasks=tasks,
        loss_fxns=loss_fxns,
        data_loader=test_loader,
        split=split,
        history=None,
        model_dir=args.model_dir_path,
        weights=None,
        amp=False,
        plot_history=False,
        taskname = taskfold,
        fold = foldnum
    )
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir_path', type=str, required=True, help='Path to the model directory')

    args = parser.parse_args()

    print(args)

    main(args)
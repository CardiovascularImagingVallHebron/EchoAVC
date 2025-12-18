import os
import gc
import random
import time
import datetime
import csv

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm

from sklearn import metrics

from sklearn import metrics

from ddp_utils import is_main_process

class Task():
    """Echocardiography interpretation task object."""
    def __init__(self, task_name, task_type, class_names, mean=np.nan):
        self.task_name = task_name
        self.task_type = task_type
        self.class_names = class_names  # ndarray
        self.class_indices = np.arange(class_names.size)
        self.mean = mean

def merge_task_dicts(d):
    merged_dict = {}

    # Iterate through each dictionary in the list
    for dictionary in d:
        # Iterate through keys in the dictionary
        for key, value in dictionary.items():
            if key in merged_dict:
                # Merge lists if key is present in both merged_dict and the current dictionary
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key in merged_dict[key]:
                            merged_dict[key][sub_key] += sub_value
                        else:
                            merged_dict[key][sub_key] = sub_value
                else:
                    merged_dict[key] += value
            else:
                # Otherwise, assign the corresponding dictionary
                merged_dict[key] = value

    return merged_dict

def time_elapsed(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f'{hours:.0f}h:{minutes:.0f}m:{seconds:.0f}s'

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def val_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate_echonetdynamic(model, tasks, loss_fxns, data_loader, split, history, model_dir, weights, amp=False, plot_history=False, taskname=None, fold=0):
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')
    running_loss = 0.

    overall_losses = []
    # task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'yhats_sample': [], 'fnames': [], 'path': [], 'invalid_batches': 0} for task in tasks}
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [],  'fnames': [], 'path': [], 'invalid_batches': 0} for task in tasks}
    
    # Define el archivo CSV y escribe los encabezados
    now = datetime.datetime.now()
    csv_file = os.path.join(model_dir, 'results_plots', f"inference_sample_predictions_{now.strftime('%Y%m%d_%H%M%S')}_{taskname}_{fold}.csv")

    # Encabezados del CSV con cada una de las cabezas de salida
    headers = ["batch", "path", "clip_index"] + [task.task_name for task in tasks]

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    with torch.no_grad():
        for b, batch in pbar:
            # if b <= 361:
            #     continue
            x = batch['x']
            fname = batch['fname']
            path = batch['path']
            out_dicts = []

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                # Forward pass         
                for clip in range(x.shape[2]):
                    out_dict = model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d))    
                    out_dicts.append(out_dict)
                    
                    # Inicializar fila con batch, path y clip_index
                    row = [b, path, clip]

                    # Extraer valores de cada task en orden dinámico
                    for task in tasks:
                        task_name = task.task_name
                        task_type = task.task_type
                        output = out_dict[task_name].cpu().detach().numpy()

                        if task_type == "multi-class_classification":
                            row.append(output[0])  # Suponiendo que es un solo valor
                        else:
                            row.append(output[0][0])  # Extraer solo el primer valor si es regresión o binario

                    # Guardar la fila en el CSV
                    with open(csv_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(row)
                # Forward pass

                out_dicts = [model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d)) for clip in range(x.shape[2])]

                # Compute loss for each task
                losses = []
                for task in tasks:
                    task_name = task.task_name
                    yhat = torch.stack([out_dict[task.task_name] for out_dict in out_dicts], dim=0).mean(dim=0)
                    y = batch[task.task_name].cuda()
                    mask = batch[task.task_name+'_mask'].cuda()


                    # If batch contains *only* missing values for task, then skip loss computation
                    if mask.sum() == 0:
                        task_data[task.task_name]['invalid_batches'] += 1
                        continue

                    # Mask out missing values from loss computation
                    masked_yhat = torch.masked_select(yhat, mask).reshape(-1, yhat.shape[1])
                    masked_y = torch.masked_select(y, mask).reshape(-1, y.shape[1])
                    
                    # Collect (masked) true and predicted labels
                    if task.task_type == 'multi-class_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.softmax(dim=1).numpy(force=True))
                    elif task.task_type == 'binary_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.sigmoid().numpy(force=True))
                    else:
                        task_data[task.task_name]['yhats'].append(masked_yhat.numpy(force=True))

                    task_data[task.task_name]['ys'].append(masked_y.numpy(force=True))

                    # Collect (masked) auxiliary information
                    masked_fname = [f for f, m in zip(fname, mask) if m]
                    masked_paths = [p for p, m in zip(path, mask) if m]
                    task_data[task.task_name]['fnames'].append(masked_fname)
                    task_data[task.task_name]['path'].append(masked_paths)

                    # Store batch-level predictions, fname, and path
                    task_data[task.task_name]['batch_preds'] = task_data[task.task_name].get('batch_preds', [])
                    task_data[task.task_name]['batch_fnames'] = task_data[task.task_name].get('batch_fnames', [])
                    task_data[task.task_name]['batch_paths'] = task_data[task.task_name].get('batch_paths', [])

                    task_data[task.task_name]['batch_preds'].append(masked_yhat.cpu().numpy(force=True))
                    task_data[task.task_name]['batch_fnames'].append(fname)
                    task_data[task.task_name]['batch_paths'].append(path)

                    # For CrossEntropyLoss, target must have shape (N,)
                    if task.task_type == 'multi-class_classification':
                        masked_y = masked_y.squeeze(1)

                    # Compute task loss
                    loss = loss_fxns[task.task_name](masked_yhat, masked_y)
                    # Scale down regression loss based on mean value
                    if task.task_type == 'regression':
                        loss /= task.mean
                    losses.append(loss)

                    # Keep track of task losses for each batch
                    task_data[task.task_name]['losses'].append(loss.item())
                    del loss

                # Compute overall loss
                if len(losses) == 0:
                    continue
                else:
                    loss = sum(losses) / len(losses)
                    del losses
                    
                    # Keep running sum of losses for each batch
                    running_loss += loss.item()
                overall_losses.append(loss.item())

            pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

        # Write combined batch predictions to CSV at the end of each epoch
        combined_data = {}
        for task in tasks:
            task_name = task.task_name
            task_preds = []
            # task_preds_sample = []
            task_fnames = []
            task_paths = []
            
            for fnames, paths, preds in zip(task_data[task_name]['batch_fnames'],
                                           task_data[task_name]['batch_paths'],
                                           task_data[task_name]['batch_preds']):
                for fname, path, pred in zip(fnames, paths, preds):
                    task_fnames.append(fname)
                    task_paths.append(path)
                    task_preds.append(pred.tolist() if isinstance(pred, np.ndarray) else pred)

            # Merge task predictions into a combined dictionary
            for i, (fname, path, pred) in enumerate(zip(task_fnames, task_paths, task_preds)):
                if (path, fname) not in combined_data:
                    combined_data[(path, fname)] = {'path': path, 'fname': fname}
                combined_data[(path, fname)][f'{task_name}_pred'] = pred
                
        # Convert combined data to a pandas DataFrame and save as CSV
        df = pd.DataFrame(list(combined_data.values()))
        now = datetime.datetime.now()

        output_path = os.path.join(model_dir, 'results_plots', f"inference_predictions_{now.strftime('%Y%m%d_%H%M%S')}_{taskname}_{fold}.csv")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    del x, mask, out_dicts
    torch.cuda.empty_cache()
    gc.collect()
    # dist.barrier()

    s = time.perf_counter()

    # Gather task data (dict) across processes and merge by concatenating lists from shared keys together
    task_data = merge_task_dicts([task_data])  # Pasar task_data como una lista

    # Compute and log metrics for each task
    out_str = ''
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        task_data[task.task_name]['invalid_batches'] = np.array(task_data[task.task_name]['invalid_batches']).sum()  # sum reduce (currently list from each process)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            out_str += f'--- {task.task_name} [{task.task_type}] (N=0) ---\n'
            continue

        study_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'fname': np.concatenate(task_data[task.task_name]['fnames'], axis=0),
            'study': [os.path.join(*path.split(os.sep)[-3:-1])  # Extrae los últimos 3 a 2 segmentos relevantes del path
                      for path in np.concatenate(task_data[task.task_name]['path'], axis=0)]
        })

        out_str += f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---\n'

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Initialize performance summary plots
            roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
            pr_fig, pr_ax = plt.subplots(1, 1, figsize=(6, 6))

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    fpr, tpr, _ = metrics.roc_curve(binary_y, yhat[:, class_idx])
                    prs, res, _ = metrics.precision_recall_curve(binary_y, yhat[:, class_idx])

                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                    # Plot class-specific ROC curve
                    roc_ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUROC: {auroc:.3f})')
                    # Plot class-specific PR curve
                    p = pr_ax.plot(res, prs, lw=2, label=f'{class_name} (AP: {ap:.3f})')
                    pr_ax.axhline(y=binary_y.sum()/binary_y.size, color=p[0].get_color(), lw=2, linestyle='--')

                    aurocs.append(auroc)
                    aps.append(ap)
                out_str += f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}\n'

            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            out_str += f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}\n'

            # Keep track of overall mean classification metrics
            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)

            if is_main_process():
                # Save ROC plot
                roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                roc_ax.set_xlim([-0.05, 1.0])
                roc_ax.set_ylim([0.0, 1.05])
                roc_ax.set_xlabel('1 - Specificity', fontsize=13)
                roc_ax.set_ylabel('Sensitivity', fontsize=13)
                roc_ax.legend(loc="lower right", fontsize=11)
                roc_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_roc.png'), dpi=300, bbox_inches='tight')
                roc_fig.clear()
                plt.close(roc_fig)

                # Save PR plot
                pr_ax.set_xlim([-0.05, 1.05])
                pr_ax.set_ylim([-0.05, 1.05])
                pr_ax.set_xlabel('Recall', fontsize=13)
                pr_ax.set_ylabel('Precision', fontsize=13)
                pr_ax.legend(loc="upper right", fontsize=11)
                pr_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_pr.png'), dpi=300, bbox_inches='tight')
                pr_fig.clear()
                plt.close(pr_fig)

        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Initialize performance summary plots
            roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
            pr_fig, pr_ax = plt.subplots(1, 1, figsize=(6, 6))

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

                fpr, tpr, _ = metrics.roc_curve(y, yhat)
                prs, res, _ = metrics.precision_recall_curve(y, yhat)

                # Plot class-specific ROC curve
                roc_ax.plot(fpr, tpr, lw=2, label=f'{task.class_names[1]} (AUROC: {auroc:.3f})')
                # Plot class-specific PR curve
                p = pr_ax.plot(res, prs, lw=2, label=f'{task.class_names[1]} (AP: {ap:.3f})')
                pr_ax.axhline(y=y.sum()/y.size, color=p[0].get_color(), lw=2, linestyle='--')
            
            out_str += f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}\n'

            # Keep track of overall mean classification metrics
            classification_aurocs.append(auroc)
            classification_aps.append(ap)

            if is_main_process():
                # Save ROC plot
                roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                roc_ax.set_xlim([-0.05, 1.0])
                roc_ax.set_ylim([0.0, 1.05])
                roc_ax.set_xlabel('1 - Specificity', fontsize=13)
                roc_ax.set_ylabel('Sensitivity', fontsize=13)
                roc_ax.legend(loc="lower right", fontsize=11)
                roc_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_roc.png'), dpi=300, bbox_inches='tight')
                roc_fig.clear()
                plt.close(roc_fig)

                # Save PR plot
                pr_ax.set_xlim([-0.05, 1.05])
                pr_ax.set_ylim([-0.05, 1.05])
                pr_ax.set_xlabel('Recall', fontsize=13)
                pr_ax.set_ylabel('Precision', fontsize=13)
                pr_ax.legend(loc="upper right", fontsize=11)
                pr_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_pr.png'), dpi=300, bbox_inches='tight')
                pr_fig.clear()
                plt.close(pr_fig)

                # Crear un DataFrame para facilitar el uso de Seaborn
                study_pred_df['yhat'] = study_pred_df['yhat'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
                study_pred_df['y'] = study_pred_df['y'].astype(int)  # Convertir a enteros si es necesario

                plot_df = study_pred_df[['y', 'yhat']]

                # Crear el violin plot
                plt.figure(figsize=(8, 6))
                sns.violinplot(
                    x='y', 
                    y='yhat', 
                    data=plot_df, 
                    palette="muted", 
                    inner="quartile", 
                    cut=0, 
                    hue='y', 
                    legend=False  # Evitar leyendas duplicadas
                )
                sns.stripplot(
                    x='y', 
                    y='yhat', 
                    data=plot_df, 
                    color='k',  # Color negro para los puntos
                    alpha=0.5,  # Transparencia para los puntos
                    dodge=True  # Separar los puntos por clase
                )
                plt.title(f"Distribution of Predictions Task: {task.task_name}")
                plt.xlabel("Class")
                plt.ylabel("Predicted Probability")
                plt.xticks([0, 1], ["0", "1"])
                plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2) 
                plt.ylim(-0.1, 1.1)  # Limitar el eje Y
                plt.grid(axis='y', linestyle='--', alpha=0.6)
                violinplot_path = os.path.join(model_dir, 'results_plots', task.task_name, 'violinplot.png')
                os.makedirs(os.path.dirname(violinplot_path), exist_ok=True)
                plt.savefig(violinplot_path, dpi=300, bbox_inches='tight')
                plt.close()
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)
            
            out_str += f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}\n'

            # Keep track of overall mean classification metrics
            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)

            if is_main_process():
                # Performance evaluation scatter plot
                plt.figure(figsize=(8, 6))
                plt.scatter(y, yhat, alpha=0.6, edgecolor='k')
                plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', linewidth=2)  # Línea y=x
                plt.title(f"{task.task_name} - R^2 = {r2:.3f} | MSE = {mse:.3f} | MAE: {mae:.3f}")
                plt.xlabel("True Values")
                plt.ylabel("Predicted Values")
                plt.grid(alpha=0.5, linestyle='--')
                plt.tight_layout()
                scatterplot_path = os.path.join(model_dir, 'results_plots', task.task_name, f'{task.task_name}_scatterplot.png')
                os.makedirs(os.path.dirname(scatterplot_path), exist_ok=True)
                plt.savefig(scatterplot_path, dpi=300, bbox_inches='tight')
                plt.close()

        if is_main_process():
            # Save video-level and study-level predictions for task
            study_pred_df.to_csv(os.path.join(model_dir, 'results_plots', task.task_name, f'aovcalcium_{split}_{task.task_name}_preds.csv'), index=False)

        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    mean_classification_auroc, mean_classification_ap = np.nanmean(classification_aurocs), np.nanmean(classification_aps)
    mean_regression_r2, mean_regression_mse, mean_regression_mae = np.nanmean(regression_r2s), np.nanmean(regression_mses), np.nanmean(regression_maes)
    out_str += f'[CLASSIFICATION] Mean AUROC: {mean_classification_auroc:.3f} | Mean AP: {mean_classification_ap:.3f} ({classification_aurocs.size} total classification tasks)\n'
    out_str += f'[REGRESSION] Mean R^2: {mean_regression_r2:.3f} | Mean MSE: {mean_regression_mse:.3f} | Mean MAE: {mean_regression_mae:.3f} ({regression_r2s.size} total regression tasks)\n'

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')
        
    # Print overall summary text and save to text file
    print(out_str)
    if is_main_process():
        f = open(os.path.join(model_dir, f'aovcalcium_{split}_summary.txt'), 'w')
        f.write(out_str)
        f.close()
    
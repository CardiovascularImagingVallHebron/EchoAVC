import math

import timm
import torch
import torchvision


class MultiTaskModelBBViewHead(torch.nn.Module):
    """Multi-task model where view_head is applied first and feeds other heads."""
    def __init__(self, backbone, tasks, fc_dropout=0):
        super().__init__()
        self.backbone = backbone
        self.tasks = tasks
        self.classification_heads = {}
        self.regression_heads = {}

        # Buscar y configurar la cabeza 'view'
        self.view_task = next(task for task in tasks if task.task_name == 'view')
        self.view_head = torch.nn.Sequential(
            torch.nn.Dropout(p=fc_dropout),
            torch.nn.Linear(768, self.view_task.class_names.size)
        )

        for task in self.tasks:
            if task.task_name == 'view':
                continue  # Ya está manejada

            input_dim = 768 + self.view_task.class_names.size  # Nueva entrada extendida

            if task.task_type == 'multi-class_classification':
                head = torch.nn.Sequential(
                    torch.nn.Dropout(p=fc_dropout),
                    torch.nn.Linear(input_dim, task.class_names.size)
                )
                self.add_module(f"{task.task_name}_head", head)
                self.classification_heads[task.task_name] = head

            elif task.task_type == 'binary_classification':
                head = torch.nn.Sequential(
                    torch.nn.Dropout(p=fc_dropout),
                    torch.nn.Linear(input_dim, 1)
                )
                self.add_module(f"{task.task_name}_head", head)
                self.classification_heads[task.task_name] = head

            elif task.task_type == 'regression' and task.task_name != 'AoCalcium-score':
                head = torch.nn.Sequential(
                    torch.nn.Dropout(p=fc_dropout),
                    torch.nn.Linear(input_dim, 1)
                )
                self.add_module(f"{task.task_name}_head", head)
                head[-1].bias.data[0] = task.mean
                self.regression_heads[task.task_name] = head

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.forward_features(x)
        out_dict = {}

        # Primero el head de view
        view_output = self.view_head(x)
        out_dict['view'] = view_output

        # Concatenar con el embedding
        x_extended = torch.cat([x, view_output], dim=-1)

        # Clasificación
        for task_name, head in self.classification_heads.items():
            out_dict[task_name] = head(x_extended)

        # Regresión
        for task_name, head in self.regression_heads.items():
            out_dict[task_name] = head(x_extended)

        return out_dict

    
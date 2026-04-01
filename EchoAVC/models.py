import torch


class MultiTaskModelBBScoreHead(torch.nn.Module):
    """Multi-task model based on PanEcho backbone and task list."""

    def __init__(self, backbone, tasks, fc_dropout=0):
        super().__init__()
        self.backbone = backbone  # PanEcho backbone
        self.tasks = tasks
        self.classification_heads = {}
        self.regression_heads = {}

        self.classification_output_size = 0  # Used to compute the total concatenation size

        for task in self.tasks:
            if task.task_type == 'multi-class_classification':
                head = torch.nn.Sequential(
                    torch.nn.Dropout(p=fc_dropout),
                    torch.nn.Linear(768, task.class_names.size)  # 768 is the backbone output dimension
                )
                self.add_module(f"{task.task_name}_head", head)
                self.classification_heads[task.task_name] = head
                self.classification_output_size += task.class_names.size

            elif task.task_type == 'binary_classification':
                head = torch.nn.Sequential(
                    torch.nn.Dropout(p=fc_dropout),
                    torch.nn.Linear(768, 1)
                )
                self.add_module(f"{task.task_name}_head", head)
                self.classification_heads[task.task_name] = head
                self.classification_output_size += 1

            elif task.task_type == 'regression' and task.task_name != 'AoCalcium-score':
                head = torch.nn.Sequential(
                    torch.nn.Dropout(p=fc_dropout),
                    torch.nn.Linear(768, 1)
                )
                self.add_module(f"{task.task_name}_head", head)
                head[-1].bias.data[0] = task.mean  # Initialize bias with the training mean
                self.regression_heads[task.task_name] = head

        # Define the score head with the correct size after knowing classification_output_size
        self.add_module("AoCalcium_score_head", torch.nn.Sequential(
            torch.nn.Dropout(p=fc_dropout),
            torch.nn.Linear(768 + self.classification_output_size, 1)
        ))

    def forward_features(self, x):
        """Extract features from the backbone."""
        return self.backbone(x)

    def forward(self, x):
        """Propagate features through the full model."""
        # Get embeddings from the backbone
        x = self.forward_features(x)
        out_dict = {}
        classification_outputs = []

        # Get classification outputs first
        for task_name, head in self.classification_heads.items():
            class_output = head(x)
            out_dict[task_name] = class_output
            classification_outputs.append(class_output)

        # Concatenate classification outputs with the original embedding
        if classification_outputs:
            classification_outputs = torch.cat(classification_outputs, dim=-1)  # Concatenate on the last dimension
            x_extended = torch.cat([x, classification_outputs], dim=-1)
        else:
            x_extended = x  # No classifications, keep the original input

        # Get the score output after the classifications
        out_dict['AoCalcium-score'] = self.AoCalcium_score_head(x_extended)

        # Get other regression outputs
        for task_name, head in self.regression_heads.items():
            out_dict[task_name] = head(x)

        return out_dict

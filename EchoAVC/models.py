import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import CLIP_LEN, EMB_D, EMB_T, Task, log


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=EMB_D, max_len=EMB_T):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_term = div_term[:pe[:, 1::2].shape[1]]
        pe[:, 1::2] = torch.cos(position * cos_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        t = x.size(1)
        return x + self.pe[:t].unsqueeze(0)


class FrameDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if (not self.training) or self.p <= 0:
            return x
        b, t, d = x.shape
        mask = torch.bernoulli(torch.full((b, t, 1), 1 - self.p, device=x.device))
        return x * mask


class AttentionPool1D(nn.Module):
    def __init__(self, d_model=EMB_D, hidden=128, dropout=0.0):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, h):
        scores = self.attn(h).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        z = torch.bmm(weights.unsqueeze(1), h)
        return z.squeeze(1), weights


class TinyTemporalTransformer(nn.Module):
    def __init__(self, d_in=EMB_D, nhead=2, n_layers=4, ff=768, dropout=0.3, use_posenc=True, frame_dropout=0.1):
        super().__init__()
        self.posenc = PositionalEncoding(d_model=d_in, max_len=EMB_T) if use_posenc else nn.Identity()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_in,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.frame_drop = FrameDropout(p=frame_dropout)
        self.norm = nn.LayerNorm(d_in)

    def forward(self, x):
        h = self.posenc(x)
        h = self.frame_drop(h)
        h = self.enc(h)
        h = self.norm(h)
        return h


class ModelDiagnostic(nn.Module):
    def __init__(
        self,
        d_in=EMB_D,
        nhead=2,
        n_layers=4,
        ff=512,
        dropout=0.3,
        frame_dropout=0.1,
        attn_hidden=64,
        head_hidden=64,
        head_dropout=0.4,
    ):
        super().__init__()
        self.backbone = TinyTemporalTransformer(
            d_in=d_in,
            nhead=nhead,
            n_layers=n_layers,
            ff=ff,
            dropout=dropout,
            use_posenc=True,
            frame_dropout=frame_dropout,
        )
        self.pool = AttentionPool1D(d_model=d_in, hidden=attn_hidden, dropout=dropout)

        self.h_reg = nn.Sequential(
            nn.Linear(d_in, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )
        self.h_b0 = nn.Sequential(
            nn.Linear(d_in, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        h = self.backbone(x)
        z, _ = self.pool(h)
        y_reg = self.h_reg(z).squeeze(-1)
        y_b0 = self.h_b0(z).squeeze(-1)
        return y_reg, y_b0


class MultiTaskModelBBScoreHead(torch.nn.Module):
    """Multi-task model based on PanEcho backbone and task list."""

    def __init__(self, backbone, tasks, fc_dropout=0):
        super().__init__()
        self.backbone = backbone
        self.tasks = tasks
        self.classification_heads = {}
        self.regression_heads = {}
        self.classification_output_size = 0

        for task in self.tasks:
            if task.task_type == 'multi-class_classification':
                head = torch.nn.Sequential(
                    torch.nn.Dropout(p=fc_dropout),
                    torch.nn.Linear(768, task.class_names.size)
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
                head[-1].bias.data[0] = task.mean
                self.regression_heads[task.task_name] = head

        self.add_module("AoCalcium_score_head", torch.nn.Sequential(
            torch.nn.Dropout(p=fc_dropout),
            torch.nn.Linear(768 + self.classification_output_size, 1)
        ))

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.forward_features(x)
        out_dict = {}
        classification_outputs = []

        for task_name, head in self.classification_heads.items():
            class_output = head(x)
            out_dict[task_name] = class_output
            classification_outputs.append(class_output)

        if classification_outputs:
            classification_outputs = torch.cat(classification_outputs, dim=-1)
            x_extended = torch.cat([x, classification_outputs], dim=-1)
        else:
            x_extended = x

        out_dict['AoCalcium-score'] = self.AoCalcium_score_head(x_extended)

        for task_name, head in self.regression_heads.items():
            out_dict[task_name] = head(x)

        return out_dict


def load_model(tasks_path: str, checkpoint_path: str, device: torch.device):
    tasks = np.load(tasks_path, allow_pickle=True)

    task_inclusion = [
        'AoCalcium-presence',
        'AoCalcium-score',
        'AoCalcium-multiclass-low',
        'AoCalcium-cutoff-low',
        'AoCalcium-multiclass-high',
        'AoCalcium-cutoff-high',
    ]
    tasks = [t for t in tasks if t.task_name in task_inclusion]

    log("Loading PanEcho model (backbone_only, clip_len=32) ...")
    model = MultiTaskModelBBScoreHead(
        backbone=torch.hub.load(
            'CarDS-Yale/PanEcho',
            'PanEcho',
            force_reload=False,
            backbone_only=True,
            clip_len=CLIP_LEN
        ),
        tasks=tasks,
        fc_dropout=0.4
    )

    chkpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = chkpt['weights']
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    return model

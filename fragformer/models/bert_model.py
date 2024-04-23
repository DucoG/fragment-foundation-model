import pandas as pd
import torch
import torch.nn as nn
import lightning as L
import torch
from typing import List, Optional, Union
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, dim_feedforward, nheads, n_blocks, dropout):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True),

            num_layers=n_blocks,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False)

    def forward(self, tokens, segments):
        x = self.token_embedding(tokens) + self.segment_embedding(segments) + self.position_embedding
        x = self.encoder(x)
        return x

class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model),
                                 nn.GELU(),
                                 nn.LayerNorm(d_model),
                                 nn.Linear(d_model, vocab_size))

    def forward(self, X, mlm_pos_and_y):
        # mlm_pos_and_y shape: (num_mlm_preds_for_batch, 3) 
        # (batch_idx, pred_positions, mlm_Y)
        batch_size, seq_len, d_model = X.shape
        preds_per_sequence = mlm_pos_and_y.shape[0] // batch_size

        masked_X = X[mlm_pos_and_y[:, 0], mlm_pos_and_y[:, 1]].reshape((batch_size, preds_per_sequence, d_model))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

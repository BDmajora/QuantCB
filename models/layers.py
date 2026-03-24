import torch
import torch.nn as nn

class QuantCB_FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 1. Expand: From d_model to a larger hidden dimension (usually 4x)
        self.w_1 = nn.Linear(d_model, d_ff)
        # 2. Activation: GELU is standard for high-performance LLMs
        self.activation = nn.GELU()
        # 3. Contract: Back to d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        # 4. Dropout for regularization during training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Seq, d_model)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x
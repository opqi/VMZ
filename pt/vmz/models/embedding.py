import torch.nn as nn
import torch


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float().to(device='cuda')
        pe.require_grad = True
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe)
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BERTEmbedding(nn.Module):
    def __init__(self, input_dim, max_len, dropout=0.1):
        super().__init__()
        self.learnedPosition = LearnedPositionalEmbedding(
            d_model=input_dim, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        x = self.learnedPosition(sequence) + sequence
        return self.dropout(x)

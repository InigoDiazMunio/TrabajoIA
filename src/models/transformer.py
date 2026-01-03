"""
Modelos Transformer para tareas de NLP
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Codificación posicional para Transformers."""
    
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SmallTransformer(nn.Module):
    """Transformer pequeño para clasificación de texto (IMDB)."""
    
    def __init__(self, vocab_size=30522, d_model=256, nhead=4, num_layers=4, num_classes=2, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        self.d_model = d_model

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x[:, 0, :]  # Primer token (CLS-like)
        return self.classifier(x)


class MediumTransformer(nn.Module):
    """Transformer mediano para language modeling (WikiText)."""
    
    def __init__(self, vocab_size=50257, d_model=512, nhead=8, num_layers=8, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.embedding.weight  # Tie weights
        self.d_model = d_model

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.lm_head(x)
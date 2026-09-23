import torch
import torch.nn as nn

from weird_ai.transformer import TransformerBlock
from weird_ai.layer_norm import LayerNorm


class WeirdAIModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length=64,
        emb_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        qkv_bias=False,
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    emb_dim=emb_dim,
                    context_length=context_length,
                    num_heads=num_heads,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = LayerNorm(emb_dim)
        self.output_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x):
        batch_size, sequence_length = x.shape

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds "
                f"context length ({self.context_length})."
            )

        token_embeddings = self.token_embedding(x)

        position_ids = torch.arange(
            sequence_length,
            device=x.device,
        )

        position_embeddings = self.position_embedding(position_ids)

        x = token_embeddings + position_embeddings
        x = self.dropout(x)

        x = self.transformer_blocks(x)
        x = self.final_norm(x)

        logits = self.output_head(x)

        return logits
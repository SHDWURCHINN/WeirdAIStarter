from torch import nn as nn

from .layer_norm import LayerNorm
from .feed_forward import FeedForward
from .attention import CausalAttention

class TransformerBlock(nn.Module):
    # TODO 
    # Create a TransformerBlock class, inheriting from nn.Module
    # using 
    #  - LayerNorm
    #  - SelfAttention from previous assignment
    #  - FeedForward
    #  - Residual connections
    def __init__(self, emb_dim, context_length, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.att = CausalAttention(
            embedding_dim=emb_dim,
            output_dim=emb_dim,
            context_length=context_length,
            dropout=dropout,
            qkv_bias=qkv_bias
        )

        self.ff = FeedForward(emb_dim)

        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)

        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x):

        
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    

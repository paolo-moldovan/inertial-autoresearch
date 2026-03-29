"""Multi-head attention wrapper for future extensibility."""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Thin wrapper around ``nn.MultiheadAttention``.

    Provides a consistent interface and a single place to swap in
    alternative attention implementations (e.g. flash attention) later.

    Args:
        d_model: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass delegating to the underlying MultiheadAttention.

        Args:
            query: Query tensor of shape (batch, seq_len, d_model).
            key: Key tensor of shape (batch, seq_len, d_model).
            value: Value tensor of shape (batch, seq_len, d_model).
            attn_mask: Optional attention mask.
            key_padding_mask: Optional key padding mask.

        Returns:
            Tuple of (attention output, attention weights).
        """
        output, weights = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        return cast(Tensor, output), cast(Tensor | None, weights)

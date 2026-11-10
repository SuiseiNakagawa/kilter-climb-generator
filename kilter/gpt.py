"""Climb generator: a small GPT that writes holds one token at a time.

Input is a climb as tokens (kilter/tokenizer.py):

    <bos> angle=40 V3 1096:foot 1113:foot ... 1392:finish <eos>

The model sees everything to the left and predicts the next token. It is
trained only on the hold tokens and <eos>: the angle and grade are given, not
predicted, so it learns p(holds | angle, grade). At generation time you write
"<bos> angle=30 V3" and it continues.

Architecture: a standard decoder-only transformer (as in GPT-2 / nanoGPT),
just very small: token + position embeddings, a few blocks of causal
self-attention and an MLP, then a linear layer back to the vocabulary that
shares its weights with the token embedding.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Each position looks only at itself and earlier positions. Padding is
    always at the end of a sequence, so the causal mask already keeps real
    tokens from seeing it; no separate padding mask is needed."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ClimbGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 48, dim: int = 192,
                 layers: int = 4, heads: int = 4, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.block_size, self.pad_id = block_size, pad_id
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(block_size, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(dim, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx [batch, time] -> logits [batch, time, vocab]"""
        t = idx.shape[1]
        x = self.drop(self.tok(idx) + self.pos(torch.arange(t, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    def loss(self, ids: torch.Tensor, prompt_len: int = 3) -> torch.Tensor:
        """Next-token cross-entropy on the hold tokens and <eos> only."""
        logits = self(ids[:, :-1])
        targets = ids[:, 1:].clone()
        targets[:, : prompt_len - 1] = -100      # don't learn to predict angle and grade
        targets[targets == self.pad_id] = -100   # nor padding
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

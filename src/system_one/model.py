"""Compact encoder: one forward pass, typed heads, no text generation."""

from __future__ import annotations

import re

import torch
from torch import nn

from system_one.schema import TOOLS

PAD = 0
CLS = 1


def stable_hash(text: str, modulo: int) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value % modulo


def tokenize(text: str, vocab: int, max_len: int) -> list[int]:
    """Token ids are 2..vocab-1. Id 0 is padding and id 1 is the CLS pool token."""
    pieces = re.findall(r"[A-Za-z0-9_]+|[^\s]", text)
    limit = max_len - 1
    return [stable_hash(piece.lower(), vocab - 2) + 2 for piece in pieces[:limit]]


class Encoder(nn.Module):
    def __init__(self, vocab: int, d_model: int, layers: int, heads: int, max_len: int, dropout: float):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.position = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if token_ids.size(1) + 1 > self.max_len:
            raise ValueError("token row does not leave room for the CLS token")
        batch = token_ids.size(0)
        cls = torch.full((batch, 1), CLS, dtype=torch.long, device=token_ids.device)
        cls_mask = torch.ones(batch, 1, dtype=torch.bool, device=token_ids.device)
        token_ids = torch.cat([cls, token_ids], dim=1)
        mask = torch.cat([cls_mask, mask], dim=1)
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        hidden = self.embed(token_ids) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        return self.norm(hidden[:, 0])


class DecisionModel(nn.Module):
    """Map state plus the fixed question block to all answers in one pass."""

    def __init__(
        self,
        vocab: int = 4096,
        d_model: int = 64,
        layers: int = 2,
        heads: int = 4,
        max_len: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.encoder = Encoder(vocab, d_model, layers, heads, max_len, dropout)
        self.tool = nn.Linear(d_model, len(TOOLS))
        self.live = nn.Linear(d_model, 1)
        self.destructive = nn.Linear(d_model, 1)
        self.risk = nn.Linear(d_model, 3)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.encoder(token_ids, mask)
        return {
            "tool": self.tool(pooled),
            "live": self.live(pooled).squeeze(-1),
            "destructive": self.destructive(pooled).squeeze(-1),
            "risk": self.risk(pooled),
        }


class SequentialJudge(nn.Module):
    """Autoregressive stand-in for an LLM judge that emits a JSON decision.

    Each field is one decode step conditioned on the previous field. The final
    JSON string is rendered in code. This is not a frontier chat model.
    """

    def __init__(
        self,
        vocab: int = 4096,
        d_model: int = 64,
        layers: int = 2,
        heads: int = 4,
        max_len: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.encoder = Encoder(vocab, d_model, layers, heads, max_len, dropout)
        self.step = nn.GRUCell(d_model, d_model)
        self.tool_emb = nn.Embedding(len(TOOLS), d_model)
        self.live_emb = nn.Embedding(2, d_model)
        self.destructive_emb = nn.Embedding(2, d_model)
        self.tool = nn.Linear(d_model, len(TOOLS))
        self.live = nn.Linear(d_model, 2)
        self.destructive = nn.Linear(d_model, 2)
        self.risk = nn.Linear(d_model, 3)

    def teacher_logits(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        tool: torch.Tensor,
        live: torch.Tensor,
        destructive: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(token_ids, mask)
        tool_logits = self.tool(hidden)
        hidden = self.step(self.tool_emb(tool), hidden)
        live_logits = self.live(hidden)
        hidden = self.step(self.live_emb(live), hidden)
        destructive_logits = self.destructive(hidden)
        hidden = self.step(self.destructive_emb(destructive), hidden)
        risk_logits = self.risk(hidden)
        return {
            "tool": tool_logits,
            "live": live_logits,
            "destructive": destructive_logits,
            "risk": risk_logits,
        }

    def generate(self, token_ids: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(token_ids, mask)
        tool_logits = self.tool(hidden)
        tool = tool_logits.argmax(dim=-1)
        hidden = self.step(self.tool_emb(tool), hidden)
        live_logits = self.live(hidden)
        live = live_logits.argmax(dim=-1)
        hidden = self.step(self.live_emb(live), hidden)
        destructive_logits = self.destructive(hidden)
        destructive = destructive_logits.argmax(dim=-1)
        hidden = self.step(self.destructive_emb(destructive), hidden)
        risk_logits = self.risk(hidden)
        return {
            "tool": tool_logits,
            "live": live_logits,
            "destructive": destructive_logits,
            "risk": risk_logits,
        }

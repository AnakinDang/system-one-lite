"""Pad hashed tokens into a batch."""

from __future__ import annotations

import torch

from system_one.data import Example
from system_one.model import tokenize


def collate(examples: list[Example], vocab: int, max_len: int) -> dict[str, torch.Tensor | list[str]]:
    rows = [tokenize(example.text, vocab, max_len) for example in examples]
    if any(len(row) == 0 for row in rows):
        raise ValueError("refusing to collate an empty tokenization")
    width = max(len(row) for row in rows)
    token_ids = torch.zeros(len(examples), width, dtype=torch.long)
    mask = torch.zeros(len(examples), width, dtype=torch.bool)
    for index, row in enumerate(rows):
        token_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row)] = True
    return {
        "ids": token_ids,
        "mask": mask,
        "tool": torch.tensor([example.tool for example in examples], dtype=torch.long),
        "live": torch.tensor([example.live for example in examples], dtype=torch.long),
        "destructive": torch.tensor([example.destructive for example in examples], dtype=torch.long),
        "risk": torch.tensor([example.risk for example in examples], dtype=torch.long),
        "family": [example.family for example in examples],
    }


def iterate_batches(examples: list[Example], batch_size: int, vocab: int, max_len: int, shuffle: bool, rng):
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [examples[index] for index in order[start : start + batch_size]]
        yield collate(chunk, vocab, max_len)

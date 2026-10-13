"""Proper scoring rules. These are not TypeSafe's unpublished RLCD update."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multiclass_brier(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=-1)
    target = F.one_hot(labels, probabilities.size(-1)).to(probabilities.dtype)
    return (probabilities - target).pow(2).sum(dim=-1).mean()


def binary_brier(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    return (probabilities - labels.to(probabilities.dtype)).pow(2).mean()


def decision_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    tool = F.cross_entropy(outputs["tool"], batch["tool"])
    risk = F.cross_entropy(outputs["risk"], batch["risk"])
    live = F.binary_cross_entropy_with_logits(outputs["live"], batch["live"].float())
    destructive = F.binary_cross_entropy_with_logits(
        outputs["destructive"], batch["destructive"].float()
    )
    brier = (
        multiclass_brier(outputs["tool"], batch["tool"])
        + multiclass_brier(outputs["risk"], batch["risk"])
        + binary_brier(outputs["live"], batch["live"])
        + binary_brier(outputs["destructive"], batch["destructive"])
    )
    return tool + risk + live + destructive + 0.25 * brier


def judge_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        F.cross_entropy(outputs["tool"], batch["tool"])
        + F.cross_entropy(outputs["live"], batch["live"])
        + F.cross_entropy(outputs["destructive"], batch["destructive"])
        + F.cross_entropy(outputs["risk"], batch["risk"])
    )

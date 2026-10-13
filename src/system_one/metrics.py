"""Accuracy, proper scores, and expected calibration error."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    if confidence.numel() == 0:
        return 0.0
    edges = torch.linspace(0, 1, bins + 1, device=confidence.device)
    total = confidence.numel()
    error = confidence.new_zeros(())
    for index in range(bins):
        if index == 0:
            selected = confidence <= edges[1]
        else:
            selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        count = int(selected.sum())
        if count == 0:
            continue
        gap = (confidence[selected].mean() - correct[selected].float().mean()).abs()
        error = error + gap * (count / total)
    return float(error)


def classification_stats(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = logits.softmax(dim=-1)
    chosen = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = chosen == labels
    return {
        "accuracy": float(correct.float().mean()),
        "nll": float(F.cross_entropy(logits, labels)),
        "brier": float((probabilities - F.one_hot(labels, probabilities.size(-1)).float()).pow(2).sum(-1).mean()),
        "ece": expected_calibration_error(confidence, correct),
    }


def binary_stats(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = logits.sigmoid()
    chosen = probabilities >= 0.5
    confidence = torch.maximum(probabilities, 1 - probabilities)
    correct = chosen == labels.bool()
    return {
        "accuracy": float(correct.float().mean()),
        "nll": float(F.binary_cross_entropy_with_logits(logits, labels.float())),
        "brier": float((probabilities - labels.float()).pow(2).mean()),
        "ece": expected_calibration_error(confidence, correct),
    }


def scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / temperature


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, binary: bool = False) -> float:
    best_temperature = 1.0
    best_loss = float("inf")
    for step in range(15, 61):
        temperature = step / 20
        scaled = scale_logits(logits, temperature)
        if binary:
            loss = float(F.binary_cross_entropy_with_logits(scaled, labels.float()))
        else:
            loss = float(F.cross_entropy(scaled, labels))
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return best_temperature

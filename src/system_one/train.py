"""Train the one-pass decision model and the sequential JSON judge."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from system_one.batch import collate, iterate_batches
from system_one.data import Example, make_split
from system_one.loss import decision_loss, judge_loss
from system_one.metrics import binary_stats, classification_stats, fit_temperature, scale_logits
from system_one.model import DecisionModel, SequentialJudge
from system_one.schema import TOOLS, choice_from_probs, noul_from_prob, score_from_probs

ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def collect(model, examples: list[Example], batch_size: int, device: torch.device, sequential: bool) -> dict:
    model.eval()
    parts: list[dict[str, torch.Tensor]] = []
    families: list[str] = []
    labels: list[dict[str, torch.Tensor]] = []
    for batch in iterate_batches(examples, batch_size, model.vocab, model.max_len, False, random.Random(0)):
        families.extend(batch["family"])
        hosted = move_batch(batch, device)
        if sequential:
            outputs = model.generate(hosted["ids"], hosted["mask"])
        else:
            outputs = model(hosted["ids"], hosted["mask"])
        parts.append({key: value.detach().cpu() for key, value in outputs.items()})
        labels.append(
            {
                "tool": batch["tool"],
                "live": batch["live"],
                "destructive": batch["destructive"],
                "risk": batch["risk"],
            }
        )
    merged = {key: torch.cat([part[key] for part in parts], dim=0) for key in parts[0]}
    gold = {key: torch.cat([part[key] for part in labels], dim=0) for key in labels[0]}
    merged["family"] = families
    merged["gold"] = gold
    return merged


def apply_temperatures(outputs: dict, temperatures: dict[str, float]) -> dict:
    return {
        "tool": scale_logits(outputs["tool"], temperatures["tool"]),
        "live": scale_logits(outputs["live"], temperatures["live"]),
        "destructive": scale_logits(outputs["destructive"], temperatures["destructive"]),
        "risk": scale_logits(outputs["risk"], temperatures["risk"]),
        "family": outputs["family"],
        "gold": outputs["gold"],
    }


def fit_all(outputs: dict, binary_live: bool) -> dict[str, float]:
    gold = outputs["gold"]
    temperatures = {
        "tool": fit_temperature(outputs["tool"], gold["tool"]),
        "risk": fit_temperature(outputs["risk"], gold["risk"]),
    }
    if binary_live:
        temperatures["live"] = fit_temperature(outputs["live"], gold["live"], binary=True)
        temperatures["destructive"] = fit_temperature(outputs["destructive"], gold["destructive"], binary=True)
    else:
        temperatures["live"] = fit_temperature(outputs["live"], gold["live"])
        temperatures["destructive"] = fit_temperature(outputs["destructive"], gold["destructive"])
    return temperatures


def score_outputs(outputs: dict, binary_live: bool) -> dict[str, dict[str, float]]:
    gold = outputs["gold"]
    if binary_live:
        live = binary_stats(outputs["live"], gold["live"])
        destructive = binary_stats(outputs["destructive"], gold["destructive"])
    else:
        live = classification_stats(outputs["live"], gold["live"])
        destructive = classification_stats(outputs["destructive"], gold["destructive"])
    return {
        "tool": classification_stats(outputs["tool"], gold["tool"]),
        "needs_live_data": live,
        "is_destructive": destructive,
        "risk": classification_stats(outputs["risk"], gold["risk"]),
    }


def family_accuracy(outputs: dict) -> dict[str, dict[str, float]]:
    chosen = outputs["tool"].argmax(dim=-1)
    gold = outputs["gold"]["tool"]
    report = {}
    for name in sorted(set(outputs["family"])):
        index = [i for i, family in enumerate(outputs["family"]) if family == name]
        selector = torch.tensor(index, dtype=torch.long)
        correct = chosen[selector] == gold[selector]
        report[name] = {"n": len(index), "accuracy": float(correct.float().mean())}
    return report


def majority_accuracy(train: list[Example], test: list[Example]) -> float:
    counts = [0, 0, 0, 0, 0]
    for example in train:
        counts[example.tool] += 1
    majority = max(range(5), key=lambda index: counts[index])
    hits = sum(example.tool == majority for example in test)
    return hits / len(test)


def json_text(tool: int, live: int, destructive: int, risk: int) -> str:
    payload = {
        "tool": TOOLS[tool],
        "needs_live_data": bool(live),
        "is_destructive": bool(destructive),
        "risk": risk,
    }
    return json.dumps(payload, separators=(",", ":"))


def latency(model, examples: list[Example], device: torch.device, sequential: bool, n: int = 200) -> dict[str, float]:
    model.eval()
    sample = examples[:n]
    prepared = [collate([example], model.vocab, model.max_len) for example in sample]

    def run_one(index: int) -> None:
        batch = move_batch(prepared[index], device)
        with torch.inference_mode():
            if sequential:
                model.generate(batch["ids"], batch["mask"])
            else:
                model(batch["ids"], batch["mask"])

    for index in range(min(20, len(prepared))):
        run_one(index)
    elapsed = []
    for index in range(len(prepared)):
        start = time.perf_counter()
        run_one(index)
        elapsed.append((time.perf_counter() - start) * 1000)
    elapsed.sort()
    return {
        "n": len(elapsed),
        "p50_ms": elapsed[len(elapsed) // 2],
        "p95_ms": elapsed[min(len(elapsed) - 1, int(len(elapsed) * 0.95))],
        "mean_ms": sum(elapsed) / len(elapsed),
    }


def train_one(model, examples, epochs, batch_size, lr, device, sequential, seed) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    rng = random.Random(seed)
    model.train()
    for _ in range(epochs):
        total = 0.0
        seen = 0
        for batch in iterate_batches(examples, batch_size, model.vocab, model.max_len, True, rng):
            hosted = move_batch(batch, device)
            optimizer.zero_grad()
            if sequential:
                outputs = model.teacher_logits(
                    hosted["ids"], hosted["mask"], hosted["tool"], hosted["live"], hosted["destructive"]
                )
                loss = judge_loss(outputs, hosted)
            else:
                outputs = model(hosted["ids"], hosted["mask"])
                loss = decision_loss(outputs, hosted)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * hosted["tool"].size(0)
            seen += hosted["tool"].size(0)
        losses.append(total / seen)
        print(f"  epoch {len(losses)} loss {losses[-1]:.4f}", flush=True)
    return losses


def preview(outputs: dict, examples: list[Example], binary_live: bool) -> dict:
    if binary_live:
        live_prob = float(outputs["live"][0].sigmoid())
        destructive_prob = float(outputs["destructive"][0].sigmoid())
    else:
        live_prob = float(outputs["live"][0].softmax(dim=-1)[1])
        destructive_prob = float(outputs["destructive"][0].softmax(dim=-1)[1])
    decision = {
        "next_tool": choice_from_probs(TOOLS, outputs["tool"][0].softmax(dim=-1).tolist()),
        "needs_live_data": noul_from_prob(live_prob),
        "is_destructive": noul_from_prob(destructive_prob),
        "risk": score_from_probs(outputs["risk"][0].softmax(dim=-1).tolist()),
    }
    example = examples[0]
    return {
        "state": example.text.split("state:\n", 1)[1],
        "label": {
            "tool": TOOLS[example.tool],
            "needs_live_data": bool(example.live),
            "is_destructive": bool(example.destructive),
            "risk": example.risk,
        },
        "next_tool": decision["next_tool"].choice,
        "tool_confidence": decision["next_tool"].confidence,
        "needs_live_data": decision["needs_live_data"].noul,
        "is_destructive": decision["is_destructive"].noul,
        "risk_score": decision["risk"].score,
        "risk_confidence": decision["risk"].confidence,
    }


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    train = make_split(args.train_size, args.seed)
    calibration = make_split(args.cal_size, args.seed + 1)
    test = make_split(args.test_size, args.seed + 2)
    kwargs = dict(vocab=args.vocab, d_model=args.d_model, layers=args.layers, heads=4, max_len=args.max_len, dropout=0.1)
    decision = DecisionModel(**kwargs).to(device)
    judge = SequentialJudge(**kwargs).to(device)

    print("decision model", flush=True)
    decision_losses = train_one(decision, train, args.epochs, args.batch_size, args.lr, device, False, args.seed)
    print("sequential judge", flush=True)
    judge_losses = train_one(judge, train, args.epochs, args.batch_size, args.lr, device, True, args.seed)

    decision_cal = collect(decision, calibration, args.batch_size, device, False)
    judge_cal = collect(judge, calibration, args.batch_size, device, True)
    decision_temperature = fit_all(decision_cal, True)
    judge_temperature = fit_all(judge_cal, False)

    decision_test = collect(decision, test, args.batch_size, device, False)
    judge_test = collect(judge, test, args.batch_size, device, True)
    decision_raw = score_outputs(decision_test, True)
    judge_raw = score_outputs(judge_test, False)
    decision_scaled = score_outputs(apply_temperatures(decision_test, decision_temperature), True)
    judge_scaled = score_outputs(apply_temperatures(judge_test, judge_temperature), False)

    rendered = [
        json_text(int(tool), int(live), int(destructive), int(risk))
        for tool, live, destructive, risk in zip(
            judge_test["tool"].argmax(-1).tolist(),
            judge_test["live"].argmax(-1).tolist(),
            judge_test["destructive"].argmax(-1).tolist(),
            judge_test["risk"].argmax(-1).tolist(),
        )
    ]
    report = {
        "seed": args.seed,
        "torch": torch.__version__,
        "device": args.device,
        "train_size": args.train_size,
        "calibration_size": args.cal_size,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "d_model": args.d_model,
        "layers": args.layers,
        "majority_tool_accuracy": majority_accuracy(train, test),
        "decision_train_loss": decision_losses,
        "judge_train_loss": judge_losses,
        "decision_temperature": decision_temperature,
        "judge_temperature": judge_temperature,
        "decision_before_temperature": decision_raw,
        "decision_after_temperature": decision_scaled,
        "judge_before_temperature": judge_raw,
        "judge_after_temperature": judge_scaled,
        "decision_tool_by_family": family_accuracy(decision_test),
        "latency_ms": {
            "decision": latency(decision, test, device, False),
            "judge": latency(judge, test, device, True),
        },
        "cost": {
            "decision_forward_passes": 1,
            "decision_output_tokens": 0,
            "judge_decode_steps": 4,
            "judge_mean_json_chars": sum(len(text) for text in rendered) / len(rendered),
        },
        "preview": preview(apply_temperatures(decision_test, decision_temperature), test, True),
        "scope": {
            "architecture": "hashed-token transformer encoder, CLS pooling, parallel Choice Score and Noul heads",
            "objective": "cross-entropy and binary cross-entropy plus 0.25 times Brier score",
            "not_included": "TypeSafe Jev weights, unpublished architecture, and the unpublished RLCD update",
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the System One lite tool router")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--cal-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tool_routing.json")
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    tool = report["decision_after_temperature"]["tool"]
    judge = report["judge_after_temperature"]["tool"]
    print(
        json.dumps(
            {
                "decision_tool": tool,
                "judge_tool": judge,
                "latency_ms": report["latency_ms"],
                "cost": report["cost"],
                "wrote": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

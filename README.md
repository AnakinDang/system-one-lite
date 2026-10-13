# system-one-lite

Open, runnable reproduction of the public [System One](https://docs.typesafe.ai/concepts/system-one) decision contract, measured on a synthetic agent tool-routing task.

TypeSafe's Jev is the first System One model. This repository implements the published interface: one forward pass reads a state and returns typed Choice, Score, and Noul answers with probabilities. Confidence follows the formulas on TypeSafe's [confidence](https://docs.typesafe.ai/confidence.md) page. TypeSafe has not published Jev's weights, model architecture, or the RLCD training update. The objective in this repo is a proper scoring rule, written out below.

## Task

The state is a short agent request. Four questions are fixed:

| Question | Primitive | Answers |
| --- | --- | --- |
| `next_tool` | Choice | `search`, `calculator`, `code`, `ask_user`, `finish` |
| `needs_live_data` | Noul | probability the request needs information outside the state |
| `is_destructive` | Noul | probability execution would change external state |
| `risk` | Score | `0` read-only, `1` local write, `2` external side effect |

Labels are drawn from 10 scenario families with known frequencies. Ambiguous families, such as "write a script to calculate this", are mixed on purpose. A fixed schema does not accept an arbitrary new question list at runtime; the heads are the schema.

`ToolRouteDecision.action()` keeps the typed answer and decides what code does with it: confidence below 0.45 escalates, a destructive call below 0.85 confidence asks for confirmation, and a clear safe call returns the tool name. A legal enum can still be the wrong tool.

## Model

Both systems use a 2-layer transformer encoder, width 64, 4 heads, and a stable hash tokenizer. There is no pretrained language model.

- **Decision model.** A CLS token is pooled once. Choice and Score are softmax heads. Each Noul is a sigmoid. Nothing is decoded into text.
- **Sequential judge.** The same encoder, then four GRU steps that emit tool, live-data, destructive, and risk, conditioned on the previous choice. Code renders those fields as JSON. This stands in for a constrained LLM judge. It is not a call to a frontier model.

## Objective

For the decision model:

```text
L = CE(tool) + CE(risk) + BCE(live) + BCE(destructive)
    + 0.25 * (Brier(tool) + Brier(risk) + Brier(live) + Brier(destructive))
```

Cross-entropy and Brier score are strictly proper scoring rules: the probability that minimizes them is the conditional probability of the outcome. That is the public calibration standard (a predicted 0.8 should be right about 80% of the time, as a group). It is not an implementation of RLCD. RLCD would need a published policy, reward, and gradient estimator; those are not in the public docs.

After training, each head gets a temperature chosen on a calibration split by minimizing negative log-likelihood. Temperature does not change argmax, so accuracy stays put. The held-out test split is not used to pick the temperature.

## Results

Seed 42. Train / calibration / test = 5000 / 1000 / 1500. CPU, PyTorch 2.14.1, batch size 1, 200 timed examples after 20 warmup calls. Full numbers are in [`results/tool_routing.json`](results/tool_routing.json).

| | Decision model | Sequential judge | Train-set majority |
| --- | ---: | ---: | ---: |
| Tool accuracy | 68.6% | 68.7% | 31.8% |
| Tool NLL, after temperature | 0.983 | 0.977 | |
| Tool ECE, before → after temperature | 0.048 → 0.029 | 0.041 → 0.039 | |
| `needs_live_data` accuracy | 86.1% | 85.9% | |
| `is_destructive` accuracy | 93.2% | 93.2% | |
| Risk accuracy | 74.0% | 73.5% | |
| p50 latency | 0.35 ms | 0.39 ms | |
| Generated text | none | 4 steps, ~73 JSON characters | |

Tool accuracy is unchanged by temperature, and tool ECE drops from 0.048 to 0.029. The live-data temperature selected on the calibration split (0.9) moves test ECE from 0.015 to 0.021, so a calibration fit is checked on a separate test split before being trusted.

At this width the encoder dominates the clock, so four extra decode steps are only about 1.1× the wall time. The contract difference is still there: the decision model bills no output tokens. This run does not measure a frontier LLM.

On `code_or_calc` and `unclear`, accuracy sits near the frequency of the most common label. Those families are mixed by construction. In the saved preview, the state "no further action for leo" is a `done` scenario whose live-data rate is 0.03. That particular draw was the rare positive label, and the model returned 0.027. One outcome does not make a grouped probability wrong.

## Run

Python 3.12. The training entry point reads `PYTHONPATH=src`.

```bash
uv venv --python 3.12
uv pip install torch pytest
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m system_one.train
```

`system_one.train` rewrites `results/tool_routing.json`. Latency will move across machines; the seed fixes the data and the initialization.

## Scope

The confidence helpers are the published Choice and Score formulas, covered by unit tests against the documented examples. The benchmark text is synthetic English. Hash embeddings do not provide general language understanding, and the measured latency is not Jev's published 70–500 ms serving claim.

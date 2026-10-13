import torch

from system_one.batch import collate
from system_one.data import FAMILIES, make_split
from system_one.loss import decision_loss
from system_one.metrics import fit_temperature
from system_one.model import DecisionModel, tokenize


def test_splits_are_stable():
    first = make_split(5, seed=7)
    second = make_split(5, seed=7)
    assert [example.text for example in first] == [example.text for example in second]
    assert [example.tool for example in first] == [example.tool for example in second]


def test_family_distributions_are_normalized():
    for family in FAMILIES:
        assert abs(sum(family.tool) - 1) < 1e-9
        assert abs(sum(family.risk) - 1) < 1e-9
        assert 0 <= family.live <= 1
        assert 0 <= family.destructive <= 1


def test_hash_tokenizer_is_stable_and_bounded():
    ids = tokenize("Calculate 17 * 23", vocab=128, max_len=16)
    assert ids == tokenize("Calculate 17 * 23", vocab=128, max_len=16)
    assert all(2 <= token < 128 for token in ids)


def test_forward_shapes_and_loss_drop_on_a_repeated_batch():
    torch.manual_seed(0)
    example = make_split(1, seed=1)[0]
    batch = collate([example] * 16, vocab=128, max_len=64)
    model = DecisionModel(vocab=128, d_model=32, layers=1, heads=4, max_len=64, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def current() -> torch.Tensor:
        return decision_loss(model(batch["ids"], batch["mask"]), batch)

    before = float(current().detach())
    for _ in range(15):
        optimizer.zero_grad()
        loss = current()
        loss.backward()
        optimizer.step()
    after = float(current().detach())
    assert after < before


def test_positive_temperature_does_not_change_argmax():
    logits = torch.tensor([[0.2, 1.4, -0.3], [2.0, 2.0, 0.1]])
    labels = torch.tensor([1, 0])
    temperature = fit_temperature(logits, labels)
    assert temperature > 0
    assert torch.equal(logits.argmax(-1), (logits / temperature).argmax(-1))

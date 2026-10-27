"""Tests for section 8 of demo_network.py: python -m pytest test_demo_network.py -q"""
from pathlib import Path

import pytest
import torch

import demo_network as demo

RUNS = {"mlp": Path("submission/networkA_mlp"), "lstm": Path("submission/networkB_lstm")}


@pytest.fixture(scope="module", params=sorted(RUNS))
def loaded(request):
    if not all((RUNS[request.param] / f).exists() for f in demo.REQUIRED_FILES):
        pytest.skip(f"no saved run in {RUNS[request.param]}")
    _, cfg, wrappers, model, vn = demo.load_run(RUNS[request.param])
    return cfg, wrappers, model, vn


def test_every_trainable_tensor_gets_a_finite_nonzero_gradient(loaded):
    cfg, wrappers, model, vn = loaded
    batch = demo.rollout(model, cfg, wrappers, vn, n_steps=32)
    _, _, grads = demo.gradients(model.policy, batch, cfg)
    assert grads
    for name, _, g in grads:
        assert demo.flag(g) is None, f"{name}: {demo.flag(g)}"


def test_the_section_leaves_the_saved_weights_unchanged(loaded):
    cfg, wrappers, model, vn = loaded
    before = demo.checksum(model.policy)
    batch = demo.rollout(model, cfg, wrappers, vn, n_steps=32)
    clone = demo.copy.deepcopy(model.policy)
    demo.gradients(model.policy, batch, cfg)
    loss_before, loss_after = demo.adam_steps(clone, batch, cfg, n=3)
    assert demo.checksum(model.policy) == before
    assert all(p.grad is None for p in model.policy.parameters())
    assert loss_after != loss_before  # the copy did move
    assert not any(torch.equal(a, b) for a, b in zip(clone.parameters(), model.policy.parameters())
                   if a.requires_grad)

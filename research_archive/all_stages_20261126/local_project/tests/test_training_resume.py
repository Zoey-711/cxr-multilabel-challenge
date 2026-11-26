"""An epoch-boundary resume reproduces AMP, optimizer, scheduler and RNG state."""

import io
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cxr.engine.early_stopping import EarlyStopTracker
from cxr.utils.training_resume import capture_rng_state, restore_rng_state


class WorkerAugmentedDataset(TensorDataset):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x + 0.01 * torch.rand_like(x) + 0.001 * random.random() + 0.001 * np.random.random(), y


def setup(workers=0):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    x = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 32
    y = (x[:, :2].sum(dim=1) > 1).float().unsqueeze(1)
    train = DataLoader(WorkerAugmentedDataset(x, y) if workers else TensorDataset(x, y),
                       batch_size=4, shuffle=True,
                       generator=torch.Generator().manual_seed(42), num_workers=workers,
                       persistent_workers=False)
    val = DataLoader(TensorDataset(x[:4], y[:4]), batch_size=4, shuffle=False,
                     generator=torch.Generator().manual_seed(42), num_workers=0)
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.2), nn.Linear(8, 1)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    scaler = torch.amp.GradScaler("cuda")
    tracker = EarlyStopTracker(patience=8)
    return train, val, model, optimizer, scheduler, scaler, tracker


def epoch(state, number):
    train, val, model, optimizer, scheduler, scaler, tracker = state
    model.train()
    for x, y in train:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = nn.functional.binary_cross_entropy_with_logits(model(x.cuda()), y.cuda())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    list(val)
    tracker.observe(0.6 + number * 0.01, number)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("workers", [0, 2])
def test_full_epoch_resume_is_identical(workers):
    uninterrupted = setup(workers)
    epoch(uninterrupted, 1)
    epoch(uninterrupted, 2)
    expected_model = {key: value.detach().clone() for key, value in uninterrupted[2].state_dict().items()}
    expected_lr = uninterrupted[3].param_groups[0]["lr"]
    expected_scale = uninterrupted[5].get_scale()
    interrupted = setup(workers)
    epoch(interrupted, 1)
    train, val, model, optimizer, scheduler, scaler, tracker = interrupted
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
             "early_stop": tracker.state_dict(), **capture_rng_state(train, val)}
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    resumed = setup(workers)
    train, val, model, optimizer, scheduler, scaler, tracker = resumed
    model.load_state_dict(restored["model"])
    optimizer.load_state_dict(restored["optimizer"])
    scheduler.load_state_dict(restored["scheduler"])
    scaler.load_state_dict(restored["scaler"])
    for key in ("best_auc", "best_epoch", "patience_reference_auc", "stale", "last_epoch"):
        setattr(tracker, key, restored["early_stop"][key])
    restore_rng_state(restored, train, val)
    epoch(resumed, 2)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in expected_model.items())
    assert optimizer.param_groups[0]["lr"] == expected_lr
    assert scaler.get_scale() == expected_scale
    assert tracker.best_epoch == 2 and tracker.stale == 0

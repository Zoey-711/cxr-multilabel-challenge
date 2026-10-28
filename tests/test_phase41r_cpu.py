"""CPU-only regression for transient AMP overflow and frozen-result import."""
import json

import pytest
import torch
import yaml

from cxr.phase39_run import step_group
from cxr.phase41.recovery import import_completed_base
from cxr.phase41.protocol import ROOT, digest, load_plan, server_config


class OverflowScaler:
    def __init__(self):
        self.value = 1024.

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.fill_(float("inf"))

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.value

    def step(self, optimizer):
        pass

    def update(self):
        self.value /= 2


def test_fp16_overflow_backs_off_without_parameter_or_ema_update():
    model = torch.nn.Linear(2, 10)
    ema = torch.optim.swa_utils.AveragedModel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    before = [p.detach().clone() for p in model.parameters()]
    batch = dict(image=torch.ones(2, 2), target=torch.zeros(2, 10),
                 valid_mask=torch.ones(2, 10, dtype=torch.bool))
    diagnostic = {}
    loss, skipped = step_group({}, model, ema, [batch], optimizer, scheduler,
                               OverflowScaler(), torch.device("cpu"), diagnostic)
    assert loss > 0 and skipped and diagnostic["stage"] == "gradients"
    assert diagnostic["optimizer_skipped"] and diagnostic["scaler_after"] == 512
    assert all(torch.equal(old, new) and new.grad is None for old, new in zip(before, model.parameters()))
    assert int(ema.n_averaged) == 0 and scheduler.last_epoch == 0


def test_recovery_import_preserves_historical_receipt(tmp_path):
    source = tmp_path / "old" / "experiments" / "P41_BASE_F02"
    source.mkdir(parents=True)
    names = ("ema_best.pth", "oof_predictions.csv", "val_predictions.csv", "test_predictions.csv",
             "resolved_config.yaml", "run_manifest.json", "metrics.json")
    for name in names:
        (source / name).write_text("synthetic", encoding="utf-8")
    (source / "run_manifest.json").write_text(json.dumps(dict(task="P41_BASE_F02",
                              identity_sha256="historical")), encoding="utf-8")
    receipt = dict(state="DONE", identity_sha256="historical",
                   sha256={name: digest(source / name) for name in names})
    (source / "DONE.json").write_text(json.dumps(receipt), encoding="utf-8")
    target = tmp_path / "new"
    assert import_completed_base(tmp_path / "old", target, "repaired")["state"] == "REUSED"
    assert import_completed_base(tmp_path / "old", target, "repaired")["state"] == "REUSED"
    copied = json.loads((target / "experiments/P41_BASE_F02/DONE.json").read_text())
    assert copied["historical_identity_sha256"] == "historical" and copied["training_reused"]
    assert copied["sha256"] == receipt["sha256"]
    with pytest.raises(ValueError, match="changed"):
        import_completed_base(tmp_path / "old", target, "different")


def test_recovery_requires_separate_root_budget_and_result_branch(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/phase41r_server.example.yaml").read_text())
    for name in ("data_root", "pretrained_root", "protocol_root", "fold1_results_root",
                 "recovery_source_root", "output_root"):
        cfg[name] = str(tmp_path / name)
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="authorize"):
        server_config(path)
    cfg["gpu_hours"] = 400
    cfg["upload"]["enabled"] = True
    cfg["upload"]["checkout"] = str(tmp_path / "upload")
    path.write_text(yaml.safe_dump(cfg))
    assert server_config(path)["recovery_source_root"] == cfg["recovery_source_root"]
    cfg["upload"]["branch"] = "results/phase41-fivefold"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="dedicated private result branch"):
        server_config(path)


def test_repair_queue_runs_only_seven_missing_tasks(tmp_path, monkeypatch):
    from cxr.batch.plan import write_json
    from cxr.phase41 import queue

    source = tmp_path / "old" / "experiments/P41_BASE_F02"
    source.mkdir(parents=True)
    names = ("ema_best.pth", "oof_predictions.csv", "val_predictions.csv", "test_predictions.csv",
             "resolved_config.yaml", "run_manifest.json", "metrics.json")
    for name in names:
        (source / name).write_text("synthetic", encoding="utf-8")
    write_json(source / "run_manifest.json", dict(task="P41_BASE_F02", identity_sha256="old"))
    write_json(source / "DONE.json", dict(state="DONE", identity_sha256="old",
               sha256={name: digest(source / name) for name in names}))
    root = tmp_path / "new"
    import_completed_base(tmp_path / "old", root, "repair")
    calls = []

    def fake_worker(task, output, server_file, gpu, identity, attempt):
        calls.append(task["id"])
        folder = output / "experiments" / task["id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metrics.json").write_text("{}", encoding="utf-8")
        write_json(folder / "DONE.json", dict(state="DONE", identity_sha256=identity,
                   sha256={"metrics.json": digest(folder / "metrics.json")}))
        return dict(state="DONE", gpu_hours=.01)

    monkeypatch.setattr(queue, "worker", fake_worker)
    cfg = dict(output_root=str(root), gpu_ids=[0, 1], max_concurrent=2,
               gpu_hours=400., upload=dict(enabled=False))
    state = queue.run(load_plan(), cfg, tmp_path / "server.yaml", "repair", poll=.001)
    assert len(calls) == 7 and "P41_BASE_F02" not in calls
    assert sum(v["state"] == "DONE" for v in state["tasks"].values()) == 8

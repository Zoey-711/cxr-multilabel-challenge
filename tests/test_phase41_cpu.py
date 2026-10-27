"""CPU-only Phase41 protocol, recovery, analysis and queue checks."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from cxr.phase39_model import build_model
from cxr.phase39_run import optimizer_scheduler, step_group
from cxr.phase41 import queue
from cxr.phase41.analysis import auc_summary, bootstrap, matrix
from cxr.phase41.protocol import load_plan, resolved_config, server_config


def test_exact_eight_frozen_tasks_and_paired_training_settings():
    plan = load_plan()
    assert len(plan["tasks"]) == 8
    assert {t["fold"] for t in plan["tasks"]} == {2, 3, 4, 5}
    for fold in range(2, 6):
        a = resolved_config(plan, next(t for t in plan["tasks"] if t["variant"] == "BASE" and t["fold"] == fold))
        b = resolved_config(plan, next(t for t in plan["tasks"] if t["variant"] == "FUSION" and t["fold"] == fold))
        assert a["micro_batch"] == b["micro_batch"] == 4
        assert a["accumulation"] == b["accumulation"] == 4
        assert a["ema_decay"] == b["ema_decay"] == .999
        assert a["ema_use_buffers"] is b["ema_use_buffers"] is True
        assert a["fold_expected_counts"] == b["fold_expected_counts"]
        assert a["input_size"] == b["input_size"] == 384
    assert all(t["fold"] != 1 for t in plan["tasks"])


def test_no_budget_or_missing_gpu_permission_rejected(tmp_path):
    import yaml
    from cxr.phase41.protocol import ROOT
    template = yaml.safe_load((ROOT / "configs/phase41_server.example.yaml").read_text())
    for key in ("data_root", "pretrained_root", "protocol_root", "fold1_results_root", "output_root"):
        template[key] = str(tmp_path / key)
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(template))
    with pytest.raises(ValueError, match="authorize"):
        server_config(path)
    template["gpu_hours"] = 100
    template["gpu_ids"] = [0, 0]
    path.write_text(yaml.safe_dump(template))
    with pytest.raises(ValueError, match="distinct"):
        server_config(path)


def test_checkpoint_resume_and_ema_forward_cpu(tmp_path):
    torch.set_num_threads(2)
    cfg = resolved_config(load_plan(), load_plan()["tasks"][0])
    model = build_model("P39_BASE_F1", pretrained=False)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(.999), use_buffers=True)
    opt, sch = optimizer_scheduler(cfg, model, 1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = dict(image=torch.randn(2, 3, 64, 64), target=torch.rand(2, 10),
                 valid_mask=torch.ones(2, 10, dtype=torch.bool))
    loss, skipped = step_group(cfg, model, ema, [batch], opt, sch, scaler, torch.device("cpu"))
    assert np.isfinite(loss) and not skipped and int(ema.n_averaged) == 1
    state = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                 scheduler=sch.state_dict(), scaler=scaler.state_dict(), epoch=1,
                 tracker=dict(patience=6, min_delta=0., best_auc=.7, best_epoch=1,
                              patience_reference_auc=.7, stale=0, last_epoch=1))
    checkpoint = tmp_path / "resume.pt"
    torch.save(state, checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    restored = build_model("P39_BASE_F1", pretrained=False)
    restored.load_state_dict(saved["model"])
    with torch.no_grad():
        assert torch.equal(model.eval()(batch["image"]), restored.eval()(batch["image"]))
    assert saved["optimizer"]["state"] and saved["scheduler"]["last_epoch"] == sch.last_epoch
    assert saved["tracker"]["best_epoch"] == 1


def test_study_id_alignment_and_patient_bootstrap(tmp_path):
    keys = [(f"subject_{i:05d}", f"study_{i:05d}") for i in range(30)]
    truth = {k: tuple(float((i + j) % 2) for j in range(10)) for i, k in enumerate(keys)}
    base = {k: np.full(10, .2 if i % 2 else .8) for i, k in enumerate(keys)}
    fusion = {k: np.full(10, .8 if i % 2 else .2) for i, k in enumerate(keys)}
    assert 0 <= auc_summary(truth, base, keys)["macro_auc"] <= 1
    result = bootstrap(truth, base, fusion, keys, draws=50)
    assert result["valid"] + result["invalid"] == 50
    path = tmp_path / "oof.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("Subject_id,Study_id," + ",".join(f"Prob_{j}" for j in range(1, 11)) + "\n")
        for k in keys:
            f.write(k[0] + "," + k[1] + "," + ",".join("0.5" for _ in range(10)) + "\n")
    assert len(matrix(path, set(keys))) == 30
    with pytest.raises(ValueError, match="coverage"):
        matrix(path, set(keys[1:]))


def test_queue_partial_failure_and_done_reuse_without_gpu(tmp_path, monkeypatch):
    from cxr.batch.plan import write_json
    from cxr.phase41.protocol import digest
    plan = load_plan()
    calls = []

    def fake_worker(task, root, server_file, gpu, identity, attempt):
        calls.append((task["id"], gpu))
        folder = root / "experiments" / task["id"]
        folder.mkdir(parents=True, exist_ok=True)
        if task["id"] == "P41_BASE_F02":
            return dict(state="FAILED", reason="synthetic fatal data error", retryable=False, gpu_hours=.01)
        (folder / "metrics.json").write_text("{}")
        write_json(folder / "DONE.json", dict(identity_sha256=identity,
                   sha256={"metrics.json": digest(folder / "metrics.json")}))
        return dict(state="DONE", gpu_hours=.01)

    monkeypatch.setattr(queue, "worker", fake_worker)
    cfg = dict(output_root=str(tmp_path), gpu_ids=[0, 1], max_concurrent=2,
               gpu_hours=400., upload=dict(enabled=False))
    state = queue.run(plan, cfg, tmp_path / "config.yaml", "frozen", poll=.001)
    assert len(calls) == 8
    assert sum(r["state"] == "DONE" for r in state["tasks"].values()) == 7
    assert state["tasks"]["P41_BASE_F02"]["state"] == "FAILED"
    assert (tmp_path / "reports/experiment_status.csv").exists()
    state2 = queue.run(plan, cfg, tmp_path / "config.yaml", "frozen", poll=.001)
    assert len(calls) == 8 and state2["tasks"]["P41_FUSION_F05"]["state"] == "DONE"

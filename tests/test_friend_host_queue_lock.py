"""CPU-only cross-process check for the shared host GPU queue guard."""
import os
import subprocess
import sys
from pathlib import Path

import yaml

from cxr.batch.host_gpu_lock import host_gpu_lock


def test_phase32_and_e08_tune_cannot_own_host_together(tmp_path):
    path = tmp_path / "host_gpu.lock"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "CUDA_VISIBLE_DEVICES": "-1"}
    # A small script avoids relying on lock semantics within one process.
    code = "import sys\nfrom cxr.batch.host_gpu_lock import host_gpu_lock\nwith host_gpu_lock(sys.argv[1]):\n print('acquired')\n"
    with host_gpu_lock(path):
        blocked = subprocess.run([sys.executable, "-c", code, str(path)], env=env, capture_output=True, text=True)
        assert blocked.returncode != 0
        assert "Another Phase32 or E08-Tune queue" in blocked.stderr
    resumed = subprocess.run([sys.executable, "-c", code, str(path)], env=env, capture_output=True, text=True)
    assert resumed.returncode == 0 and resumed.stdout.strip() == "acquired"


def test_separate_output_roots_and_result_branches():
    from cxr.phase32.protocol import load_plan as phase_plan
    from cxr.e08_tune.protocol import load_plan as tune_plan
    from cxr.batch.plan import ROOT
    import yaml
    phase = yaml.safe_load((ROOT / "configs/phase32_server.example.yaml").read_text())
    tune = yaml.safe_load((ROOT / "configs/e08_tune_server.example.yaml").read_text())
    assert phase["output_root"] != tune["output_root"]
    assert phase["upload"]["branch"] != tune["upload"]["branch"]
    assert not ({t["id"] for t in phase_plan()["tasks"]} & {t["id"] for t in tune_plan()["tasks"]})
    assert len(phase_plan()["tasks"]) == 8 and len({t["id"] for t in tune_plan()["tasks"] if t["fold"] == 1}) == 5


def _local_config(tmp_path, example, output_name):
    from cxr.batch.plan import ROOT
    config = yaml.safe_load((ROOT / example).read_text(encoding="utf-8"))
    for key in ("data_root", "pretrained_root", "protocol_root", "baseline_results_root"):
        config[key] = str(tmp_path / key)
    config["output_root"] = str(tmp_path / output_name)
    config["gpu_hours"] = 100
    if "extension_gpu_hours" in config:
        config["extension_gpu_hours"] = 20
    path = tmp_path / (output_name + ".yaml")
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_phase32_preflight_action_never_starts_queue(tmp_path, monkeypatch, capsys):
    from scripts import phase32_run
    import cxr.phase32.preflight as probe
    import cxr.phase32.queue as queue

    path = _local_config(tmp_path, "configs/phase32_server.example.yaml", "phase32_batch")
    calls = []
    monkeypatch.setattr(probe, "prepare", lambda cfg, plan: calls.append("prepare") or {"source": "synthetic"})
    monkeypatch.setattr(probe, "preflight", lambda cfg, plan, identity: calls.append("preflight") or 0.1)
    monkeypatch.setattr(queue, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("formal queue started")))
    monkeypatch.setattr(sys, "argv", ["phase32_run.py", "--config", str(path), "--action", "preflight"])
    phase32_run.main()
    assert calls == ["prepare", "preflight"]
    assert "REMOTE_PREFLIGHT passed" in capsys.readouterr().out


def test_e08_preflight_action_never_starts_queue(tmp_path, monkeypatch, capsys):
    from scripts import e08_tune_run

    path = _local_config(tmp_path, "configs/e08_tune_server.example.yaml", "e08_tuning")
    calls = []
    monkeypatch.setattr(e08_tune_run, "prepare", lambda cfg, plan: calls.append(("prepare", len(plan["tasks"]))) or {"source": "synthetic"})
    monkeypatch.setattr(e08_tune_run, "preflight", lambda cfg, plan, identity: calls.append(("preflight", len(plan["tasks"]))) or 0.1)
    monkeypatch.setattr(e08_tune_run, "stage", lambda *a, **k: (_ for _ in ()).throw(AssertionError("formal queue started")))
    monkeypatch.setattr(sys, "argv", ["e08_tune_run.py", "--config", str(path), "--action", "preflight"])
    e08_tune_run.main()
    assert calls == [("prepare", 5), ("preflight", 5)]
    assert "Pilot REMOTE_PREFLIGHT passed" in capsys.readouterr().out

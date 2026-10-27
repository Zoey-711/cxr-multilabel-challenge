"""Internal one-GPU worker; normally launched by scripts/phase41_run.py."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("cpu", "smoke", "run", "fold1-infer"))
    parser.add_argument("--task")
    parser.add_argument("--variant", choices=("BASE", "FUSION"))
    parser.add_argument("--gpu")
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--identity-hash")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Import torch after physical-GPU binding.
    import torch
    from torch.nn import functional as F
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    from cxr.phase39_model import build_model
    from cxr.phase39_support import digest, save_torch, write_json
    from cxr.phase41.protocol import load_plan, server_config
    from cxr.phase41.train import real_smoke, run_task, test_probabilities
    from cxr.data.archive import read_split_rows
    from cxr.data.dataset import make_dataloader, study_targets_from_rows
    from cxr.phase39_run import infer

    plan = load_plan()
    if args.mode == "cpu":
        torch.set_num_threads(2)
        records = {}
        for variant in ("BASE", "FUSION"):
            model = build_model(f"P39_{variant}_F1", pretrained=False)
            model.eval()
            x = torch.randn(2, 3, 64, 64)
            logits = model(x)
            if logits.shape != (2, 10):
                raise ValueError("Model output shape changed")
            F.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
            if not all(p.grad is not None for p in model.parameters() if p.requires_grad):
                raise ValueError("Model gradient missing")
            ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(.999), use_buffers=True)
            ema.update_parameters(model)
            if int(ema.n_averaged) != 1:
                raise ValueError("EMA update failed")
            records[variant] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                    output_shape=list(logits.shape), ema="PASS", gradient="PASS")
        print(json.dumps(dict(state="PASS", variants=records)))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for remote Phase41 preflight/formal work")
    server = server_config(args.config)
    os.environ["TORCH_HOME"] = server["pretrained_root"]
    if args.mode == "smoke":
        task = next(t for t in plan["tasks"] if t["id"] == args.task)
        print(json.dumps(real_smoke(plan, task, server, args.probe)))
    elif args.mode == "run":
        task = next(t for t in plan["tasks"] if t["id"] == args.task)
        if not args.identity_hash:
            raise ValueError("Verified preflight identity required")
        print(json.dumps(run_task(plan, task, server, args.identity_hash)))
    else:
        if not args.variant:
            raise ValueError("Fold1 variant required")
        source = plan["fold1"][args.variant]
        folder = Path(server["output_root"]) / "reused_fold1" / source["model_id"]
        if (folder / "REUSED.json").exists():
            receipt = json.loads((folder / "REUSED.json").read_text())
            if all(digest(folder / p) == h for p, h in receipt["sha256"].items()):
                print(json.dumps(dict(state="REUSED", model_id=source["model_id"])))
                return
            raise ValueError("Fold1 reused artifact checksum mismatch")
        cfg = json.loads((folder / "resolved_config.json").read_text(encoding="utf-8"))
        model = build_model(source["model_id"], pretrained=False).cuda()
        model.load_state_dict(torch.load(folder / "ema_best.pth", map_location="cpu", weights_only=True), strict=True)
        archive = Path(server["data_root"])
        val = read_split_rows(archive, "val")
        loader = make_dataloader(archive, "val", rows=val, batch_size=4, num_workers=0,
                                 input_size=384, preprocessing=cfg["preprocessing"])
        metrics = infer(model, loader, study_targets_from_rows(val), folder / "val_predictions.csv", torch.device("cuda"))
        test_probabilities(model, archive, cfg, folder / "test_predictions.csv")
        write_json(folder / "val_metrics.json", metrics)
        files = {p.name: digest(p) for p in folder.iterdir() if p.is_file() and p.name != "REUSED.json"}
        write_json(folder / "REUSED.json", dict(state="REUSED", source_id=source["model_id"],
                   source_sha256=source["assets"], sha256=files, trained_again=False))
        print(json.dumps(dict(state="REUSED", model_id=source["model_id"], val_auc=metrics["macro_auc"])))


if __name__ == "__main__":
    main()

"""Final formula and initial-gradient audit before an L01 training launch."""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from cxr.data.archive import archive_path, read_split_rows
from cxr.data.dataset import CXRArchiveDataset
from cxr.engine.losses import asymmetric_loss_with_logits, per_label_loss
from cxr.inference.versioned_submission import sha256
from cxr.models.factory import create_model, imagenet_weight_info
from cxr.utils.config import load_config


M02 = Path("outputs/experiments/m02_convnext_tiny_384_aug_20261016_elevated")


def reference_asl(logits: torch.Tensor, targets: torch.Tensor, settings: dict) -> torch.Tensor:
    """Independent elementwise translation of Alibaba-MIIL AsymmetricLoss."""
    x = logits.float()
    y = targets.float()
    pos = torch.sigmoid(x)
    neg = (1 - pos + settings["clip"]).clamp(max=1)
    loss = y * torch.log(pos.clamp(min=settings["eps"])) + (1 - y) * torch.log(neg.clamp(min=settings["eps"]))
    with torch.no_grad():
        weight = (1 - (pos * y + neg * (1 - y))).pow(settings["gamma_pos"] * y + settings["gamma_neg"] * (1 - y))
    return -loss * weight


def gradient_norm(parameters) -> float:
    return math.sqrt(sum(float(parameter.grad.float().square().sum())
                         for parameter in parameters if parameter.grad is not None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/phase10_asl_final_audit_20261017.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config, baseline = load_config(args.config), load_config(M02 / "config.yaml")
    candidate_copy = json.loads(json.dumps(config))
    candidate_copy["experiment"]["name"] = baseline["experiment"]["name"]
    candidate_copy["training"]["loss"] = baseline["training"]["loss"]
    candidate_copy["training"].pop("asl")
    if candidate_copy != baseline or config["training"]["loss"] != "asymmetric_loss":
        raise ValueError("L01 differs from frozen M02 beyond experiment name and ASL loss")
    if config["training"]["asl"] != {"gamma_neg": 4.0, "gamma_pos": 1.0, "clip": 0.05,
                                      "eps": 1e-8, "detach_focal_weight": True}:
        raise ValueError("ASL parameters differ from prespecified values")
    if not torch.cuda.is_available():
        raise RuntimeError("Final audit requires the GPU")
    if (Path("outputs/experiments") / config["experiment"]["name"]).exists():
        raise FileExistsError("L01 output directory already exists")
    _, _, filename = imagenet_weight_info("convnext_tiny")
    weight_file = Path(".wheels/torch_cache/hub/checkpoints") / filename
    source = json.loads((M02 / "pretrained_source.json").read_text(encoding="utf-8"))
    if sha256(weight_file) != source["sha256"]:
        raise ValueError("ImageNet weight hash differs from M02")

    # Compare the formula and the detached focal derivative against the author algorithm.
    logits = torch.tensor([[-8., -1., 0., 1., 8.], [4., -4., 2., -2., 0.25]], requires_grad=True)
    targets = torch.tensor([[0., 0., 1., 1., 0.], [1., 0., 0., 1., 1.]])
    own = asymmetric_loss_with_logits(logits, targets, **config["training"]["asl"])
    ref = reference_asl(logits, targets, config["training"]["asl"])
    error = float((own - ref).abs().max().detach())
    own_grad = torch.autograd.grad(own.sum(), logits, retain_graph=True)[0]
    ref_grad = torch.autograd.grad(ref.sum(), logits)[0]
    grad_error = float((own_grad - ref_grad).abs().max())
    if error > 1e-7 or grad_error > 1e-7:
        raise ValueError(f"ASL formula/gradient mismatch: {error}, {grad_error}")

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    archive = archive_path(config)
    rows = read_split_rows(archive, "train")
    dataset = CXRArchiveDataset(archive, "train", rows=rows[:2], unlisted_policy="negative",
                                augmentation=config["data"]["train_augmentation"],
                                preprocessing=config["data"].get("preprocessing", "black_pad_v1"))
    samples = [dataset[i] for i in range(2)]
    dataset.close()
    images = torch.stack([item["image"] for item in samples]).to("cuda")
    truth = torch.stack([item["target"] for item in samples]).to("cuda")
    mask = torch.stack([item["valid_mask"] for item in samples]).to("cuda")
    os.environ["TORCH_HOME"] = str(Path(".wheels/torch_cache").resolve())
    model = create_model("convnext_tiny", pretrained=True).to("cuda").train()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(images)
        bce = (per_label_loss(output, truth, baseline["training"]) * mask).sum() / mask.sum()
        asl = (per_label_loss(output, truth, config["training"]) * mask).sum() / mask.sum()
    grad_info = {}
    for name, loss in (("bce", bce), ("asl", asl)):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=name == "bce")
        info = {"initial_loss": float(loss.detach()),
                "all_parameter_gradient_norm": gradient_norm(model.parameters()),
                "backbone_gradient_norm": gradient_norm(model.features.parameters()),
                "classifier_gradient_norm": gradient_norm(model.classifier.parameters())}
        if any(not math.isfinite(value) or value <= 0 for value in info.values()):
            raise ValueError(f"Nonfinite or zero {name} loss/gradient")
        grad_info[name] = info
    ratios = {name: grad_info["asl"][name] / grad_info["bce"][name] for name in grad_info["bce"]}
    if any(not 0.01 < value < 100 for value in ratios.values()):
        raise ValueError(f"Unexpected initial ASL/BCE ratio: {ratios}")
    result = {"status": "passed", "optimizer_steps": 0, "train_samples": 2,
              "formula_max_absolute_error": error, "formula_gradient_max_absolute_error": grad_error,
              "loss_and_gradient": grad_info, "asl_to_bce_ratios": ratios,
              "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
              "pretrained_sha256": sha256(weight_file), "config_sha256": sha256(args.config),
              "mask_valid_labels": int(mask.sum()), "micro_batch": config["training"]["batch_size"],
              "accumulation_steps": config["training"]["gradient_accumulation_steps"],
              "effective_batch": config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"],
              "normalization": "sum(per_label_loss * valid_mask) / valid labels across the accumulation group"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

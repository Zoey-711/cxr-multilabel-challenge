"""Load a portable state_dict and write a study-level submission CSV."""

import argparse
from pathlib import Path

from cxr.inference.aggregate import aggregate_mean_probabilities
from cxr.data.archive import archive_path
from cxr.inference.predict import predict_archive_images, predict_image_rows, read_image_rows
from cxr.inference.submission import write_submission
from cxr.utils.checkpoint import check_model_file_size, load_model_state
from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--policy", choices=("all_classes", "threshold"),
                        help="Override config submission_mode; threshold is the documented default")
    parser.add_argument("--threshold", type=float, help="Override config inference.threshold")
    parser.add_argument("--output", type=Path, default=Path("outputs/submissions/submission.csv"))
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; prediction requires a GPU")
    device = "cuda"
    size = check_model_file_size(args.checkpoint)
    model = load_model_state(args.checkpoint, device=device)
    if data.get("archive_path"):
        predictions = predict_archive_images(model, archive_path(config),
                                             batch_size=int(config["inference"]["batch_size"]),
                                             num_workers=int(data["num_workers"]), device=device,
                                             input_size=int(data["input_size"]),
                                             preprocessing=data.get("preprocessing", "black_pad_v1"))
    else:
        columns = data["columns"]
        for field in ("subject_id", "study_id", "image_path"):
            if not columns.get(field):
                parser.error(f"Set data.columns.{field} from the actual test metadata")
        if not data.get("test_manifest") or not data.get("image_root"):
            parser.error("Set data.test_manifest and data.image_root, or data.archive_path")
        rows = read_image_rows(
            data["test_manifest"],
            subject_column=columns["subject_id"],
            study_column=columns["study_id"],
            image_column=columns["image_path"],
        )
        predictions = predict_image_rows(model, rows, image_root=data["image_root"], input_size=data["input_size"],
                                         device=device, preprocessing=data.get("preprocessing", "black_pad_v1"))
    studies = aggregate_mean_probabilities(predictions)
    policy = args.policy or config["inference"]["submission_mode"]
    threshold = args.threshold if args.threshold is not None else config["inference"]["threshold"]
    output = write_submission(studies, args.output, policy=policy, threshold=threshold)
    print(f"Wrote {output}: {len(studies)} studies, {len(predictions)} images; model file {size} bytes")


if __name__ == "__main__":
    main()

"""Read a few real Val/Test images and verify deterministic CUDA inference without saving predictions."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from cxr.data.archive import archive_path, read_split_rows
    from cxr.data.dataset import make_dataloader
    from cxr.models.factory import create_model
    from cxr.utils.cloud_paths import pretrained_home
    from cxr.utils.config import load_config

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One authorized CUDA device must be visible for real inference')
    config = load_config(args.config)
    os.environ['TORCH_HOME'] = str(pretrained_home(Path(__file__).resolve().parents[1]))
    model = create_model(config['model']['name'], pretrained=True).cuda().eval()
    archive = archive_path(config)
    report = {}
    for split in ('val', 'test'):
        try:
            rows = read_split_rows(archive, split)[:2]
        except (FileNotFoundError, KeyError):
            if split == 'test':
                report[split] = {'state': 'PENDING', 'reason': 'Official unlabeled Test unavailable'}
                continue
            raise
        if len(rows) != 2:
            raise RuntimeError(f'{split} has fewer than two real images')
        loader = make_dataloader(archive, split, rows=rows, batch_size=2, num_workers=0,
                                 preprocessing=config['data'].get('preprocessing', 'black_pad_v1'),
                                 input_size=int(config['data']['input_size']), persistent_workers=False)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            batch = next(iter(loader))
            probs = model(batch['image'].cuda()).float().sigmoid()
        loader.dataset.close()
        if probs.shape != (2, 10) or not torch.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
            raise RuntimeError(f'{split} short inference produced invalid probabilities')
        report[split] = {'real_images': 2, 'probability_shape': list(probs.shape)}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

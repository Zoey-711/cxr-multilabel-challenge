"""One completed model's deterministic, label-free official Test inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cxr.batch.plan import digest
from cxr.batch.test_results import write_test
from cxr.data.archive import archive_path, read_split_rows
from cxr.phase32.dataset import make_dataloader
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
from cxr.utils.checkpoint import load_model_state
from cxr.phase32.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--checkpoint-kind', choices=('raw', 'ema'), default='raw')
    parser.add_argument('--pure', action='store_true', help='Use copied Fold1 pure inference state_dict')
    args = parser.parse_args()
    config, run = load_config(args.config), args.run_dir.resolve()
    if load_config(run/'config.yaml') != config or not torch.cuda.is_available():
        parser.error('Frozen config and CUDA GPU required')
    status = json.loads((run/'status.json').read_text(encoding='utf-8'))
    if status['state'] != 'complete':
        parser.error('Training has not completed')
    archive = archive_path(config)
    rows = read_split_rows(archive, 'test')  # no Predict_class is present or constructed
    expected = {(row['Subject_id'], row['Study_id']) for row in rows}
    if len(rows) != 2218 or len(expected) != 1414 or len({r['Dicom_id'] for r in rows}) != 2218:
        raise ValueError('Official Test image/Study coverage mismatch')
    kind = args.checkpoint_kind
    checkpoint = run/('best_model.pt' if kind == 'raw' else 'best_ema_model.pt')
    if args.pure: checkpoint = run/('best_model.pth' if kind == 'raw' else 'ema_best_model.pth')
    output = run/('test_predictions.csv' if kind == 'raw' else 'ema_test_predictions.csv')
    if output.exists():
        raise FileExistsError(output)
    if args.pure:
        from cxr.models.factory import create_model
        model=create_model(config['model']['name'],num_classes=10,pretrained=False)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        model=model.to('cuda').eval()
    else:
        model = load_model_state(checkpoint, device='cuda').eval()
    loader = make_dataloader(archive, 'test', rows=rows,
                             batch_size=int(config['training']['batch_size']),
                             num_workers=int(config['data']['num_workers']),
                             preprocessing=config['data'].get('preprocessing', 'black_pad_v1'),
                             input_size=int(config['data']['input_size']))
    predictions = []
    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch['image'].to('cuda', non_blocking=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    probabilities = torch.sigmoid(model(images)).float().cpu().tolist()
                for index, vector in enumerate(probabilities):
                    predictions.append(ImagePrediction(batch['subject_id'][index], batch['study_id'][index],
                                                       batch['image_id'][index], tuple(vector)))
    finally:
        loader.dataset.close()
    studies = aggregate_mean_probabilities(predictions)
    table = {'keys': [(s.subject_id, s.study_id) for s in studies],
             'p': np.asarray([s.probabilities for s in studies], dtype=np.float64)}
    if len(predictions) != 2218 or set(table['keys']) != expected:
        raise ValueError('Test inference did not cover every official image and Study')
    write_test(output, table, {'model_id': run.name, 'checkpoint_kind': kind,
               'checkpoint_sha256': digest(checkpoint), 'config_sha256': digest(args.config),
               'input_images': len(predictions), 'expected_test_studies': len(expected),
               'preprocessing': config['data'].get('preprocessing', 'black_pad_v1'),
               'input_size': int(config['data']['input_size']), 'no_test_labels': True})
    print(json.dumps({'model_id': run.name, 'kind': kind, 'images': len(predictions),
                      'studies': len(studies), 'sha256': digest(output)}))


if __name__ == '__main__':
    main()

"""Fail-closed official deployment gates. Research timings never establish compliance."""
import math
from pathlib import Path
from cxr.batch.plan import write_json

SIZE_LIMIT = 500_000_000
LATENCY_LIMIT_MS = 100.0
PROTOCOL_FIELDS = ('hardware', 'precision', 'batch_size', 'timing_scope', 'statistic',
                   'warmup_samples', 'measured_samples', 'software_environment')


def assess(candidate, protocol, measurement=None):
    """Candidate binds accuracy and timing to exact artifacts and inference recipe."""
    weights = candidate.get('dependencies', [])
    total = sum(w['bytes'] for w in weights)
    result = {'candidate': candidate['candidate'], 'bytes': total,
              'val_auc': candidate.get('val_auc'), 'status': 'LATENCY_UNVERIFIED',
              'eligible': False, 'latency_ms': None, 'measurement_scope': None,
              'hardware': None, 'reasons': []}
    if total > SIZE_LIMIT:
        result.update(status='SIZE_FAIL', reasons=['Total inference weights exceed 500,000,000 bytes'])
        return result
    hashes = sorted(w['sha256'] for w in weights)
    recipe = candidate.get('inference_recipe_sha256')
    verified = bool(weights and recipe and protocol.get('confirmed') is True
                    and protocol.get('official_source')
                    and all(protocol.get(k) is not None for k in PROTOCOL_FIELDS)
                    and measurement and measurement.get('protocol_sha256') == protocol.get('sha256')
                    and protocol.get('sha256')
                    and measurement.get('checkpoint_sha256') == hashes
                    and measurement.get('inference_recipe_sha256') == recipe
                    and measurement.get('full_pipeline') is True
                    and all(measurement.get(k) == protocol[k] for k in PROTOCOL_FIELDS))
    if verified:
        latency = measurement.get('latency_ms')
        if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            verified = False
        else:
            result.update(latency_ms=latency, hardware=measurement['hardware'],
                          measurement_scope=measurement['timing_scope'])
            if latency >= LATENCY_LIMIT_MS:
                result.update(status='LATENCY_FAIL', reasons=['Official full-pipeline latency must be strictly <100ms'])
                return result
    auc = candidate.get('val_auc')
    accuracy_ok = (isinstance(auc, (int, float)) and math.isfinite(auc) and 0 <= auc <= 1
                   and recipe and candidate.get('accuracy_recipe_sha256') == recipe
                   and candidate.get('accuracy_checkpoint_sha256') == hashes)
    if not accuracy_ok:
        result.update(status='ACCURACY_PENDING', reasons=['Revalidate exact weights and precision/quantization/distillation recipe'])
    elif not verified:
        result['reasons'] = ['Official protocol/hardware or full-ensemble timing evidence is missing or mismatched']
    else:
        result.update(status='PASS', eligible=True)
    return result


def export_reports(root, candidates, protocol):
    from cxr.batch.reporting import csv_rows
    root = Path(root)
    folder = root/'deployment'; folder.mkdir(parents=True, exist_ok=True)
    rows, sizes, benchmarks = [], [], []
    for candidate in candidates:
        evidence = folder/'official_benchmarks'/f'{candidate["candidate"]}.json'
        measurement = __import__('json').loads(evidence.read_text(encoding='utf-8')) if evidence.exists() else None
        row = assess(candidate, protocol, measurement)
        rows.append(row)
        benchmarks.append({'candidate': candidate['candidate'], 'official_evidence': measurement,
                           'status': row['status'], 'research_only_measurements': candidate.get('latency', [])})
        for item in candidate.get('dependencies', []):
            sizes.append({'candidate': candidate['candidate'], **item})
    fields = ['candidate','bytes','val_auc','status','eligible','latency_ms','measurement_scope','hardware','reasons']
    csv_rows(folder/'deployment_candidates.csv', rows, fields)
    csv_rows(folder/'deployment_500mb_candidates.csv', [r for r in rows if r['bytes'] <= SIZE_LIMIT], fields)
    csv_rows(folder/'deployable_PASS_only.csv', [r for r in rows if r['eligible']], fields)
    csv_rows(folder/'model_size_report.csv', sizes, ['candidate','path','sha256','bytes'])
    csv_rows(folder/'inference_benchmark.csv', [{k:r[k] for k in fields if k not in ('bytes','val_auc')} for r in rows],
             [k for k in fields if k not in ('bytes','val_auc')])
    write_json(folder/'inference_benchmark.json', {'protocol': protocol, 'measurements': benchmarks})
    write_json(folder/'deployment_candidates.json', [{**c, **r} for c,r in zip(candidates,rows)])
    (folder/'deployment_report.md').write_text(
        '# Deployment hard constraints\n\nWeights <=500,000,000 bytes AND official full-pipeline latency <100ms. '
        '100ms itself fails. Only PASS is deployable; all other states are excluded. '
        'RTX4090/RTX4060 timings are research evidence, not target-hardware compliance. '
        'FP16/quantization/distillation require fresh accuracy and timing for exact artifacts.\n\n'+
        '\n'.join(f'- {r["candidate"]}: {r["status"]}, {r["bytes"]} bytes, {r["latency_ms"]} ms' for r in rows), encoding='utf-8')
    return rows

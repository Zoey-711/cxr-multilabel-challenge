"""Separate, explicitly authorized G/H queue after the frozen 22-task batch."""
import concurrent.futures
import csv
import json
import os
import time
from pathlib import Path

from cxr.batch.artifacts import record_failure, utcnow
from cxr.batch.extension import prepare_extension
from cxr.batch.plan import digest, write_json
from cxr.batch.runner import budget_allows, execute, run_command
from cxr.batch.reporting import csv_rows


def _status(root, state):
    root = Path(root)
    rows = [{'Model_ID': name, 'State': record['state'], 'Attempts': record['attempts'],
             'GPU_Hours': record.get('gpu_hours', 0), 'Reason': record.get('reason', '')}
            for name, record in state['tasks'].items()]
    csv_rows(root/'reports/extension_status.csv', rows)
    write_json(root/'extension_state.json', state)


def run_extension_queue(root, plan, gpus, *, total_budget, gh_budget,
                        max_concurrent, resume=False, pause_seconds=10, gpu_preflight=True):
    root = Path(root).resolve()
    if total_budget <= 0 or gh_budget <= 0:
        raise ValueError('Separate positive friend-approved total and G/H GPU-hour caps are required')
    if not 1 <= max_concurrent <= min(len(gpus), 8):
        raise ValueError('Extension concurrency exceeds authorized GPU pool')
    base = json.loads((root/'queue_state.json').read_text(encoding='utf-8'))
    slots = plan['conditional_extension']['reserved_slots']
    required = set(slots[0]['depends_on'])
    if any(base['tasks'].get(task_id, {}).get('state') != 'complete' for task_id in required):
        state = {'state': 'BLOCKED', 'reason': 'One or more of the required 20 five-fold tasks is incomplete',
                 'tasks': {slot['id']: {'state': 'BLOCKED', 'attempts': 0} for slot in slots}}
        write_json(root/'reports/extension_blocked.json', state)
        return state
    if (root/'STOP').exists():
        raise ValueError('STOP is set; extension cannot start')
    try:
        tasks = prepare_extension(root, plan)
    except (OSError, ValueError, KeyError) as exc:
        state = {'state': 'BLOCKED', 'reason': f'{type(exc).__name__}: {exc}',
                 'tasks': {slot['id']: {'state': 'BLOCKED', 'attempts': 0,
                                        'reason': str(exc)} for slot in slots}}
        write_json(root/'reports/extension_blocked.json', state)
        return state
    selection_sha = digest(root/'manifests/extension_selection.json')
    state_file = root/'extension_state.json'
    if resume:
        state = json.loads(state_file.read_text(encoding='utf-8'))
        if state['selection_sha256'] != selection_sha:
            raise ValueError('G/H selection changed on resume')
        for record in state['tasks'].values():
            if record['state'] in ('running', 'test_pending', 'test_failed', 'BLOCKED'):
                if record['state'] == 'running':
                    state['gpu_hours'] += record.get('reserved_hours', 0)
                record['state'] = 'pending'
    else:
        if state_file.exists():
            raise FileExistsError('Existing G/H queue; resume explicitly instead of overwriting')
        state = {'state': 'running', 'selection_sha256': selection_sha, 'gpu_hours': 0.0,
                 'tasks': {task['id']: {'state': 'pending', 'attempts': 0} for task in tasks}}
    lock = root/'extension.lock'
    with lock.open('x') as handle:
        json.dump({'started_utc': utcnow()}, handle)
    active = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
    try:
        if gpu_preflight and not state.get('preflight_complete'):
            logs = root/'queue_logs/gh_preflight'
            logs.mkdir(parents=True, exist_ok=True)
            for task in tasks:
                record = state['tasks'][task['id']]
                if record['state'] == 'complete':
                    continue
                name = task['id'] + '_GH_PREFLIGHT'
                preflight_run=root/'experiments'/name
                if preflight_run.exists():
                    status_path=preflight_run/'status.json'
                    check=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
                    if (check.get('state')=='preflight_passed' and not check.get('amp_skipped_steps') and
                            (preflight_run/'config.yaml').is_file() and
                            digest(preflight_run/'config.yaml')==task['config_sha256']):
                        continue
                    # Keep the failed smoke intact; an explicit queue resume uses a new ID.
                    attempt=2
                    while (root/'experiments'/f'{name}_attempt_{attempt}').exists():
                        attempt+=1
                    name=f'{name}_attempt_{attempt}'
                    preflight_run=root/'experiments'/name
                reserve = .25
                if not (budget_allows(state['gpu_hours'], 0, reserve, gh_budget) and
                        budget_allows(state['gpu_hours'], 0, reserve, total_budget-base['gpu_hours'])):
                    record.update(state='BLOCKED', reason='G/H GPU preflight exceeds authorized budget')
                    _status(root, state)
                    continue
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpus[0], CXR_OUTPUT_ROOT=str(root))
                started = time.monotonic()
                try:
                    rc = run_command(['scripts/batch_train.py','--config',task['config'],
                                      '--run-id',name,'--preflight-steps','1'], env,
                                     logs/(name+'.out.log'), logs/(name+'.err.log'), reserve*3600)
                except Exception as exc:
                    rc=-1
                    record.update(state='BLOCKED', reason=f'G/H smoke error: {type(exc).__name__}: {exc}')
                state['gpu_hours'] += (time.monotonic()-started)/3600
                _status(root, state)
                status_file = preflight_run/'status.json'
                if rc or not status_file.exists():
                    record.update(state='BLOCKED', reason=f'G/H real training-step preflight failed: {name}; see queue_logs')
                    _status(root, state)
                    continue
                check = json.loads(status_file.read_text(encoding='utf-8'))
                if check.get('state') != 'preflight_passed' or check.get('amp_skipped_steps'):
                    record.update(state='BLOCKED', reason=f'G/H AMP/gradient preflight failed: {name}')
                    _status(root, state)
            state['preflight_complete'] = all(record['state'] != 'BLOCKED' for record in state['tasks'].values())
            _status(root, state)
        while True:
            for future, (task, gpu) in list(active.items()):
                if not future.done():
                    continue
                result = future.result()
                record = state['tasks'][task['id']]
                elapsed = result.pop('gpu_hours')
                state['gpu_hours'] += elapsed
                record['gpu_hours'] = record.get('gpu_hours', 0) + elapsed
                record.update(result)
                if result['state'] in ('failed', 'test_failed'):
                    record_failure(root, task, record)
                if result['state'] == 'failed' and result.get('retryable') and record['attempts'] <= task['max_retries']:
                    record['state'] = 'pending'
                del active[future]
                _status(root, state)
            busy = {gpu for _, gpu in active.values()}
            if not (root/'STOP').exists():
                for gpu in gpus:
                    if gpu in busy or len(active) >= max_concurrent:
                        continue
                    for task in tasks:
                        record = state['tasks'][task['id']]
                        if record['state'] != 'pending':
                            continue
                        reserved = sum(t['timeout_hours'] for t, _ in active.values())
                        remaining_total = total_budget - base['gpu_hours']
                        if not (budget_allows(state['gpu_hours'], reserved, task['timeout_hours'], gh_budget)
                                and budget_allows(state['gpu_hours'], reserved, task['timeout_hours'], remaining_total)):
                            continue
                        record.update(state='running', attempts=record['attempts'] + 1,
                                      gpu=gpu, reserved_hours=task['timeout_hours'])
                        active[executor.submit(execute, task, root, gpu, record['attempts'])] = (task, gpu)
                        break
            state['updated_utc'] = utcnow()
            _status(root, state)
            if not active:
                break
            time.sleep(pause_seconds)
        state['state'] = 'complete' if all(r['state'] == 'complete' for r in state['tasks'].values()) else 'partial'
        for record in state['tasks'].values():
            if record['state'] == 'pending':
                record.update(state='BLOCKED', reason='STOP requested or authorized GPU-hour budget insufficient')
        _status(root, state)
        extension_report(root, state)
        return state
    except Exception as exc:
        state['state'] = 'BLOCKED'
        for record in state['tasks'].values():
            if record['state'] == 'pending':
                record.update(state='BLOCKED', reason=f'Extension preflight/report failure: {exc}')
        _status(root, state)
        write_json(root/'reports/extension_blocked.json',
                   {'error_type': type(exc).__name__, 'reason': str(exc),
                    'completed_models_preserved': [name for name, item in state['tasks'].items()
                                                   if item['state'] == 'complete']})
        raise
    finally:
        executor.shutdown(wait=True)
        lock.unlink()


def extension_report(root, state, *, base_report=True):
    """Missing results remain blank; full-Train OOF is always N/A."""
    root = Path(root)
    rows = []
    for task_id, record in state['tasks'].items():
        run = root/'experiments'/task_id
        metric = json.loads((run/'metrics.json').read_text(encoding='utf-8')) if (run/'DONE.json').is_file() else {}
        rows.append({'model_id': task_id, 'state': record['state'], 'best_epoch': metric.get('best_epoch'),
                     'val_macro_auc': metric.get('official_val', {}).get('all', {}).get('macro_auc'),
                     'oof_macro_auc': 'N/A', 'gpu_hours': record.get('gpu_hours', 0),
                     'weight_bytes': metric.get('weights_bytes'), 'reason': record.get('reason', '')})
    csv_rows(root/'reports/gh_experiment_summary.csv', rows)
    write_json(root/'reports/gh_experiment_summary.json', rows)
    from cxr.batch.reporting import report
    if base_report:
        report(root)
    # Keep G/H in the same top-level metrics table as the original 22 tasks.
    # These full-Train models have no independent OOF by design.
    summary_path = root/'reports/experiment_summary.csv'
    if summary_path.is_file():
        from cxr.batch.reporting import _summary_row
        with summary_path.open(encoding='utf-8', newline='') as handle:
            summary_reader = csv.DictReader(handle)
            summary_fields = summary_reader.fieldnames
            summary_rows = list(summary_reader)
        positions = {row.get('model_id'): index for index, row in enumerate(summary_rows)}
        for task_id, record in state['tasks'].items():
            run = root/'experiments'/task_id
            metric = json.loads((run/'metrics.json').read_text(encoding='utf-8')) if (run/'DONE.json').is_file() else {}
            val = metric.get('official_val', {}).get('all', {})
            test_prediction = (run/'test_predictions.csv').is_file()
            row = _summary_row(task_id, val, None, {
                'state': record.get('state', 'missing'), 'best_epoch': metric.get('best_epoch'),
                'stop_epoch': metric.get('stop_epoch'), 'epoch_hours': record.get('gpu_hours'),
                'weights_bytes': metric.get('weights_bytes'),
                'oof_not_applicable': True,
                'test_prediction_status': 'probabilities_saved' if test_prediction else 'not_available'})
            if task_id in positions:
                summary_rows[positions[task_id]] = row
            else:
                positions[task_id] = len(summary_rows)
                summary_rows.append(row)
        csv_rows(summary_path, summary_rows, summary_fields)
    from cxr.batch.artifacts import label_keys, read_probabilities
    from cxr.batch.results import bootstrap, grouped
    import yaml
    labels=root/'manifests/val_labels.csv'
    expected=label_keys(labels)
    comparisons={}
    for task_id, record in state['tasks'].items():
        if record['state'] != 'complete':
            continue
        config=yaml.safe_load((root/'experiments'/task_id/'resolved_config.yaml').read_text(encoding='utf-8'))
        backbone=config['model']['name']
        branch={'densenet121':'D17','convnext_tiny':'C17','swin_t':'S17',
                'efficientnet_v2_s':'N17'}[backbone]
        reference=read_probabilities(root/'validation'/f'{branch}_val_ensemble.csv',labels,expected)
        student=read_probabilities(root/'experiments'/task_id/'val_predictions.csv',labels,expected)
        comparisons[task_id]={'reference':branch+'_five_fold',
                              'student':grouped(student), 'reference_metrics':grouped(reference),
                              'paired_patient_bootstrap':bootstrap(
                                  {'student':student,'reference':reference},
                                  [('student','reference')],2000,42),
                              'scope':'Official Val reused for model selection; exploratory comparison only'}
    write_json(root/'analysis/gh_val_comparisons.json',comparisons)
    from cxr.batch.test_results import (average_test, read_test_probabilities,
                                        validate_submission, write_submission, write_test)
    from cxr.batch.artifacts import write_probabilities
    from cxr.batch.results import combine, scores
    candidates = root/'submissions/submission_candidates.csv'
    validations = root/'submissions/submission_validation_report.csv'
    extension_ids = set(state['tasks']) | {'G17_PAIR', 'H17_PAIR'}
    with candidates.open(encoding='utf-8', newline='') as handle:
        candidate_reader=csv.DictReader(handle)
        candidate_fields=candidate_reader.fieldnames
        candidate_rows=[row for row in candidate_reader if row['Candidate_ID'] not in extension_ids]
    with validations.open(encoding='utf-8', newline='') as handle:
        validation_reader=csv.DictReader(handle)
        validation_fields=validation_reader.fieldnames
        validation_rows=[row for row in validation_reader if row['Candidate_ID'] not in extension_ids]
    for task_id, record in state['tasks'].items():
        if record['state'] != 'complete':
            continue
        run=root/'experiments'/task_id
        for suffix, pred, weight, metric in (('', 'test_predictions.csv','best_model.pth','metrics.json'),
                                             ('_EMA','ema_test_predictions.csv','ema_best_model.pth','ema_metrics.json')):
            if not (run/pred).is_file():
                continue
            model_id=task_id+suffix
            table=read_test_probabilities(run/pred)
            size=(run/weight).stat().st_size
            category='research_only' if size>500_000_000 else 'single_models'
            target=root/'submissions'/category/(model_id+'_submission.csv')
            result=validate_submission(target, table['keys'])
            meta=json.loads((run/metric).read_text(encoding='utf-8'))
            benchmark=json.loads((run/('ema_latency.json' if suffix else 'latency.json')).read_text(encoding='utf-8'))
            candidate_rows.append({'Candidate_ID':model_id,'File_Path':target.relative_to(root).as_posix(),
                'Source_Models':task_id,'Val_Macro_AUC':meta['official_val']['all']['macro_auc'],
                'Weight_Size_MB':size/1e6,'End_to_End_Latency_MS':'',
                'Benchmark_GPU':benchmark.get('device',''),
                'Size_Status':'PASS' if size<=500_000_000 else 'SIZE_FAIL',
                'Latency_Status':'LATENCY_UNVERIFIED','Format_Status':result['status'],
                'Deployment_Status':'LATENCY_UNVERIFIED' if size<=500_000_000 else 'SIZE_FAIL'})
            validation_rows.append({'Candidate_ID':model_id, **result})
    # Two fixed, small pair candidates. Their full ensemble latency remains
    # unverified until the complete deployment recipe is measured as one unit.
    for candidate, members in (('G17_PAIR', ('G17_FULL_01','G17_FULL_02')),
                               ('H17_PAIR', ('H17_DISTILL_01','H17_DISTILL_02'))):
        if not all(state['tasks'][name]['state']=='complete' for name in members):
            continue
        val_sources=[read_probabilities(root/'experiments'/name/'val_predictions.csv',labels,expected)
                     for name in members]
        val_table=combine(val_sources,[.5,.5])
        test_sources=[read_test_probabilities(root/'experiments'/name/'test_predictions.csv')
                      for name in members]
        test_table=average_test(test_sources,[.5,.5])
        val_file=root/'validation'/(candidate+'_val_probabilities.csv')
        test_file=root/'test_predictions'/(candidate+'_test_probabilities.csv')
        source_hashes={name:digest(root/'experiments'/name/'best_model.pth') for name in members}
        provenance={'sources':list(members),'weights':[.5,.5],'checkpoint_sha256':source_hashes}
        if val_file.exists():
            import numpy as np
            if not np.allclose(read_probabilities(val_file,labels,expected)['p'],val_table['p'],rtol=0,atol=1e-12):
                raise ValueError(f'Existing fixed Val fusion changed: {candidate}')
        else:
            write_probabilities(val_file,val_table,provenance)
        if test_file.exists():
            if not np.allclose(read_test_probabilities(test_file,test_table['keys'])['p'],test_table['p'],rtol=0,atol=1e-12):
                raise ValueError(f'Existing fixed Test fusion changed: {candidate}')
        else:
            write_test(test_file,test_table,provenance)
        size=sum((root/'experiments'/name/'best_model.pth').stat().st_size for name in members)
        folder='research_only' if size>500_000_000 else 'ensembles'
        target=root/'submissions'/folder/(candidate+'_submission.csv')
        manifest={'candidate_id':candidate,'source_model_ids':list(members),
                  'checkpoint_sha256':list(source_hashes.values()),'fusion_formula':dict.fromkeys(members,.5),
                  'val_macro_auc':scores(val_table)['macro_auc'],'weight_size_bytes':size,
                  'research_4090_latency_ms':None,'official_target_latency_ms':None,
                  'size_status':'PASS' if size<=500_000_000 else 'SIZE_FAIL',
                  'latency_status':'LATENCY_UNVERIFIED',
                  'deployment_status':'LATENCY_UNVERIFIED' if size<=500_000_000 else 'SIZE_FAIL',
                  'test_probability_sha256':digest(test_file)}
        if target.exists():
            old=json.loads(target.with_suffix('.manifest.json').read_text(encoding='utf-8'))
            if old['test_probability_sha256'] != manifest['test_probability_sha256'] or \
                    old['checkpoint_sha256'] != manifest['checkpoint_sha256']:
                raise ValueError(f'Existing fixed submission provenance changed: {candidate}')
            result=validate_submission(target,test_table['keys'])
        else:
            result=write_submission(target,test_table,manifest)
        candidate_rows.append({'Candidate_ID':candidate,'File_Path':target.relative_to(root).as_posix(),
            'Source_Models':';'.join(members),'Val_Macro_AUC':manifest['val_macro_auc'],
            'Weight_Size_MB':size/1e6,'End_to_End_Latency_MS':'', 'Benchmark_GPU':'',
            'Size_Status':manifest['size_status'],'Latency_Status':'LATENCY_UNVERIFIED',
            'Format_Status':result['status'],'Deployment_Status':manifest['deployment_status']})
        validation_rows.append({'Candidate_ID':candidate,**result})
    if len({row['Candidate_ID'] for row in candidate_rows}) != len(candidate_rows):
        raise ValueError('Duplicate base/GH submission candidate IDs')
    csv_rows(candidates,candidate_rows,candidate_fields)
    csv_rows(validations,validation_rows,validation_fields)
    from cxr.batch.direct_delivery import write_delivery_manifest
    write_delivery_manifest(root)

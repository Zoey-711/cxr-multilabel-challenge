"""CPU-only, predeclared Test fusion and full-ten-class submission export."""
import csv
import json
from pathlib import Path

from cxr.batch.plan import digest, write_json
from cxr.batch.test_results import average_test, read_test_probabilities, write_submission, write_test, validate_submission


def emit_single_submission(run, expected_studies=1414):
    """Immediately persist validated candidates after a model's DONE, without GPU."""
    run=Path(run)
    if not (run/'DONE.json').exists():
        raise ValueError('Cannot emit a submission from an incomplete model')
    root=run.parent.parent
    outputs=[]
    for suffix, prediction, weight, metric in (
        ('','test_predictions.csv','best_model.pth','metrics.json'),
        ('_EMA','ema_test_predictions.csv','ema_best_model.pth','ema_metrics.json')):
        if not (run/prediction).exists():
            continue
        table=read_test_probabilities(run/prediction,expected_studies=expected_studies)
        weight_path=run/weight
        bytes_used=weight_path.stat().st_size
        folder='research_only' if bytes_used>500_000_000 else 'single_models'
        path=root/'submissions'/folder/(run.name+suffix+'_submission.csv')
        details=json.loads((run/metric).read_text(encoding='utf-8'))
        latency_file=run/('ema_latency.json' if suffix else 'latency.json')
        research=json.loads(latency_file.read_text(encoding='utf-8')) if latency_file.exists() else {}
        manifest={'candidate_id':run.name+suffix,'source_model_ids':[run.name],
                  'checkpoint_sha256':[digest(weight_path)],'fusion_formula':'single model',
                  'val_macro_auc':details.get('official_val',{}).get('all',{}).get('macro_auc'),
                  'weight_size_bytes':bytes_used,'official_target_latency_ms':None,
                  'research_4090_latency_ms':research.get('mean_ms') if '4090' in str(research.get('device','')) else None,
                  'research_benchmark_gpu':research.get('device'),
                  'size_status':'SIZE_FAIL' if bytes_used>500_000_000 else 'PASS',
                  'latency_status':'LATENCY_UNVERIFIED',
                  'deployment_status':'SIZE_FAIL' if bytes_used>500_000_000 else 'LATENCY_UNVERIFIED',
                  'test_probability_sha256':digest(run/prediction)}
        if path.exists():
            validate_submission(path,table['keys'],expected_studies)
            old=json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
            if old['test_probability_sha256']!=manifest['test_probability_sha256']:
                raise ValueError(f'Existing single submission differs: {path}')
        else:
            write_submission(path,table,manifest,expected_studies)
        outputs.append(path)
    return outputs


def _save_probabilities(path, table, provenance):
    if path.exists():
        old = read_test_probabilities(path, table['keys'], len(table['keys']))
        if not __import__('numpy').array_equal(old['p'], table['p']):
            raise ValueError(f'Existing frozen Test fusion differs: {path}')
        sidecar=path.with_suffix('.provenance.json')
        if not sidecar.exists():
            raise ValueError(f'Missing Test fusion provenance: {sidecar}')
        recorded=json.loads(sidecar.read_text(encoding='utf-8'))
        if recorded.get('csv_sha256')!=digest(path) or any(recorded.get(key)!=value for key,value in provenance.items()):
            raise ValueError(f'Existing Test fusion provenance differs: {path}')
        return
    write_test(path, table, provenance)


def generate_test_reports(root, plan, task_rows, val_tables, deployment, eligibility):
    root = Path(root)
    expected_studies=plan.get('expected_test_studies',1414)
    test_dir, submit_dir = root/'test_predictions', root/'submissions'
    test_dir.mkdir(exist_ok=True)
    for name in ('single_models', 'ensembles', 'research_only'):
        (submit_dir/name).mkdir(parents=True, exist_ok=True)
    test = {}
    for task in plan['tasks']:
        run = root/'experiments'/task['id']
        if not (run/'DONE.json').exists():
            continue
        for suffix, filename in (('', 'test_predictions.csv'), ('_EMA', 'ema_test_predictions.csv')):
            source = run/filename
            if not source.exists():
                continue
            key = task['id']+suffix
            provenance = source.with_suffix('.provenance.json')
            if not provenance.exists() or json.loads(provenance.read_text(encoding='utf-8'))['csv_sha256'] != digest(source):
                raise ValueError(f'Test probability provenance mismatch: {source}')
            test[key] = read_test_probabilities(source,expected_studies=expected_studies)
    incomplete = {}
    for branch in ('D17','C17','S17','N17'):
        for suffix in ('', '_EMA') if branch in ('C17','N17') else ('',):
            ids = [f'{branch}_F{i:02d}{suffix}' for i in range(1, 6)]
            name = branch+suffix+'_test_5fold'
            missing = [key for key in ids if key not in test]
            if missing:
                incomplete[name] = missing
                write_json(test_dir/(name+'.status.json'), {'state':'INCOMPLETE','missing':missing})
                continue
            table = average_test([test[key] for key in ids], [.2]*5,expected_studies)
            file = test_dir/(name+'.csv')
            _save_probabilities(file, table, {'sources':ids, 'source_sha256':{
                key:digest(root/'experiments'/key.replace('_EMA','')/
                           ('ema_test_predictions.csv' if key.endswith('_EMA') else 'test_predictions.csv')) for key in ids},
                'formula': 'equal weight 0.2 per fold', 'no_test_labels':True})
            write_json(test_dir/(name+'.status.json'), {'state':'COMPLETE','sources':ids,
                                                       'sha256':digest(file)})
            test[name] = table
    names = ('D17_test_5fold','C17_test_5fold','S17_test_5fold')
    if all(name in test for name in names):
        formula = dict(zip(names, (.25,.25,.5)))
        table = average_test([test[n] for n in names], list(formula.values()),expected_studies)
        _save_probabilities(test_dir/'E17_test_probabilities.csv', table,
                            {'formula':formula,'source_sha256':{n:digest(test_dir/(n+'.csv')) for n in names},
                             'no_test_labels':True})
        test['E17'] = table
    else:
        incomplete['E17'] = [name for name in names if name not in test]
    write_json(test_dir/'status.json', {'complete_candidates':list(test), 'incomplete':incomplete})

    by_name = {r['candidate']:r for r in eligibility}
    by_deployment = {r['candidate']:r for r in deployment}
    by_task = {r['model_id']:r for r in task_rows}
    candidate_rows, validation_rows = [], []
    for name, table in test.items():
        if name.endswith('_test_5fold'):
            branch = name.split('_test_5fold')[0]
            value_name = {'D17':'D5','C17':'C5','S17':'S5','N17':'N5',
                          'C17_EMA':'C5_EMA','N17_EMA':'N5_EMA'}[branch]
            category = 'ensembles'
        else:
            value_name = name
            category = 'single_models' if '_F' in name else 'ensembles'
        gate = by_name.get(value_name, {'status':'ACCURACY_PENDING','bytes':None,'latency_ms':None,'hardware':None})
        deployment_meta = by_deployment.get(value_name, {})
        folder = 'research_only' if gate['status'] in ('SIZE_FAIL','LATENCY_FAIL') else category
        path = submit_dir/folder/(name+'_submission.csv')
        sources = deployment_meta.get('dependencies', [])
        benchmark= (root/'deployment/research_benchmarks'/(value_name+'.json'))
        research=json.loads(benchmark.read_text(encoding='utf-8')) if benchmark.exists() else None
        if research is None and category=='single_models':
            single=root/'experiments'/name.replace('_EMA','')/'inference_benchmark.json'
            benchmark_info=json.loads(single.read_text(encoding='utf-8')) if single.exists() else {}
            research=benchmark_info.get('ema_research_single_model' if name.endswith('_EMA') else 'research_single_model')
        manifest = {'candidate_id':name, 'source_model_ids':[Path(p['path']).parts[1] if p['path'].startswith('experiments/') else p['path'] for p in sources],
                    'checkpoint_sha256':[p['sha256'] for p in sources],
                    'fusion_formula': plan['fixed_fusions']['E17'] if name == 'E17' else
                                      'five fold equal mean' if category == 'ensembles' else 'single model',
                    'val_macro_auc':deployment_meta.get('val_auc') or by_task.get(name,{}).get('val_auc'),
                    'weight_size_bytes':gate['bytes'],
                    'research_4090_latency_ms':research.get('mean_ms') if research and '4090' in str(research.get('device','')) else None,
                    'research_benchmark_gpu':research.get('device') if research else None,
                    'official_target_latency_ms':gate['latency_ms'],
                    'size_status':'SIZE_FAIL' if gate['status']=='SIZE_FAIL' else 'PASS' if gate['bytes'] is not None else 'SIZE_UNVERIFIED',
                    'latency_status':gate['status'] if gate['status'] in ('LATENCY_FAIL','LATENCY_UNVERIFIED') else 'LATENCY_UNVERIFIED',
                    'deployment_status':gate['status'], 'test_probability_sha256': digest(
                        (root/'experiments'/name.replace('_EMA','')/
                         ('ema_test_predictions.csv' if name.endswith('_EMA') else 'test_predictions.csv'))
                        if category == 'single_models' else test_dir/(name+'.csv' if name != 'E17' else 'E17_test_probabilities.csv'))}
        if path.exists():
            validation = validate_submission(path, table['keys'],expected_studies)
            old = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
            if old['test_probability_sha256'] != manifest['test_probability_sha256']:
                raise ValueError(f'Existing submission provenance differs: {path}')
        else:
            validation = write_submission(path, table, manifest,expected_studies)
        candidate_rows.append({'Candidate_ID':name,'File_Path':path.relative_to(root).as_posix(),
            'Source_Models':';'.join(manifest['source_model_ids']),'Val_Macro_AUC':manifest['val_macro_auc'],
            'Weight_Size_MB':None if gate['bytes'] is None else gate['bytes']/1e6,
            'End_to_End_Latency_MS':gate['latency_ms'],'Benchmark_GPU':gate['hardware'] or manifest['research_benchmark_gpu'],
            'Size_Status':manifest['size_status'],'Latency_Status':manifest['latency_status'],
            'Format_Status':validation['status'],'Deployment_Status':gate['status']})
        validation_rows.append({'Candidate_ID':name, **validation})
    for filename, rows, fields in (
        ('submission_candidates.csv',candidate_rows,['Candidate_ID','File_Path','Source_Models','Val_Macro_AUC','Weight_Size_MB','End_to_End_Latency_MS','Benchmark_GPU','Size_Status','Latency_Status','Format_Status','Deployment_Status']),
        ('submission_validation_report.csv',validation_rows,['Candidate_ID','status','study_count','rows','classes_per_study','sha256'])):
        with (submit_dir/filename).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return {'candidates':len(candidate_rows),'incomplete':incomplete}

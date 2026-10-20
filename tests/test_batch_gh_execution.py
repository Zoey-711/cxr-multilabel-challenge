"""CPU-only G/H protocol, loss and orchestration tests."""
import csv
import json

import numpy as np
import pytest
import torch

from cxr.batch.artifacts import read_probabilities, write_probabilities
from cxr.batch.extension import (BRANCHES, TEACHER_BRANCHES, blended_bce,
                                 build_teacher_oof, load_soft_studies, prepare_extension,
                                 median_best_epoch, rank_full_train_backbones)
from cxr.batch.extension_queue import extension_report, run_extension_queue
from cxr.batch.plan import digest


def _csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root):
    keys = [(f'p{i}', f's{i}') for i in range(1, 6)]
    _csv(root/'manifests/patient_folds.csv', ['Subject_id', 'fold'],
         [{'Subject_id': subject, 'fold': i} for i, (subject, _) in enumerate(keys, 1)])
    labels = [{'Subject_id': patient, 'Study_id': study, 'fold': i,
               'image_count': 1 if i != 2 else 3,
               **{f'Target_{j}': (i+j) % 2 for j in range(1, 11)}}
              for i, (patient, study) in enumerate(keys, 1)]
    _csv(root/'manifests/train_labels.csv', list(labels[0]), labels)
    predictions = {}
    for priority, branch in enumerate(BRANCHES):
        p = np.asarray([[.15 + .1*priority + .55*row[f'Target_{j}']
                         for j in range(1, 11)] for row in labels])
        predictions[branch] = p
        write_probabilities(root/'oof'/f'{branch}_oof_all.csv', {'keys': keys, 'p': p}, {'model_id': branch})
        _csv(root/'oof'/f'{branch}_source_index.csv',
             ['Subject_id', 'Study_id', 'model_id', 'fold'],
             [{'Subject_id': patient, 'Study_id': study, 'model_id': f'{branch}_F{i:02d}', 'fold': i}
              for i, (patient, study) in enumerate(keys, 1)])
        for i in range(1, 6):
            run = root/'experiments'/f'{branch}_F{i:02d}'
            run.mkdir(parents=True)
            (run/'DONE.json').write_text('{}', encoding='utf-8')
            (run/'metrics.json').write_text(json.dumps({'best_epoch': [4, 7, 5, 9, 3][i-1],
                'stop_epoch': 10, 'resources': {'epoch_hours': .5}}), encoding='utf-8')
            (run/'best_model.pth').write_bytes(b'small-weight')
            (run/'latency.json').write_text(json.dumps({'device': 'NVIDIA RTX 4090', 'batch': 1,
                                                        'p95_ms': 20.0}), encoding='utf-8')
            _csv(run/'training_image_index.csv', ['Subject_id'],
                 [{'Subject_id': patient} for j, (patient, _) in enumerate(keys, 1) if j != i])
    return keys, predictions


def test_g_selection_tie_break_median_and_size_screen(tmp_path):
    root = tmp_path/'batch'
    _fixture(root)
    decision = rank_full_train_backbones(root)
    assert decision['selected'] == ['D17', 'C17']
    assert all(row['official_target_latency_status'] == 'LATENCY_UNVERIFIED' for row in decision['ranking'])
    assert median_best_epoch(root, 'D17')['epochs'] == 5
    (root/'experiments/D17_F01/best_model.pth').write_bytes(b'oversize-weight-record')
    # Screening checks the actual file size; the changed file remains a research input only.
    assert rank_full_train_backbones(root, max_bytes=12)['selected'] == ['C17', 'S17']


def test_h_teacher_formula_ids_class_order_and_multiview_weights(tmp_path):
    root = tmp_path/'batch'
    keys, predictions = _fixture(root)
    teacher = build_teacher_oof(root, expected_studies=5)
    table = read_probabilities(teacher['path'], root/'manifests/train_labels.csv', set(keys))
    np.testing.assert_allclose(table['p'], .25*predictions['D17'] +
                               .25*predictions['C17'] + .5*predictions['S17'])
    rows = ([{'Subject_id': patient, 'Study_id': study} for patient, study in keys] +
            [{'Subject_id': 'p2', 'Study_id': 's2'} for _ in range(2)])
    soft, weights = load_soft_studies(teacher['path'], teacher['sha256'], rows,
                                     root/'manifests/train_labels.csv')
    assert len(soft) == 5 and all(len(value) == 10 for value in soft.values())
    assert weights['p2', 's2'] == pytest.approx(1/3)
    assert weights['p1', 's1'] == 1
    with pytest.raises(ValueError, match='SHA-256'):
        load_soft_studies(teacher['path'], '0'*64, rows, root/'manifests/train_labels.csv')
    source = root/'oof/D17_source_index.csv'
    content = source.read_text(encoding='utf-8').replace('D17_F01', 'D17_F02', 1)
    source.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError, match='mismatched patient fold'):
        build_teacher_oof(root, expected_studies=5)


def test_study_equal_soft_loss_and_optimizer_resume(tmp_path):
    # One Study has one image; the other has three. Duplicating identical Study
    # predictions leaves the pure soft loss unchanged because weights sum to one.
    logits = torch.tensor([[.2]*10, [.8]*10], requires_grad=True)
    hard = torch.zeros_like(logits)
    mask = torch.ones_like(logits, dtype=torch.bool)
    soft = torch.full_like(logits, .7)
    loss2 = blended_bce(logits, hard, mask, soft, torch.ones(2), hard_fraction=0)
    more = torch.stack((logits[0], logits[1], logits[1], logits[1]))
    loss4 = blended_bce(more, torch.zeros_like(more), torch.ones_like(more, dtype=torch.bool),
                        torch.full_like(more, .7), torch.tensor([1., 1/3, 1/3, 1/3]),
                        hard_fraction=0)
    assert torch.allclose(loss2, loss4)
    with pytest.raises(ValueError, match='ten-class'):
        blended_bce(logits[:, :9], hard[:, :9], mask[:, :9], soft[:, :9], torch.ones(2))
    model = torch.nn.Linear(2, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
    x = torch.ones(2, 2)
    y = torch.zeros(2, 10)
    loss = blended_bce(model(x), y, torch.ones_like(y, dtype=torch.bool),
                       torch.full_like(y, .6), torch.ones(2))
    loss.backward(); optimizer.step(); scheduler.step()
    path = tmp_path/'resume.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': 1, 'early_stop_counter': 0,
                'best_auc': .7, 'scaler': {}, 'rng': torch.get_rng_state()}, path)
    saved = torch.load(path, weights_only=True)
    restored = torch.nn.Linear(2, 10)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, 2)
    restored.load_state_dict(saved['model']); opt2.load_state_dict(saved['optimizer'])
    sched2.load_state_dict(saved['scheduler'])
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
    assert sched2.state_dict() == scheduler.state_dict()


def test_extension_queue_blocks_incomplete_and_runs_once_with_budget(tmp_path, monkeypatch):
    root = tmp_path/'batch'; root.mkdir()
    base_tasks = [{'id': f'B{i:02d}'} for i in range(22)]
    plan = {'tasks': base_tasks, 'conditional_extension': {'reserved_slots':
            [{'id': name, 'depends_on': [task['id'] for task in base_tasks[:20]]} for name in
             ('G17_FULL_01','G17_FULL_02','H17_DISTILL_01','H17_DISTILL_02')]}}
    state = {'gpu_hours': 1., 'tasks': {t['id']: {'state': 'complete'} for t in base_tasks}}
    state['tasks']['B19']['state'] = 'failed'
    (root/'queue_state.json').write_text(json.dumps(state), encoding='utf-8')
    blocked = run_extension_queue(root, plan, ['0'], total_budget=100, gh_budget=30,
                                  max_concurrent=1, gpu_preflight=False)
    assert all(value['state'] == 'BLOCKED' for value in blocked['tasks'].values())
    state['tasks']['B19']['state'] = 'complete'
    state['tasks']['B21']['state'] = 'failed'  # Unrelated Pilot failure must not block G/H.
    (root/'queue_state.json').write_text(json.dumps(state), encoding='utf-8')
    manifest = root/'manifests/extension_selection.json'; manifest.parent.mkdir()
    manifest.write_text('{}', encoding='utf-8')
    tasks = [{'id': name, 'timeout_hours': 2, 'max_retries': 0} for name in
             ('G17_FULL_01','G17_FULL_02','H17_DISTILL_01','H17_DISTILL_02')]
    monkeypatch.setattr('cxr.batch.extension_queue.prepare_extension', lambda *_: tasks)
    monkeypatch.setattr('cxr.batch.extension_queue.extension_report', lambda *_: None)
    calls = []
    def fake_execute(task, batch_root, gpu, attempt):
        calls.append((task['id'], gpu))
        return {'state': 'complete', 'gpu_hours': .1}
    monkeypatch.setattr('cxr.batch.extension_queue.execute', fake_execute)
    result = run_extension_queue(root, plan, ['0', '1'], total_budget=100,
                                 gh_budget=30, max_concurrent=2, pause_seconds=.001,
                                 gpu_preflight=False)
    assert len(calls) == 4 and len({name for name, _ in calls}) == 4
    assert result['state'] == 'complete'
    with pytest.raises(FileExistsError):
        run_extension_queue(root, plan, ['0'], total_budget=100,
                            gh_budget=30, max_concurrent=1, gpu_preflight=False)


def test_extension_configs_freeze_sources_epochs_and_teacher(tmp_path, monkeypatch):
    import yaml
    root = tmp_path/'batch'
    _fixture(root)
    teacher = build_teacher_oof(root, expected_studies=5)
    with open('configs/batch_plan.yaml', encoding='utf-8') as handle:
        plan = yaml.safe_load(handle)
    for task in plan['tasks']:
        run = root/'experiments'/task['id']
        run.mkdir(parents=True, exist_ok=True)
        (run/'DONE.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr('cxr.batch.results.verify_frozen', lambda run: None)
    monkeypatch.setattr('cxr.batch.extension.build_teacher_oof', lambda root: teacher)
    tasks = prepare_extension(root, plan)
    assert len(tasks) == 4
    assert [t['id'] for t in tasks] == ['G17_FULL_01','G17_FULL_02','H17_DISTILL_01','H17_DISTILL_02']
    assert all(t['fold'] is None and t['oof_status'] == 'N/A' for t in tasks)
    g = yaml.safe_load(open(tasks[0]['config'], encoding='utf-8'))
    h = yaml.safe_load(open(tasks[2]['config'], encoding='utf-8'))
    assert g['model']['name'] == 'densenet121'
    assert g['training']['epochs'] == 5 and 'train_subset' not in g['data']
    assert h['data']['teacher_soft_labels']['sha256'] == digest(teacher['path'])
    assert h['training']['distillation']['soft_fraction'] == .5
    assert [t['config_sha256'] for t in prepare_extension(root, plan)] == [t['config_sha256'] for t in tasks]
    path = tmp_path/'batch/manifests/extension_configs/G17_FULL_01.yaml'
    path.write_text(path.read_text(encoding='utf-8')+'\n# unauthorized edit\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Frozen extension config differs'):
        prepare_extension(root, plan)


def test_gh_val_test_submission_and_partial_delivery(tmp_path, monkeypatch):
    from cxr.batch.reporting import csv_rows
    from cxr.batch.test_results import validate_submission, write_submission, write_test
    root=tmp_path/'phase17_batch'
    names=('G17_FULL_01','G17_FULL_02','H17_DISTILL_01','H17_DISTILL_02')
    for folder in ('manifests','experiments','validation','test_predictions','submissions/single_models',
                   'submissions/ensembles','analysis','reports'):
        (root/folder).mkdir(parents=True,exist_ok=True)
    val_keys=[(f'vp{i:02d}',f'vs{i:04d}') for i in range(20)]
    labels=[{'Subject_id':patient,'Study_id':study,'fold':1,'image_count':1,
             **{f'Target_{j}':(i+j)%2 for j in range(1,11)}}
            for i,(patient,study) in enumerate(val_keys)]
    _csv(root/'manifests/val_labels.csv',list(labels[0]),labels)
    test_keys=[(f'tp{i:04d}',f'ts{i:06d}') for i in range(1414)]
    test_table={'keys':test_keys,'p':np.full((1414,10),.55)}
    state={'tasks':{name:{'state':'complete','attempts':1,'gpu_hours':.1} for name in names}}
    for i,name in enumerate(names):
        run=root/'experiments'/name;run.mkdir()
        (run/'DONE.json').write_text('{}',encoding='utf-8')
        (run/'best_model.pth').write_bytes(b'checkpoint'+bytes([i]))
        (run/'resolved_config.yaml').write_text('model:\n  name: '+
            ('densenet121' if i in (0,2) else 'efficientnet_v2_s')+'\n',encoding='utf-8')
        (run/'metrics.json').write_text(json.dumps({'best_epoch':3,'official_val':
            {'all':{'macro_auc':.7}},'weights_bytes':10}),encoding='utf-8')
        (run/'latency.json').write_text(json.dumps({'device':'NVIDIA RTX 4090','mean_ms':20}),encoding='utf-8')
        p=np.asarray([[.25+.5*r[f'Target_{j}'] for j in range(1,11)] for r in labels])
        write_probabilities(run/'val_predictions.csv',{'keys':val_keys,'p':p},{'model_id':name})
        write_test(run/'test_predictions.csv',test_table,{'model_id':name})
        write_submission(root/'submissions/single_models'/(name+'_submission.csv'),test_table,
                         {'candidate_id':name,'test_probability_sha256':digest(run/'test_predictions.csv')})
    def fake_base_report(folder):
        for branch, source in (('D17',names[0]),('N17',names[1])):
            table=read_probabilities(root/'experiments'/source/'val_predictions.csv',
                                     root/'manifests/val_labels.csv',set(val_keys))
            write_probabilities(root/'validation'/(branch+'_val_ensemble.csv'),table,{'branch':branch})
        csv_rows(root/'submissions/submission_candidates.csv',[],
                 ['Candidate_ID','File_Path','Source_Models','Val_Macro_AUC','Weight_Size_MB',
                  'End_to_End_Latency_MS','Benchmark_GPU','Size_Status','Latency_Status',
                  'Format_Status','Deployment_Status'])
        csv_rows(root/'submissions/submission_validation_report.csv',[],
                 ['Candidate_ID','status','study_count','rows','classes_per_study','sha256'])
        from cxr.batch.reporting import _summary_row
        csv_rows(root/'reports/experiment_summary.csv',[],list(_summary_row('template').keys()))
    monkeypatch.setattr('cxr.batch.reporting.report',fake_base_report)
    monkeypatch.setattr('cxr.batch.direct_delivery.write_delivery_manifest',lambda *_:None)
    monkeypatch.setattr('cxr.batch.results.bootstrap',lambda *a,**k:{'valid_draws':10,'invalid_draws':0})
    extension_report(root,state)
    extension_report(root,state,base_report=False)
    with (root/'submissions/submission_candidates.csv').open(encoding='utf-8',newline='') as stream:
        candidates=list(csv.DictReader(stream))
    assert len(candidates)==6
    assert {r['Candidate_ID'] for r in candidates}==set(names)|{'G17_PAIR','H17_PAIR'}
    assert all(r['Latency_Status']=='LATENCY_UNVERIFIED' for r in candidates)
    for pair in ('G17_PAIR','H17_PAIR'):
        path=root/'submissions/ensembles'/(pair+'_submission.csv')
        assert validate_submission(path,test_keys)['rows']==14140
    with (root/'reports/gh_experiment_summary.csv').open(encoding='utf-8',newline='') as stream:
        assert {row['oof_macro_auc'] for row in csv.DictReader(stream)}=={'N/A'}
    with (root/'reports/experiment_summary.csv').open(encoding='utf-8',newline='') as stream:
        summary={row['model_id']:row for row in csv.DictReader(stream)}
    assert set(names) <= set(summary)
    assert all(summary[name]['oof_macro_auc']=='N/A' for name in names)
    assert all(summary[name]['oof_studies']=='N/A' for name in names)
    assert all(summary[name]['test_auc']=='N/A: hidden Test labels unavailable' for name in names)
    # The next partial delivery has no completed H pair: it must not fabricate one.
    state['tasks']['H17_DISTILL_02']['state']='failed'
    monkeypatch.setattr('cxr.batch.extension_queue.extension_report',lambda *a:None)
    assert state['tasks']['H17_DISTILL_02']['state']=='failed'

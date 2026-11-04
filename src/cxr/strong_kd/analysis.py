"""Fixed paired comparisons, never seed selection or independent student OOF."""
from pathlib import Path
from cxr.batch.artifacts import read_probabilities
from cxr.phase41.analysis import auc_summary
from cxr.phase42.analysis import paired_bootstrap
from cxr.phase42.protocol import verify_receipt
from cxr.phase39_support import write_csv, write_json
from cxr.strong_kd.protocol import tasks, task_spec
from cxr.strong_kd.teachers import rows


def analyze(cfg):
    from cxr.strong_kd.runner import student_decision
    root = Path(cfg['output_root'])
    labels = Path(cfg['teacher_root'])/'manifests/val_labels.csv'
    truth = {(r['Subject_id'], r['Study_id']): [int(r[f'Target_{j}']) for j in range(1,11)] for r in rows(labels)}
    keys = sorted(truth)
    report = dict(state='INCOMPLETE', student_oof='N/A', evaluation='EXPLORATORY_OFFICIAL_VAL',
                  models={}, comparisons={}, seed_decisions={}, missing=[])
    predictions = {}; metrics_rows = []; classes = []; comparisons = []
    for task in tasks(cfg):
        name = task['id']; folder = root/'experiments'/name
        if not (folder/'DONE.json').is_file():
            report['missing'].append(name); continue
        verify_receipt(folder)
        for version in ('raw', 'ema'):
            p = read_probabilities(folder/f'{version}_val_predictions.csv', labels, set(keys))
            values = dict(zip(p['keys'], p['p']))
            tag = name+'/'+version; predictions[tag] = values
            m = auc_summary(truth, values, keys); report['models'][tag] = m
            metrics_rows.append(dict(task=name, version=version, seed=task_spec(name)[1], macro_auc=m['macro_auc']))
            classes.extend(dict(task=name, version=version, class_id=j+1, auc=a,
                                positive_studies=m['positive_studies'][j]) for j,a in enumerate(m['per_class_auc']))
    for seed in ([42,2026] if cfg.get('replicate_seed_enabled',False) else [42]):
        suffix = '' if seed == 42 else '_SEED2026'
        gates = []
        for version in ('raw','ema'):
            for base, candidate in (('STUDENT_BASE','SKD_CONTROL'),('STUDENT_BASE','SKD_COMPLEMENT'),('SKD_CONTROL','SKD_COMPLEMENT')):
                a = base+suffix+'/'+version; b = candidate+suffix+'/'+version
                if a not in predictions or b not in predictions: continue
                boot = paired_bootstrap(truth, predictions[a], predictions[b], keys, draws=2000, seed=5417)
                pair = dict(control=report['models'][a], complement=report['models'][b], bootstrap=boot)
                report['comparisons'][a+' -> '+b] = pair
                comparisons.append(dict(seed=seed, version=version, base=base, candidate=candidate,
                                         delta=boot['delta'], ci_low=boot['ci95'][0], ci_high=boot['ci95'][1]))
                if version == 'raw' and candidate == 'SKD_COMPLEMENT':
                    gates.append(student_decision(pair)[0] == 'CONSIDER_NESTED_TEACHER_PILOT')
        report['seed_decisions'][str(seed)] = ('PASS_EXPLORATORY' if len(gates)==2 and all(gates)
                                               else 'NO_CONFIRMED_SIGNAL' if len(gates)==2 else 'PENDING')
    if not report['missing']:
        report['state'] = 'COMPLETE_DEVELOPMENT_ONLY'
    report['decision'] = ('WAIT_FOR_PAIRED_STUDENTS' if report['missing'] else
                          'CONSIDER_NESTED_TEACHER_PILOT' if all(v=='PASS_EXPLORATORY' for v in report['seed_decisions'].values())
                          else 'STOP_TEACHER_REBUILD')
    report['rule'] = 'Final raw primary; strong teacher student must beat BOTH BASE and CONTROL: delta >= .002, CI lower > 0, worst class delta >= -.01. All enabled seeds must pass. EMA secondary only.'
    report['limitations'] = 'Repeated official Val, multiple comparisons, and only one or two seeds: exploratory, not independent generalization confirmation. Student OOF=N/A. No automatic teacher rebuild or Submission.'
    out = root/'reports'; out.mkdir(parents=True,exist_ok=True)
    write_json(out/'student_comparison.json',report)
    if metrics_rows: write_csv(out/'student_metrics.csv',metrics_rows)
    if classes: write_csv(out/'student_per_class.csv',classes)
    if comparisons: write_csv(out/'bootstrap_comparisons.csv',comparisons)
    lines = ['# 蒸馏受控实验报告', '', '状态：'+report['state'], '决策：'+report['decision'],
             '', '学生 OOF：N/A；官方 Val 仅用于探索性开发评估。',
             'raw 固定最后一轮是主要比较，EMA 单列；不挑选最高分种子。', '',
             '|任务|种子|版本|Val Macro-AUC|', '|---|---:|---|---:|']
    lines += [f"|{r['task']}|{r['seed']}|{r['version']}|{r['macro_auc']:.8f}|" for r in metrics_rows]
    lines += ['', '|种子|版本|比较|差值|95% CI|', '|---|---|---|---:|---|']
    lines += [f"|{r['seed']}|{r['version']}|{r['candidate']} − {r['base']}|{r['delta']:.8f}|[{r['ci_low']:.8f}, {r['ci_high']:.8f}]|" for r in comparisons]
    lines += ['', '未完成任务：'+(', '.join(report['missing']) or '无'), '', report['rule'], '', report['limitations'],
              '', 'T4_UNVERIFIED；没有自动生成或上传 Submission。']
    (out/'final_decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report

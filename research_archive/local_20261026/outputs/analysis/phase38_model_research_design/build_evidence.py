"""CPU-only extraction of existing audited statistics; no model/data mutation."""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
WORK = PROJECT.parent
sources = []

def read(path):
    sources.append(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

rows = []
p25 = PROJECT / 'outputs/analysis/phase25_local_pilots/aggregation_oof_comparison.csv'
for r in read(p25):
    if r['rule'] == 'Mean' and r['group'] == 'all':
        for c in range(1, 11):
            rows.append(dict(model='LOCAL_' + r['model'], scope='FULL_OOF', class_id=c,
                             auc=float(r[f'auc_{c}']), source=str(p25)))
p28 = WORK / 'cxr-friend-results/analysis/phase28_cbfc5997/per_class_auc_and_correlation.csv'
for r in read(p28):
    if r['candidate'] in ('FRIEND_E17', 'N17_EMA'):
        rows.append(dict(model=r['candidate'], scope='FULL_OOF', class_id=int(r['class_id']),
                         auc=float(r['oof_auc']), source=str(p28)))
p37 = WORK / 'cxr-friend-results/analysis/phase37_phase35_final/per_class_auc.csv'
for r in read(p37):
    if r['Scope'] == 'FULL':
        rows.append(dict(model=r['Variant'] + '_EMA', scope='FULL_OOF', class_id=int(r['Class']),
                         auc=float(r['AUC']), source=str(p37)))
p18 = PROJECT / 'outputs/analysis/phase18_leaderboard_diagnosis/e07_e08_per_class.csv'
for r in read(p18):
    rows.append(dict(model='E08', scope='OFFICIAL_VAL_REUSED', class_id=int(r['class_id']),
                     auc=float(r['e08_auc']), source=str(p18)))
save('per_class_evidence.csv', rows)
weak = []
for model in dict.fromkeys(r['model'] for r in rows):
    rr = sorted([r for r in rows if r['model'] == model], key=lambda x: x['auc'])
    weak.append(dict(model=model, scope=rr[0]['scope'], weakest_classes=','.join(str(r['class_id']) for r in rr[:3]),
                     auc_1=next(r['auc'] for r in rr if r['class_id'] == 1),
                     auc_4=next(r['auc'] for r in rr if r['class_id'] == 4),
                     auc_8=next(r['auc'] for r in rr if r['class_id'] == 8)))
save('weak_class_evidence.csv', weak)
view = PROJECT / 'outputs/analysis/phase19_actionable_improvements/study_view_stratification.csv'
save('study_view_evidence.csv', read(view))
costs = [
    dict(candidate='MSF384', extra_conv_linear_parameters=101568,
         extra_conv_linear_MACs=29823616, inference_module_removable=False),
    dict(candidate='LOCAL384', extra_conv_linear_parameters=1600,
         extra_conv_linear_MACs=921600, inference_module_removable=False),
    dict(candidate='LAYOUT384', extra_conv_linear_parameters=6400,
         extra_conv_linear_MACs=2950400, inference_module_removable=False),
]
for r in costs:
    r.update(cost_status='THEORETICAL_EXCLUDES_BIASES_NORMS_POOLING_AND_MEMORY', T4_status='T4_UNVERIFIED')
save('candidate_cost_estimates.csv', costs)
extra = [
    PROJECT / 'src/cxr/models/factory.py', PROJECT / 'src/cxr/models/study_pool.py',
    PROJECT / 'src/cxr/models/study_residual.py', PROJECT / 'src/cxr/phase34_aux.py',
    PROJECT / 'outputs/experiments/phase34_aux_supervision/final_decision.md',
    PROJECT / 'outputs/analysis/phase25_local_pilots/label_semantics_audit.md',
    WORK / 'cxr-friend-results/analysis/phase27_cbfc5997_recomputed/phase27_handoff.md',
    WORK / 'cxr-friend-results/analysis/phase37_phase35_final/complete_oof_comparison.csv',
    WORK / 'cxr-friend-results/analysis/phase37_phase35_final/folds2_5_confirmation.csv',
]
manifest = [dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in dict.fromkeys(sources + extra)]
(OUT / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(weak, ensure_ascii=False, indent=2))
print('CPU extraction complete; no GPU/model execution.')

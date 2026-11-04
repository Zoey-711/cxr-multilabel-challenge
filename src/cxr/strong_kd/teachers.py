"""Audit frozen OOF lineage and quantify complementary rankings on Train only."""
import csv
import json
from pathlib import Path
import numpy as np
import yaml
from cxr.batch.artifacts import read_probabilities, write_probabilities
from cxr.phase39_support import write_csv, write_json
from cxr.phase41.analysis import auc_summary
from cxr.phase42.analysis import paired_bootstrap
from cxr.strong_kd.protocol import RECIPES, digest, source_lock


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def audit(root, lock=None):
    root = Path(root)
    lock = source_lock() if lock is None else lock
    for rel, sha in lock['files'].items():
        path = (root/rel).resolve()
        if root.resolve() not in path.parents or digest(path) != sha:
            raise ValueError('Teacher source SHA-256 mismatch: '+rel)
    labels_path = root/'manifests/train_labels.csv'
    label_rows = rows(labels_path)
    truth = {(r['Subject_id'], r['Study_id']): [int(r[f'Target_{j}']) for j in range(1, 11)] for r in label_rows}
    counts = {(r['Subject_id'], r['Study_id']): int(r['image_count']) for r in label_rows}
    pf = rows(root/'manifests/patient_folds.csv')
    folds = {r['Subject_id']: int(r['fold']) for r in pf}
    if (len(truth) != len(label_rows) or len(truth) != lock['studies'] or
        len(folds) != len(pf) or len(folds) != lock['patients'] or
        set(folds) != {k[0] for k in truth} or set(folds.values()) != set(range(1, 6)) or
        sum(counts.values()) != lock['images'] or min(counts.values()) < 1):
        raise ValueError('Frozen Train Study/patient/image coverage differs')
    sf = rows(root/'manifests/study_folds.csv')
    if len(sf) != len(truth) or {(r['Subject_id'], r['Study_id']) for r in sf} != set(truth):
        raise ValueError('Study Fold index differs')
    if any(int(r['fold']) != folds[r['Subject_id']] for r in sf + label_rows):
        raise ValueError('Patient assigned to multiple folds')
    val_patients = {r['Subject_id'] for r in rows(root/'manifests/val_labels.csv')}
    if val_patients & set(folds):
        raise ValueError('Train/Val patient overlap')
    tables, sources = {}, []
    fold_hashes = {name: lock['files']['manifests/'+name] for name in
                   ('patient_folds.csv', 'study_folds.csv', 'image_folds.csv')}
    for teacher in ('D17', 'C17', 'S17', 'N17_EMA'):
        table = {}
        branch = teacher.removesuffix('_EMA')
        for fold in range(1, 6):
            task = f'{branch}_F{fold:02d}'
            folder = root/'experiments'/task
            done = json.loads((folder/'DONE.json').read_text())
            manifest = json.loads((folder/'run_manifest.json').read_text())
            cfg = yaml.safe_load((folder/'resolved_config.yaml').read_text(encoding='utf-8'))
            if done.get('state') != 'complete' or done.get('model_id') != task or manifest['model_id'] != task:
                raise ValueError('Incomplete or mismatched teacher task: '+task)
            if (manifest['fold_hashes'] != fold_hashes or cfg['data']['train_subset']['fold'] != fold or
                cfg['data']['train_subset']['sha256'] != fold_hashes):
                raise ValueError('Teacher fold configuration differs: '+task)
            held = {p for p, f in folds.items() if f == fold}
            index = rows(folder/'training_image_index.csv')
            if {r['Subject_id'] for r in index} != set(folds)-held:
                raise ValueError('Teacher training-patient isolation failure: '+task)
            if len({(r['Subject_id'], r['Study_id'], r['Dicom_id']) for r in index}) != len(index):
                raise ValueError('Duplicate teacher training image')
            expected_images = sum(n for k, n in counts.items() if k[0] not in held)
            if len(index) != expected_images or any((r['Subject_id'], r['Study_id']) not in truth for r in index):
                raise ValueError('Teacher training image coverage differs')
            probability = 'ema_oof_predictions.csv' if teacher.endswith('_EMA') else 'oof_predictions.csv'
            weight = 'ema_best_model.pth' if teacher.endswith('_EMA') else 'best_model.pth'
            for name in ('run_manifest.json', 'resolved_config.yaml', probability):
                if done['verified_sha256'][name] != digest(folder/name):
                    raise ValueError('DONE does not bind teacher artifact: '+name)
            if manifest['training_image_index_sha256'] != digest(folder/'training_image_index.csv'):
                raise ValueError('Manifest does not bind training index')
            if manifest['files'][probability]['sha256'] != digest(folder/probability):
                raise ValueError('Manifest does not bind prediction')
            if manifest['files'][weight]['sha256'] != done['verified_sha256'][weight]:
                raise ValueError('Teacher checkpoint provenance mismatch')
            expected = {k for k in truth if k[0] in held}
            pred = read_probabilities(folder/probability, labels_path, expected)
            table.update(zip(pred['keys'], pred['p']))
            sources.append(dict(teacher=teacher, task=task, fold=fold,
                                prediction_sha256=digest(folder/probability),
                                training_index_sha256=digest(folder/'training_image_index.csv'),
                                checkpoint_sha256=done['verified_sha256'][weight],
                                checkpoint_downloaded=False))
        if set(table) != set(truth):
            raise ValueError('Incomplete fivefold teacher OOF')
        tables[teacher] = table
    return truth, counts, tables, dict(state='PASS', source_commit=lock['source_commit'],
        source_files=lock['files'], studies=len(truth), patients=len(folds), sources=sources,
        training_patient_overlap=0, student_oof='N/A', outer_fold_student_training='BLOCKED',
        limitation='Teacher OOF excludes each predicted patient from fitting, but teacher/model selection reused official Val. No independent student OOF claim.')


def blend(tables, recipe):
    if not np.isclose(sum(recipe.values()), 1.) or any(v < 0 for v in recipe.values()):
        raise ValueError('Invalid fixed teacher mixture')
    sets = [set(tables[name]) for name in recipe]
    if any(s != sets[0] for s in sets):
        raise ValueError('Teacher Study IDs differ')
    return {k: sum(w*tables[name][k] for name, w in recipe.items()) for k in sorted(sets[0])}


def prepare(root, output, *, draws=2000, lock=None):
    truth, counts, tables, record = audit(root, lock)
    output = Path(output)
    keys = sorted(truth)
    recipes = {name: blend(tables, recipe) for name, recipe in RECIPES.items()}
    all_tables = dict(tables, **recipes)
    summary = {name: auc_summary(truth, pred, keys) for name, pred in all_tables.items()}
    control, strong = recipes.values()
    boot = paired_bootstrap(truth, control, strong, keys, draws=draws, seed=5417)
    y = np.asarray([truth[k] for k in keys])
    a = np.asarray([control[k] for k in keys]); b = np.asarray([tables['N17_EMA'][k] for k in keys])
    c = np.asarray([strong[k] for k in keys])
    per_class = []
    rng = np.random.default_rng(5417)
    for j in range(10):
        pos, neg = np.flatnonzero(y[:, j] == 1), np.flatnonzero(y[:, j] == 0)
        ip, inn = rng.choice(pos, 20000), rng.choice(neg, 20000)
        ea, eb = a[ip, j] <= a[inn, j], b[ip, j] <= b[inn, j]
        corr = float(np.corrcoef(a[:, j], b[:, j])[0, 1]) if a[:, j].std() and b[:, j].std() else None
        per_class.append(dict(class_id=j+1, positive_studies=len(pos), negative_studies=len(neg),
            e17_auc=summary['SKD_CONTROL']['per_class_auc'][j],
            n17_ema_auc=summary['N17_EMA']['per_class_auc'][j],
            strong_auc=summary['SKD_COMPLEMENT']['per_class_auc'][j],
            probability_correlation=corr, mean_absolute_disagreement=float(np.abs(a[:, j]-b[:, j]).mean()),
            e17_wrong_n17_right_fraction=float(np.mean(ea & ~eb)),
            n17_wrong_e17_right_fraction=float(np.mean(eb & ~ea)),
            ranking_pairs=20000, ranking_ties='counted as incorrect',
            strong_brier=float(np.mean((c[:, j]-y[:, j])**2))))
    record.update(recipes=RECIPES, metrics=summary, bootstrap=boot,
                  ranking_pair_sampling='fixed seed 5417; sampled positive-negative Study pairs, descriptive only')
    target = output/'teacher_audit.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != record:
        raise ValueError('Existing teacher audit differs; use a new output directory')
    for name, pred in recipes.items():
        path = output/'teachers'/f'{name}.csv'
        values = np.asarray([pred[k] for k in keys])
        if path.exists():
            old = read_probabilities(path, Path(root)/'manifests/train_labels.csv', set(keys))
            if old['keys'] != keys or not np.array_equal(old['p'], values):
                raise ValueError('Existing frozen teacher targets changed')
            provenance = json.loads(path.with_suffix('.provenance.json').read_text())
            if (provenance['csv_sha256'] != digest(path) or provenance['recipe'] != RECIPES[name]
                or provenance['source_commit'] != record['source_commit']):
                raise ValueError('Existing teacher target provenance changed')
        else:
            write_probabilities(path, dict(keys=keys, p=values),
                                dict(recipe=RECIPES[name], source_commit=record['source_commit'], role='OOF soft training targets'))
    write_csv(output/'teacher_complementarity.csv', per_class)
    write_json(target, record)
    write_json(output/'teacher_files_sha256.json', {
        p.relative_to(output).as_posix(): digest(p) for p in output.glob('teachers/*') if p.is_file()})
    return record

import hashlib,json
from pathlib import Path
import pytest
from scripts import protocol_assets
from scripts.import_friend_results import import_result

def test_protocol_import_exact_hash_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(protocol_assets,'ROOT',tmp_path)
    data=b'fixed-private-metadata'
    plan={'artifacts':{'batch_assets/labels.csv':hashlib.sha256(data).hexdigest()},'fold_directory':'folds','fold_hashes':{}}
    source=tmp_path/'authorized';(source/'batch_assets').mkdir(parents=True)
    (source/'batch_assets/labels.csv').write_bytes(data)
    protocol_assets.import_directory(source,plan)
    assert (tmp_path/'batch_assets/labels.csv').read_bytes()==data
    protocol_assets.import_directory(source,plan)
    (tmp_path/'batch_assets/labels.csv').write_bytes(b'changed')
    with pytest.raises(FileExistsError):protocol_assets.import_directory(source,plan)

def test_import_rejects_unknown_commit_or_split_before_extract(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'manifests').mkdir()
    m={'git_commit':'unapproved','code_sha256':{},'fold_hashes':{},'data_sha256':'data',
       'completed_tasks':[],'important_files':{}}
    (bundle/'manifests/delivery_manifest.json').write_text(json.dumps(m))
    plan={'fold_hashes':{},'data_sha256':'data','artifacts':{},'tasks':[]}
    with pytest.raises(ValueError,match='commit'):import_result(bundle,tmp_path/'out',plan,'approved')
    assert not (tmp_path/'out').exists()


def test_import_accepts_frozen_gh_config_without_cluster_path(tmp_path, monkeypatch):
    source=tmp_path/'source'; run=source/'experiments/G17_FULL_01';run.mkdir(parents=True)
    (source/'manifests').mkdir()
    config=b'full-train-reviewed\n'
    (run/'resolved_config.yaml').write_bytes(config)
    (run/'run_manifest.json').write_text(json.dumps({'model_id':'G17_FULL_01',
        'fold_hashes':{'patient_folds.csv':'fixed'},'git_commit':'commit'}),encoding='utf-8')
    (source/'manifests/extension_selection.json').write_text(json.dumps({
        'source_fold_hashes':{'patient_folds.csv':'fixed'},
        'tasks':[{'id':'G17_FULL_01','config_sha256':hashlib.sha256(config).hexdigest()}]}),encoding='utf-8')
    manifest={'git_commit':'commit','fold_hashes':{'patient_folds.csv':'fixed'},
              'data_sha256':'dataset','code_sha256':{},'completed_tasks':['G17_FULL_01']}
    monkeypatch.setattr('scripts.import_friend_results.verify_direct_folder',lambda *a,**k:manifest)
    plan={'fold_hashes':{'patient_folds.csv':'fixed'},'data_sha256':'dataset',
          'artifacts':{},'tasks':[],
          'conditional_extension':{'reserved_slots':[{'id':'G17_FULL_01'}]}}
    (source/'manifests/delivery_manifest.json').write_text('{}',encoding='utf-8')
    destination=tmp_path/'independent_import'
    import_result(source,destination,plan,'commit')
    assert (destination/'experiments/G17_FULL_01/resolved_config.yaml').read_bytes()==config
    assert (destination/'IMPORT_VERIFIED.json').is_file()

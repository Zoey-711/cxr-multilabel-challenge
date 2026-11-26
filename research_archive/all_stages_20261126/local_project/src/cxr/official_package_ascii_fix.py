"""Repack the prepared kit with ASCII archive member names for Linux unzip."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from cxr.official_deployment_tools import lock_manifest, sha256, verify_package, write_json


RENAMES = {
    '提交前必读.md': 'READ_ME_FIRST.md',
    '模型核验材料与数据申明清单.xlsx': 'model_declaration.xlsx',
    'materials/history/旧版声明清单_历史待验收.xlsx': 'materials/history/legacy_declaration.xlsx',
}


def repair(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Choose a fresh, independent output directory')
    verify_package(source)
    output.mkdir(parents=True)
    count = 0
    for path in sorted(source.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(source).as_posix()
        if relative == 'manifest_sha256.json':
            continue
        translated = RENAMES.get(relative, relative)
        if not translated.isascii():
            raise ValueError('Unmapped non-ASCII archive path: '+relative)
        target = output/translated
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha256(path) != sha256(target):
            raise ValueError('Copied file differs: '+relative)
        count += 1
    note = output/'ASCII_PATH_FIX.md'
    note.write_text(
        '# 2026-11-11 Linux unzip filename repair\n\n'
        'All ZIP member names use ASCII. This avoids shell/unzip locale conversions of Chinese filenames.\n'
        'No model checkpoint, ONNX graph, submission probabilities, inference arithmetic or fusion weight changed.\n'
        'The declaration workbook is model_declaration.xlsx; old workbook copy is materials/history/legacy_declaration.xlsx.\n'
        'The organizer T4 build, full Test consistency and timing remain PENDING.\n', encoding='utf-8')
    if sha256(output/'submission.csv') != sha256(source/'submission.csv'):
        raise ValueError('Historical submission changed')
    for stem in ('phase65_source/predict.py', 'phase65_source/src/cxr/e08_s17_t4_cost.py',
                 'phase65_source/src/cxr/e08_s17_t4_full_audit.py'):
        if sha256(output/stem) != sha256(source/stem):
            raise ValueError('Inference code changed')
    for name in ('B01','B03','N01_EMA','S17_F3'):
        rel=f'phase65_source/model/{name}.pt'
        if sha256(output/rel) != sha256(source/rel):
            raise ValueError('Frozen checkpoint changed')
    for name in ('B02','M02'):
        for suffix in ('.pt','_fp32.onnx'):
            rel=f'build_sources/{name}{suffix}'
            if sha256(output/rel) != sha256(source/rel):
                raise ValueError('Engine source changed')
    records=lock_manifest(output)
    verify_package(output)
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                member='official_kit/'+path.relative_to(output).as_posix()
                if not member.isascii():
                    raise ValueError('ZIP member is not ASCII')
                z.write(path,member)
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('Duplicate ZIP member')
        for relative,expected in records.items():
            with z.open('official_kit/'+relative) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                    raise ValueError('ZIP content hash mismatch: '+relative)
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failure')
    result={'status':'PREPARED_PENDING_T4_BUILD','issue_fixed':'ASCII ZIP member names',
            'source_package':str(source),'archive':str(archive),'bytes':archive.stat().st_size,
            'sha256':sha256(archive),'files_verified':len(records),'zip_crc':'PASS',
            'zip_members_ascii':True,'submission_sha256':sha256(output/'submission.csv'),
            'all_model_and_inference_file_hashes_unchanged':True,'t4_latency':'UNVERIFIED',
            'full_test_consistency':'UNVERIFIED'}
    write_json(archive.with_suffix('.zip.sha256.json'),result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(repair(args.source,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

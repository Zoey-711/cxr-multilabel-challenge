"""Import immutable private fold/label metadata from an authorized directory."""
import argparse
import shutil
from pathlib import Path

import yaml

from cxr.batch.plan import ROOT, digest


def expected_assets(plan):
    assets = {k:v for k,v in plan['artifacts'].items() if k.startswith('batch_assets/')}
    assets.update({plan['fold_directory']+'/'+name:h for name,h in plan['fold_hashes'].items()})
    return assets


def import_directory(directory, plan):
    source_root = Path(directory).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    assets = expected_assets(plan)
    for name, expected in assets.items():
        source = source_root/name
        target = (ROOT/name).resolve()
        if not source.is_file() or digest(source) != expected:
            raise ValueError(f'Private protocol metadata missing or changed: {source}')
        if ROOT not in target.parents:
            raise ValueError('Unsafe protocol asset path')
        if target.exists() and digest(target) != expected:
            raise FileExistsError(f'Existing protocol asset differs: {target}')
    for name in assets:
        target = ROOT/name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root/name, target)
    print('Frozen protocol metadata verified; no split regenerated')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--action',required=True,choices=['import','verify'])
    parser.add_argument('--directory',required=True,type=Path)
    args=parser.parse_args()
    plan=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.action=='import':
        import_directory(args.directory,plan)
    else:
        for name, expected in expected_assets(plan).items():
            if digest(args.directory/name)!=expected:
                raise ValueError(f'Protocol asset changed: {name}')
        print('Protocol directory verified')


if __name__=='__main__':
    main()

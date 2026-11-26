"""Fetch the pinned complete deployment via authenticated GitHub CLI; no training."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path,PurePosixPath

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('release_lock.json'))
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--archive',type=Path,help='Use an already downloaded Release ZIP')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    target=args.destination.resolve()
    if target.exists():raise ValueError('Destination must be new; existing files are never overwritten')
    archive=args.archive
    if archive is None:
        archive=target.parent/cfg['asset']
        if not archive.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['gh','release','download',cfg['tag'],'--repo',cfg['repository'],'--pattern',cfg['asset'],'--dir',str(target.parent)],check=True)
    if archive.stat().st_size!=cfg['bytes'] or sha(archive)!=cfg['sha256']:raise ValueError('Release archive mismatch')
    with zipfile.ZipFile(archive) as z:
        members=z.infolist()
        for m in members:
            p=PurePosixPath(m.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in m.filename or ':' in m.filename or (m.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe archive member')
        manifest=json.loads(z.read('manifest_sha256.json'))
        for name,expected in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError('Manifest mismatch: '+name)
        target.mkdir(parents=True)
        z.extractall(target)
    print('Complete deployment extracted and verified:',target)
    print('GPU inference and Study CSV consistency are NOT verified by this download check.')

if __name__=='__main__':main()

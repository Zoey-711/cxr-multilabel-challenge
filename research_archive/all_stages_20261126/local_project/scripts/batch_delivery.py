"""Offline CPU reporting, verified packaging, and portable extraction."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--action',required=True,choices=['pack','extract','recompute'])
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--destination',type=Path)
    p.add_argument('--include-resume',action='store_true')
    p.add_argument('--part-bytes',type=int,default=1_000_000_000)
    a=p.parse_args()
    if not a.config.is_file():p.error('Config/manifest path missing')
    if a.action=='pack':
        from cxr.batch.delivery import pack
        print(pack(a.root,include_resume=a.include_resume,part_bytes=a.part_bytes))
    elif a.action=='extract':
        if not a.destination:p.error('--destination required')
        from cxr.batch.delivery import verify_and_extract
        print(verify_and_extract(a.root,a.destination))
    else:
        from cxr.batch.reporting import report
        result=report(a.root)
        print({k:v['all']['macro_auc'] for k,v in result['official_val'].items()})

if __name__=='__main__':main()

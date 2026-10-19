"""Recompute metrics or verify direct result folders; archives are not generated."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--action',required=True,choices=['recompute','verify'])
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    if not args.config.is_file():
        parser.error('Batch plan missing')
    if args.action=='recompute':
        from cxr.batch.reporting import report
        result=report(args.root)
        extension=args.root/'extension_state.json'
        if extension.exists():
            import json
            from cxr.batch.extension_queue import extension_report
            extension_report(args.root,json.loads(extension.read_text(encoding='utf-8')),base_report=False)
        print({key:value['all']['macro_auc'] for key,value in result['official_val'].items()})
    else:
        from cxr.batch.direct_delivery import verify_direct_folder
        print(verify_direct_folder(args.root))


if __name__=='__main__':
    main()

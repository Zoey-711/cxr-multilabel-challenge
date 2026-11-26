"""Val-only source diagnosis: restore E03's historical mixed 16/8 batches.

Not a deployment entry and not a search over precision or batch sizes.
"""
import argparse,json,time,zipfile,io
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from cxr.data.archive import read_split_rows,split_archive_path,image_member,parse_positive_classes
from cxr.data.dataset import preprocess_image
from cxr.inference.e03_h17_validation import load_models,read_probs,fuse,compare,save,csv_write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'));out=Path(cfg['output'])
    assert (out/'report.json').exists() and not (out/'running.lock').exists()
    if (out/'batch_diagnosis.json').exists():raise FileExistsError('Diagnosis already exists')
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.benchmark=False
    source=json.loads(Path(cfg['candidate_manifest']).read_text(encoding='utf-8'));models=load_models(source)
    rows=read_split_rows(cfg['archive'],'val'); keys=sorted({(r['Subject_id'],r['Study_id']) for r in rows})
    labels={(r['Subject_id'],r['Study_id']):[int(j in parse_positive_classes(r['Predict_class'])) for j in range(1,11)] for r in rows}
    y=np.array([labels[k] for k in keys]);parts={};results=[];start=time.monotonic()
    with zipfile.ZipFile(split_archive_path(cfg['archive'],'val')) as z:
        # Inputs are computed from every original Val image, not cached probabilities.
        tensors=[]
        for r in rows:
            with Image.open(io.BytesIO(z.read(image_member(r,'val')))) as im:tensors.append(preprocess_image(im,384,preprocessing='black_pad_v1'))
    for name,batch in [('B01',16),('B02',16),('B03',16),('M02',8)]:
        groups=defaultdict(list)
        for i in range(0,len(rows),batch):
            x=torch.stack(tensors[i:i+batch]).cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16): p=torch.sigmoid(models[name](x)).float().cpu().numpy().astype(np.float64)
            for r,v in zip(rows[i:i+batch],p):groups[(r['Subject_id'],r['Study_id'])].append(v)
        a=np.array([np.mean(groups[k],axis=0) for k in keys]);parts[name]=a
        ref=read_probs(cfg['references'][name],keys);results.append(compare(a,ref,y,keys,name))
        csv_write(out/f'historical_batch_{name}_val.csv',['Subject_id','Study_id']+[f'Prob_{j}' for j in range(1,11)],[[*k,*v] for k,v in zip(keys,a)])
        print(json.dumps(dict(model=name,batch=batch,**results[-1])),flush=True)
    parts['H17_BATCH1']=read_probs(out/'pass1_H17_BATCH1_val.csv',keys)
    refparts={n:read_probs(cfg['references'][n],keys) for n in parts}
    result=compare(fuse(parts),fuse(refparts),y,keys,'E03_H17_historical_batches')
    save(out/'batch_diagnosis.json',dict(protocol='E03 historic B01/B02/B03 batch16, M02 batch8; actual new H17 batch1; diagnostic only',
        components=results,fusion=result,elapsed_seconds=time.monotonic()-start,
        note='Does not change final batch1 implementation or its acceptance tolerance.'))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()

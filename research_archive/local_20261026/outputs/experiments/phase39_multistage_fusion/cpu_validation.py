"""Small CPU-only checks for Study aggregation and paired initial backbone."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from cxr.inference.aggregate import ImagePrediction, aggregate_mean_probabilities
from cxr.engine.metrics import study_multilabel_metrics
from cxr.engine.validate import StudyTarget
from cxr.phase39_model import build_model

pred = [ImagePrediction('P1','S1','I1',tuple([.2]*10)),
        ImagePrediction('P1','S1','I2',tuple([.8]*10)),
        ImagePrediction('P2','S2','I3',tuple([.1]*10))]
targets = [StudyTarget('P1','S1',tuple([1]*10)), StudyTarget('P2','S2',tuple([0]*10))]
studies = aggregate_mean_probabilities(pred)
assert len(studies)==2
assert all(abs(v-.5)<1e-6 for v in studies[0].probabilities)
assert study_multilabel_metrics(pred,targets)['macro_auc']==1.0

def backbone_digest(model):
    features = model.features if hasattr(model,'features') else model.backbone.features
    h=hashlib.sha256()
    for key,tensor in features.state_dict().items():
        h.update(key.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

torch.manual_seed(42)
base=build_model('P39_BASE_F1',pretrained=False)
torch.manual_seed(42)
fusion=build_model('P39_FUSION_F1',pretrained=False)
assert backbone_digest(base)==backbone_digest(fusion)
result={'state':'PASS','study_mean_two_images':True,'macro_auc_synthetic':1.0,
        'paired_backbone_initialization_sha256':backbone_digest(base),'cuda_used':False}
(Path(__file__).parent/'cpu_validation_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))

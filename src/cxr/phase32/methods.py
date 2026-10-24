"""Predeclared training-only interventions with complete-label/mask checks."""
import numpy as np
import torch
from torch.nn import functional as F
from cxr.data.archive import parse_positive_classes
from cxr.engine.losses import per_label_loss as baseline_loss

def study_positive_weights(rows):
    studies={}
    for row in rows:
        key=(row['Subject_id'],row['Study_id'])
        y=tuple(int(c in parse_positive_classes(row['Predict_class'])) for c in range(1,11))
        if key in studies and studies[key]!=y:raise ValueError('Conflicting Study labels')
        studies[key]=y
    y=np.asarray(list(studies.values()),dtype=float)
    if y.ndim!=2 or y.shape[1]!=10:raise ValueError('Ten classes required')
    positive=y.sum(0);negative=len(y)-positive
    if np.any(positive==0) or np.any(negative==0):raise ValueError('Training-only class support missing')
    return np.clip(np.sqrt(negative/positive),1.,5.).tolist()

def loss_elements(logits,targets,training):
    if training['loss']=='phase32_weighted_bce':
        w=torch.as_tensor(training['_resolved_pos_weight'],device=logits.device,dtype=torch.float32)
        if w.shape!=(10,) or not torch.isfinite(w).all():raise ValueError('Invalid class weights')
        return F.binary_cross_entropy_with_logits(logits.float(),targets.float(),pos_weight=w,reduction='none')
    return baseline_loss(logits,targets,training)

def mixup_batch(images,targets,mask,training):
    settings=training.get('mixup')
    if not settings:return images,targets,mask
    if settings!={'alpha':0.2,'probability':1.0,'scope':'within_microbatch','label_policy':'complete_binary_only'}:
        raise ValueError('Unreviewed Mixup settings')
    if not bool((mask==1).all()) or not bool(((targets==0)|(targets==1)).all()):
        raise ValueError('Mixup requires complete binary labels')
    lam=float(np.random.beta(.2,.2))
    order=torch.randperm(images.shape[0],device=images.device)
    return lam*images+(1-lam)*images[order],lam*targets+(1-lam)*targets[order],mask

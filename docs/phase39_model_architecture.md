# Phase39 paired Fold1 experiment

Entry: `python scripts/run_phase39.py --action run`. A repeated invocation resumes the
last completed epoch of an incomplete task and skips a verified `DONE.json` task.
The two tasks execute serially on the local GPU.

Both arms use the same torchvision EfficientNetV2-S ImageNet state, the frozen
patient Fold1, 384-pixel `black_pad_v1` preprocessing, conservative training
augmentation, BCEWithLogits, AdamW, cosine warmup, FP16 autocast, gradient
accumulation to effective batch 16, and EMA decay 0.999 with averaged buffers.
The local Windows worker count is 0 for both arms after a Windows pipe permission
error during preflight; the historical friend N17 used four workers and microbatch
8, so it is not a strict same-protocol control for this paired experiment.

BASE retains the standard final global average pooled 1280-dimensional feature
and 10-class linear classifier. FUSION extracts the installed torchvision
`features[5]` output, observed as 160 channels, applies global average pooling
and a 160-to-64 linear projection, concatenates it with the final 1280-vector,
and uses a 1344-to-10 linear classifier. The final-part classifier weights and
bias are copied from the identical BASE initialization. There is a single BCE
loss on final logits. The projection and fusion classifier remain in inference.

CPU smoke observed 20,190,298 BASE parameters versus 20,201,242 FUSION parameters
(+10,944). The projection, new classifier, and early backbone all received
nonzero finite gradients; CPU checkpoint reload produced exactly matching logits.
Short real-data CUDA smoke showed finite loss and restored optimizer, scheduler,
scaler, EMA, and RNG state. This is not a statement about final model quality.

Selection uses Fold1 OOF Study Mean Macro-AUC for EMA at each epoch. This is a
historically reused fold, so all conclusions are exploratory. Official Val and
hidden Test are not used in training or structure selection. The output
`oof_predictions.csv` is Fold1 only, never a complete five-fold OOF file.

When both models complete, a 2,000-draw paired patient bootstrap and per-class
AUC comparison are written to this directory. The RTX4060 image-to-CSV latency
measurement is a local reference. Official T4 compliance remains unverified.

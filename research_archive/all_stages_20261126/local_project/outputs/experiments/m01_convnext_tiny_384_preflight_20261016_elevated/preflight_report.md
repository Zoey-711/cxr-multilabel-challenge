# M01 ConvNeXt-Tiny preparation

This is a GPU readiness test, not an experiment result. No optimizer step, full training epoch, validation inference, or test inference was run.

## Source and model

- Official torchvision weight: `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, `https://download.pytorch.org/models/convnext_tiny-983f1562.pth`.
- SHA-256: `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`.
- Original ImageNet weight file: 114,419,221 bytes. Ten-output model: 27,827,818 parameters; portable `state_dict` checkpoint: 111,377,491 bytes (111.4 MB decimal), under the 500 MB model-file limit. The checkpoint has a newly initialized ten-output head and was reloaded with exactly matching logits.
- Local software: Python 3.12.4, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128; NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB reported device memory.

## Real-data pipeline and GPU test

One eight-image train batch was read through the project DataLoader. Shapes were `[8,3,384,384]` and `[8,10]`; images and targets were finite. It retained separate Subject, Study and image IDs, including multiple images for the same Study. All three microbatches passed FP16 forward, masked BCE, GradScaler backward, and a ten-logit shape check.

| Microbatch | Median GPU forward+backward | Peak allocated | Peak reserved | Board used after |
|---:|---:|---:|---:|---:|
| 2 | 27.2 ms | 557 MiB | 622 MiB | 1,731 MiB |
| 4 | 53.1 ms | 901 MiB | 960 MiB | 2,069 MiB |
| 8 | 135.3 ms | 1,621 MiB | 1,742 MiB | 2,851 MiB |

These times reuse one decoded batch and exclude loading, validation, optimizer update, and checkpoint I/O. They are not end-to-end inference latency measurements and cannot establish compliance on the official T4.

## Proposed formal configuration

Choose microbatch 8 and accumulation 2, giving effective batch 16. Batch 8 passed with over 6 GiB of device memory beyond its peak PyTorch reservation; the result leaves room for the optimizer state and DataLoader variability. Keep backbone learning rate `1e-4` and classifier learning rate `3e-4` as initial, unvalidated values, with AdamW weight decay `1e-4`, cosine schedule, two warmup epochs, BCEWithLogitsLoss, seed 42, and FP16 AMP. The config uses the same archive and train/validation partitions, label encoding, deterministic validation, and Study-level mean-probability AUC protocol as B01.

There are 30,344 training images, implying 3,793 microbatches and about 1,897 optimizer steps per epoch. The measured 135 ms GPU compute per microbatch implies roughly 8.6 minutes of forward/backward work per epoch. Allowing for ZIP decoding, optimizer updates, validation and checkpoint I/O, plan approximately 10–15 minutes per epoch, or 5–7.5 hours for all 30 epochs. This is a planning estimate; the first actual epoch should replace it. Early stopping may shorten the run.

The independent formal run ID is `m01_convnext_tiny_384_20261016`. The launch command is in the README and requires separate authorization. The preflight output is kept in this separate directory.

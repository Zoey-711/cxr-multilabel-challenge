# M02 formal preflight

Passed. Frozen M01 runtime config and artifact hashes verified. Only M02 experiment name and train augmentation differ.
Augmentation is one combined strategy; any future effect cannot be attributed to a single operator.
8 train images checked, 6 changed; pixel range after normalization [-2.118, 2.640], mean -0.174, SD 1.467; no NaN/Inf.
All targets and subject/study/image identifiers remained aligned. Eight validation and two test images exactly matched M01 deterministic preprocessing; test has no constructed targets.
Model, seed, AdamW/BCE, learning rates, batch 8 × accumulation 2, mean Study aggregation, patience 6, and ImageNet weight hash match M01.
Reserved formal output: `outputs\experiments\m02_convnext_tiny_384_aug_20261016_elevated` (absent at check). No training has started in this preflight.

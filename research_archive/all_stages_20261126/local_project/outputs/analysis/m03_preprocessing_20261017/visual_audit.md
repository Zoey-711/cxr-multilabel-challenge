# M03 preprocessing visualization audit

M02 already preserves the whole source image with aspect-ratio fit and black padding. M03 keeps identical geometry and uses the rounded ImageNet mean RGB fill. It tests border intensity, not a crop repair.

Train/validation image records: 31766; preprocessing crop records: 0. This is a code/metadata geometry conclusion; it cannot rule out anatomy outside the acquired source image.
Approximate padding area fraction quantiles: {'0': 0.0, '0.25': 0.16753926701570676, '0.5': 0.16753926701570676, '0.75': 0.16753926701570676, '0.95': 0.17757009345794394, '1': 0.4658753709198813}

The M03 border has RGB color (124,116,104) around a grayscale chest radiograph; this is a visible distribution change and a possible artifact, not a guaranteed improvement.

| Split | Aspect | Original | M02 black pad | M03 mean pad |
|---|---|---|---|---|
| train | 0.534 | [original](visuals\train_tall_pic_031934_original.jpg) | [M02](visuals/train_tall_pic_031934_m02.png) | [M03](visuals/train_tall_pic_031934_m03.png) |
| train | 1.000 | [original](visuals\train_square_pic_000004_original.jpg) | [M02](visuals/train_square_pic_000004_m02.png) | [M03](visuals/train_square_pic_000004_m03.png) |
| train | 1.681 | [original](visuals\train_wide_pic_015479_original.jpg) | [M02](visuals/train_wide_pic_015479_m02.png) | [M03](visuals/train_wide_pic_015479_m03.png) |
| val | 0.626 | [original](visuals\val_tall_pic_033245_original.jpg) | [M02](visuals/val_tall_pic_033245_m02.png) | [M03](visuals/val_tall_pic_033245_m03.png) |
| val | 1.000 | [original](visuals\val_square_pic_033061_original.jpg) | [M02](visuals/val_square_pic_033061_m02.png) | [M03](visuals/val_square_pic_033061_m03.png) |
| val | 1.670 | [original](visuals\val_wide_pic_034045_original.jpg) | [M02](visuals/val_wide_pic_034045_m02.png) | [M03](visuals/val_wide_pic_034045_m03.png) |

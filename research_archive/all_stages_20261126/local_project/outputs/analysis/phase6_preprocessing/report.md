# M01 preprocessing audit

Actual sequence: JPEG decode → RGB conversion → `ImageOps.pad(384×384, BILINEAR, black)` → tensor scaled to [0,1] → ImageNet mean/std normalization.
`ImageOps.pad` preserves aspect ratio, fits the complete image inside the square, and adds black padding on the short dimension. It does not apply a crop or nonuniform stretch.
The montage and six original/processed pairs show full image content with padding; this preprocessing does not remove lung apex or costophrenic angle pixels.
M02 keeps this preprocessing unchanged. Full-image fit resize plus padding is already M01's behavior, so it is not a distinct M03 candidate.

| Split | Selection | Original size | Aspect ratio | Approx. square side padded |
|---|---|---:|---:|---:|
| train | tallest | 1080×2022 | 0.534 | 46.6% |
| train | near_square | 2022×2022 | 1.000 | 0.0% |
| train | widest | 2899×1725 | 1.681 | 40.5% |
| val | tallest | 1266×2022 | 0.626 | 37.4% |
| val | near_square | 2022×2022 | 1.000 | 0.0% |
| val | widest | 3014×1805 | 1.670 | 40.1% |

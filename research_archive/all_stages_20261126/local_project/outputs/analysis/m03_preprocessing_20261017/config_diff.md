# M02/M03 config differences

Only algorithmic change: `data.preprocessing` (M02 implicit `black_pad_v1` → M03 `imagenet_mean_pad_v1`). `experiment.name` is bookkeeping. All model, optimizer, sampling, augmentation, seed, validation, and resolution fields match.

```json
{
  "data.preprocessing": {
    "M02": null,
    "M03": "imagenet_mean_pad_v1"
  },
  "experiment.name": {
    "M02": "m02_convnext_tiny_384_aug",
    "M03": "m03_convnext_tiny_meanpad_384"
  }
}
```

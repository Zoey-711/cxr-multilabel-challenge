# Phase 13 independent-patient protocol

The existing patient-level Train/Holdout files were read without changes. All images and Studies follow their Subject_id split; no patient crosses splits.

| Split | Patients | Studies | Images |
|---|---:|---:|---:|
| internal_train | 9949 | 16667 | 27301 |
| internal_val | 1106 | 1846 | 3043 |

Every class has positive and negative Studies in both splits. The scarcest Holdout class is class 1: 46 positive Studies across 43 patients. Exact Study and positive-patient supports are in `class_support.csv`.

The frozen M02 checkpoint already trained on all official Train patients, including internal Holdout. It cannot evaluate generalization to that Holdout.

Future authorized run: initialize ConvNeXt-Tiny from the same verified ImageNet weights, train only the 27,301 internal-Train images with the frozen M02 augmentation, 384 preprocessing, image shuffle, BCE, AdamW rates, schedule, seed and Study Mean validation. Select the backbone checkpoint using only the official 859-Study validation set. Extract deterministic features from the newly trained backbone for internal Train, official validation and internal Holdout. Fit V03 only on internal Train Studies, selecting its checkpoint using the official validation set; compare its Study predictions against Mean probability from the same backbone. Evaluate internal Holdout exactly once, with ten-class AUC/AP and paired patient bootstrap. Neither method may use Holdout for early stopping, tuning or training.

Estimated backbone time: 20.7 min/epoch, 3.8 h if 11 epochs or 10.3 h at the 30-epoch maximum on this machine. Add feature extraction, head fitting and validation time. Only one backbone retrain is needed because baseline and V03 share it. This is a scaling estimate, not a measured run.

The official validation set has been used in prior experiment selection. The internal Holdout remains untouched and is the final independent comparison for this protocol. No internal retraining was started.

# Competition rules and unresolved points

## Task, data, and labels

- Ten-label chest X-ray classification. Model indices 0–9 map to original class IDs 1–10: Enlarged Cardiomediastinum, Pneumothorax, Consolidation, Pneumonia, Edema, Cardiomegaly, Atelectasis, Lung Opacity, Pleural Effusion, and No Finding.
- Train has 18,513 labeled studies, validation 859 labeled studies, and test 1,414 unlabeled studies. A subject can have multiple studies; a study can have multiple images. Evaluation and submission are study-level.
- The official folder illustration uses `images/subject_xxxxx/study_xxxxx/image_xxxxxx.jpg`. The supplied ZIP actually uses `cxr_first/{train|valid|test}/{Subject_id}/{Study_id}/{Dicom_id}.jpg`; code follows the supplied files.
- The organizer states that train and validation have **complete multilabel annotations**. A class is marked only when positively mentioned in the report and present in the image. For competition targets, listed IDs are 1 and unlisted IDs are 0. An unlisted target does not assert clinical absence.
- The supplied `Predict_class` field is a semicolon-delimited list of original class IDs. No explicit per-class uncertainty code or blank labeled row was found. A future blank value must be treated as an error rather than all-negative.
- `Support Devices` is described as 1/0/-1/None in the official field list, but is absent from all three supplied CSVs and is not one of the ten target classes.

## Scoring and constraints

- Primary metric: study-level Macro-AUC. Auxiliary metric: Macro-F1, used only when primary scores tie.
- The supplied official rules allow **at most two submissions per day**. This supersedes the earlier one-per-day note. Deadline remains 2026-11-13 23:59 under the information supplied earlier.
- Official inference host: NVIDIA T4 16 GB, 8 CPU cores, 32 GB RAM, driver at least 570.117, CUDA 12.8, cuDNN 9, Python 3.12, PyTorch 2.10.0, torchvision 0.25.0.
- Total compressed model files: at most 500 MB. Per-image end-to-end inference latency: at most 100 ms.
- Only organizer-supplied training data may be used for additional training; external labeled data is forbidden. The previously supplied allowance for pretrained weights requires recording their source and compliance with the competition rules. Test labels cannot be used.
- Preliminary round: submit `submission.csv` only. Final round: prediction CSV and model; exact packaging awaits official notice.

## Submission format

The required UTF-8 CSV columns are `Subject_id,Study_id,Predict_class,Probability`. `Predict_class` uses original integer IDs 1–10 and `Probability` must lie in [0,1]. The official example uses a default 0.5 threshold and permits multiple rows per Study. The actual CSV uses string IDs such as `subject_00000` and `study_00000`; preserve them exactly, regardless of numeric-looking example IDs.

The user clarified that multiple classes can be submitted for a Study and the CSV rows should follow the chosen prediction threshold. The project defaults to threshold selection at 0.5; the threshold can be changed. It is **not yet confirmed** whether all ten class probabilities may be submitted regardless of threshold, or how to represent a Study with no class above the threshold. The writer rejects that empty-Study case until the rule is known.

## Audit status

The Phase 2 audit inspected actual fields, paths, all 33,984 images, label consistency, patient overlap, study multiplicity, and class cooccurrence. Results are in `outputs/audits/audit_report.md`. The complete-annotation rule now supplies competition-level negatives; retain validity masks for any future explicit uncertain or missing codes.

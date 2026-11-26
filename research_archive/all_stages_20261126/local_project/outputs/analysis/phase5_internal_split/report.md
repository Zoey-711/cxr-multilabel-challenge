# Fixed patient-level internal split

Seed 42; held out 1106/11055 train patients using one fixed random permutation.
All images and Studies of each Subject_id remain in one split. Official validation patients are disjoint.
No training or cross-validation was run.

| Split | Patients | Studies | Images |
|---|---:|---:|---:|
| internal_train | 9949 | 16667 | 27301 |
| internal_val | 1106 | 1846 | 3043 |

The class-level distribution is in `class_distribution.csv`; index files refer to the original ZIP without copying images.

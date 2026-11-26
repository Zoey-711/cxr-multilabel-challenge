# Tie-aware AP audit for Phase 13

Recomputed from the frozen, Study-aligned probabilities of the same 859 validation Studies. Equal scores are evaluated as one threshold group. Earlier AUC and bootstrap results are unaffected; earlier AP values can shift slightly when scores tie.

Maximum absolute difference from the earlier AP implementation: 0.001249.

| Class | M02 AP | V00 AP | V01 AP | V02 AP | V03 AP |
|---|---:|---:|---:|---:|---:|
| 1 | 0.08682 | 0.11213 | 0.10760 | 0.06840 | 0.08574 |
| 2 | 0.31107 | 0.30710 | 0.32392 | 0.29962 | 0.31168 |
| 3 | 0.16539 | 0.17134 | 0.16785 | 0.18149 | 0.16689 |
| 4 | 0.23171 | 0.21733 | 0.21909 | 0.22045 | 0.23092 |
| 5 | 0.47064 | 0.47638 | 0.47695 | 0.44116 | 0.46947 |
| 6 | 0.46124 | 0.47322 | 0.47321 | 0.46221 | 0.45917 |
| 7 | 0.52592 | 0.51167 | 0.51621 | 0.51012 | 0.52668 |
| 8 | 0.41375 | 0.41234 | 0.41034 | 0.39525 | 0.41714 |
| 9 | 0.78866 | 0.78825 | 0.78855 | 0.77720 | 0.78803 |
| 10 | 0.80094 | 0.80993 | 0.81079 | 0.78741 | 0.80253 |

Macro-AP: M02 0.42561, v00 0.42797, v01 0.42945, v02 0.41433, V03 0.42582.

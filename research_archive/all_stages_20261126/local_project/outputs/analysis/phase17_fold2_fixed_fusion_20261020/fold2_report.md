# Phase 17 Fold 2 fixed fusion

All results below use the predeclared 0.25 D17 + 0.25 C17 + 0.50 S17 weights.

| Model | Best epoch | Stop epoch | Val Macro-AUC | Fold 2 OOF Macro-AUC | Training hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| D17-F2 | 5 | 13 | 0.78989 | 0.79475 | 1.38 |
| C17-F2 | 4 | 10 | 0.79614 | 0.79968 | 3.02 |
| S17-F2 | 10 | 16 | 0.78781 | 0.79476 | 4.87 |
| E17-F2 | — | — | 0.79890 | 0.80798 | 0.00 |

Fold 1 fixed fusion: Val 0.80689; Fold 1 OOF 0.80680.
Fold 1 and Fold 2 OOF have different patient sets. Their AUCs are not averaged as full OOF AUC.
Official Val has been used repeatedly for checkpoint selection and is not an independent test set.

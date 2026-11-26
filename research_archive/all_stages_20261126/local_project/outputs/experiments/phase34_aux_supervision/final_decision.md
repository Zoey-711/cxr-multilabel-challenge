# Phase34 paired Pilot

BASE Fold1 AUC 0.799541; AUX 0.798559; delta -0.000982; patient bootstrap 95% CI [-0.0049822845693490376, 0.00285553026327391].

Coverage: 3687 Studies / 2211 patients. Both fixed at 5 epochs; identical main initialization and image order. No official Val/Test accessed.

Auxiliary-only gradient reaches conv0; pure DenseNet export contains no auxiliary head, complete Fold1 probabilities are exactly equal before/after export. Local GPU timing is reference only; T4_UNVERIFIED.

Predeclared followup evidence threshold met: False. Further folds require user decision; not auto-started. Single Fold1 is historically reused and does not establish a new independent generalization claim.

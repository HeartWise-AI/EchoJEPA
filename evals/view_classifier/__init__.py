# evals/view_classifier/__init__.py

"""The view classifier comparison: a frozen EchoJEPA encoder with an attentive probe (arm A) against the current
MViT-v2-S view classifier fine-tuned with the new classes (arm B), on the same human labels and the same folds.

Modules: `classes` (class lists and the old-to-new mapping), `labels` (the label table and the pseudo-labels),
`folds` (stratified patient-level folds), `metrics`, `train` (the shared cross-validation loop), `probe` (arm A on
cached tokens), `mvit` (arm B), `bench` (throughput on synthetic input) and `report`.
"""

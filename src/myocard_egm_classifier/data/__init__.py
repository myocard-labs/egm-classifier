"""egm-classifier training-data layer.

Torch ``Dataset`` wrappers, the patient-aware split, and the per-trace augmentation
transform. Moved here from ``myocard-egm-data`` (now pure I/O) since egm-classifier is
the sole consumer (Refactor Step 8, code-placement audit).
"""

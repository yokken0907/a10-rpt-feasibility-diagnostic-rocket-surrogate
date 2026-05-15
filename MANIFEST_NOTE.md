# Manifest policy

This public-gate package uses a manifest-excluding-self policy.

`FILE_MANIFEST.csv` and `FILE_MANIFEST.json` are generated after the repository files are prepared and are excluded from their own hash list to avoid unavoidable self-reference hash drift.

Public-gate decision: `PASS-WITH-MINOR-PUBLICATION-FIXES-A10-RPT-PUBLIC-GATE-0`
Public version: `v0.1.1-public-gate`

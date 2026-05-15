# A10-RPT Feasibility-Diagnostic Rocket-Propulsion Surrogate

This repository accompanies the AI-assisted independent research manuscript:

**A10-RPT: Mission-Variable-Preserving Structured Priors for Rocket-Propulsion Surrogates under Thermal, Supply, and Combined Stress**

Author: Keiji Yoshimura, Independent Researcher  
Status: GitHub-ready paper companion archive v0.1.1-public-gate; Zenodo-safe checkfix package

## Scope

A10-RPT is a nondimensional reduced-surrogate control study. It is designed to test feasibility-first structured priors under thermal, supply-margin, growth-rate, noise/disturbance, and combined stress.

The objective is not to propose a new propulsion mechanism, engine geometry, propellant system, or hardware-ready controller. The repository presents a paper companion archive for a closed surrogate with chamber-pressure, pressure-oscillation, thermal-load, combustion-phase, supply-margin, and cumulative-impulse states.

## Central interpretation

A10-RPT is best interpreted as a **feasibility-diagnostic control theory**. It separates regimes into:

- controller-feasible,
- resource-frontier-feasible,
- near-frontier,
- architecture-limited or unresolved.

It does not predict real engine performance.

## Technical Visual Orientation

For technically interested first-time readers, this repository includes a browser-only technical visual orientation page:

`docs/technical_visual_orientation/index.html`

This page provides a structured overview of the A10-RPT reduced rocket-surrogate logic, including the mission-variable-preserving diagnostic posture, reduced nondimensional surrogate status, thermal / supply-margin / growth-rate stress channels, combined-stress interpretation, evidence hierarchy, repository reading order, and the claim boundary.

The page is intended only as an orientation aid. It does not run propulsion simulations, does not validate a rocket engine, does not provide engine design or construction instructions, does not certify flight hardware, and does not replace the manuscript, source materials, figures, or independent expert review.

## What this repository contains

- manuscript PDF,
- Japanese and English README files,
- claim boundary and limitations,
- AI-assistance disclosure,
- practical positioning sheet,
- selected scripts and figures from the uploaded A10-RPT project archive,
- compact result-summary CSVs reconstructed from the paper,
- inventory of large raw CSVs excluded from the GitHub body.

## What this repository does not claim

This repository does **not** claim:

- engine design,
- propulsion-performance prediction,
- flight hardware readiness,
- hazardous experiment guidance,
- propellant handling instructions,
- real engine operation,
- formal barrier-certificate guarantees,
- validated aerospace safety.


## Zenodo-safe citation metadata

The active root `CITATION.cff` file is intentionally omitted in this checkfix package to avoid pre-DOI metadata-validation conflicts during Zenodo archival. Draft citation metadata is preserved at `docs/citation_metadata/CITATION_DRAFT_pre_doi.cff`. After DOI assignment, DOI metadata should be added in a follow-up DOI-metadata release.

## Repository status

This is a paper companion archive. It is not a full aerospace engineering model, not a certified controller, and not a one-command reproduction package for every large raw sweep file.

## PUBLIC-GATE-0 status

Decision: `PASS-WITH-MINOR-PUBLICATION-FIXES-A10-RPT-PUBLIC-GATE-0`  
Public version: `v0.1.1-public-gate`  
Classification: ロケット推進サロゲート・feasibility診断制御理論

This repository is a public-gate copy reviewed under an A10 Evidence-Lock Protocol style gate. The gate fixes the claim boundary, non-claims, manifest policy, and GitHub/Zenodo/Jxiv publication posture.


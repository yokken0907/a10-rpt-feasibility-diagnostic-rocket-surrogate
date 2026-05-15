#!/usr/bin/env python3
"""
A10-RPT Phase 1 v0.9 resource-frontier sweep.

Research-only nondimensional surrogate script. This is not an engine design tool,
not a propulsion-performance predictor, and not a hardware control recipe.

Purpose:
  v0.8 showed that some stress regimes become feasible only after architecture-
  level resource changes. v0.9 estimates the resource frontier: the smallest
  cooling/supply/impulse-margin multipliers, within this nondimensional
  surrogate, that open a mission-feasible region for each stress profile.

Dependencies:
  Place this file in the same directory as:
    a10_rpt_phase1_v04.py
    a10_rpt_phase1_v06_adaptive.py
    a10_rpt_phase1_v07_supervisor.py
    a10_rpt_phase1_v08_feasibility_sweep.py
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    v08 = importlib.import_module("a10_rpt_phase1_v08_feasibility_sweep")
except Exception as exc:
    print("ERROR: Could not import a10_rpt_phase1_v08_feasibility_sweep.py")
    print("Place this v0.9 script in the same folder as v0.8 and earlier scripts.")
    print(f"Original import error: {exc}")
    sys.exit(1)

v04 = v08.v04


def parse_float_grid(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def architecture_from_scales(cooling: float, supply: float, impulse: float) -> Dict[str, float]:
    """Build a smooth nondimensional architecture variant.

    cooling:
      >1 increases cooling effectiveness and modestly raises cooling slew/nominal reserve.
    supply:
      >1 reduces supply-margin depletion and increases recovery.
    impulse:
      >1 increases effective impulse integration margin.

    These are surrogate-resource multipliers, not direct hardware prescriptions.
    """
    arch: Dict[str, float] = {}

    if abs(cooling - 1.0) > 1e-12:
        arch["gamma_C"] = cooling
        arch["rC"] = 1.0 + 0.30 * (cooling - 1.0)
        arch["uC0"] = 1.0 + 0.16 * (cooling - 1.0)

    if abs(supply - 1.0) > 1e-12:
        # Calibrated to be close to v0.8 x125/x150 definitions but smooth.
        arch["lambda_P"] = max(0.45, 1.0 - 0.50 * (supply - 1.0))
        arch["lambda_dP"] = max(0.25, 1.0 - 0.80 * (supply - 1.0))
        arch["lambda_phi"] = max(0.45, 1.0 - 0.50 * (supply - 1.0))
        arch["rho_m"] = supply

    if abs(impulse - 1.0) > 1e-12:
        arch["chi_I"] = impulse

    return arch


def resource_score(cooling: float, supply: float, impulse: float) -> float:
    # Keep score simple and interpretable: total fractional resource increase.
    # Impulse margin is weighted slightly because it may represent mission or
    # propulsion efficiency margin rather than a single hardware subsystem.
    return (cooling - 1.0) + (supply - 1.0) + 0.75 * (impulse - 1.0)


def run(
    outdir: str,
    profiles: List[str],
    controllers: List[str],
    cooling_grid: List[float],
    supply_grid: List[float],
    impulse_grid: List[float],
    n_test: int,
    n_steps: int,
    seed_start: int,
    target_failure_rate: float,
):
    os.makedirs(outdir, exist_ok=True)

    for p in profiles:
        if p not in v08.STRESS_PROFILES:
            raise ValueError(f"Unknown profile {p}. Available: {sorted(v08.STRESS_PROFILES)}")

    controller_map = v08.collect_controllers(controllers)
    base = v04.Params()
    seeds = list(range(seed_start, seed_start + n_test))

    rows_summary = []
    rows_metrics = []

    total = len(profiles) * len(cooling_grid) * len(supply_grid) * len(impulse_grid) * len(controllers)
    counter = 0

    for profile_name in profiles:
        profile = v08.STRESS_PROFILES[profile_name]
        print(f"\n[profile] {profile_name}")

        for cooling in cooling_grid:
            for supply in supply_grid:
                for impulse in impulse_grid:
                    arch_name = f"C{cooling:.2f}_S{supply:.2f}_I{impulse:.2f}"
                    arch = architecture_from_scales(cooling, supply, impulse)
                    score = resource_score(cooling, supply, impulse)

                    for controller_name, controller in controller_map.items():
                        counter += 1
                        print(f"  [{counter}/{total}] {arch_name} | {controller_name}")
                        df, summary = v08.evaluate(
                            controller_name=controller_name,
                            controller=controller,
                            profile_name=profile_name,
                            profile=profile,
                            arch_name=arch_name,
                            arch=arch,
                            seeds=seeds,
                            base=base,
                            n_steps=n_steps,
                        )

                        df["cooling_scale"] = cooling
                        df["supply_scale"] = supply
                        df["impulse_scale"] = impulse
                        df["resource_score"] = score
                        rows_metrics.append(df)

                        summary.update({
                            "cooling_scale": cooling,
                            "supply_scale": supply,
                            "impulse_scale": impulse,
                            "resource_score": score,
                        })
                        rows_summary.append(summary)

    metrics = pd.concat(rows_metrics, ignore_index=True)
    summary = pd.DataFrame(rows_summary)

    # Sort for inspection.
    sort_cols = ["profile", "failure_rate", "resource_score", "mean_unsafe_duration", "cvar90_V"]
    summary_sorted = summary.sort_values(sort_cols)

    summary_path = os.path.join(outdir, "frontier_summary_all.csv")
    metrics_path = os.path.join(outdir, "frontier_metrics_by_seed.csv")
    summary_sorted.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    # Best overall per profile by mission feasibility first, then resource score.
    best_by_profile = summary_sorted.groupby("profile", as_index=False).first()
    best_by_profile.to_csv(os.path.join(outdir, "frontier_best_by_profile.csv"), index=False)

    # Feasible candidates under the chosen target failure-rate threshold.
    feasible = summary_sorted[summary_sorted["failure_rate"] <= target_failure_rate].copy()
    feasible = feasible.sort_values(["profile", "resource_score", "failure_rate", "mean_unsafe_duration", "cvar90_V"])
    feasible.to_csv(os.path.join(outdir, "frontier_feasible_candidates.csv"), index=False)

    min_feasible = None
    if len(feasible) > 0:
        min_feasible = feasible.groupby("profile", as_index=False).first()
        min_feasible.to_csv(os.path.join(outdir, "frontier_min_feasible_by_profile.csv"), index=False)

    # Heatmaps: for each profile, best failure_rate over controllers and impulse grid
    # at each cooling/supply coordinate.
    for profile_name in profiles:
        sub = summary[summary["profile"] == profile_name]
        best_grid = (
            sub.sort_values(["cooling_scale", "supply_scale", "failure_rate", "mean_unsafe_duration", "cvar90_V"])
            .groupby(["cooling_scale", "supply_scale"], as_index=False)
            .first()
        )
        pivot = best_grid.pivot(index="supply_scale", columns="cooling_scale", values="failure_rate")
        pivot = pivot.sort_index(ascending=True)

        fig, ax = plt.subplots(figsize=(8.5, 6))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{x:.2f}" for x in pivot.index])
        ax.set_title(f"Best failure_rate frontier: {profile_name}")
        ax.set_xlabel("cooling scale")
        ax.set_ylabel("supply scale")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="failure_rate")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"frontier_failure_rate_{profile_name}.png"), dpi=160)
        plt.close()

    cols = [
        "profile", "controller", "failure_rate", "mean_unsafe_duration", "cvar90_V",
        "resource_score", "cooling_scale", "supply_scale", "impulse_scale",
        "mean_final_I", "fail_h_rate", "fail_m_rate", "fail_I_rate",
        "worst_max_h", "worst_min_m", "min_final_I",
    ]

    print("\n=== Best overall by profile ===")
    print(best_by_profile[cols].to_string(index=False))

    print(f"\n=== Minimum feasible by profile, target failure_rate <= {target_failure_rate:.3f} ===")
    if min_feasible is None or len(min_feasible) == 0:
        print("No feasible candidates under the selected target.")
    else:
        print(min_feasible[cols].to_string(index=False))

    print("\nSaved:")
    for fn in [
        "frontier_summary_all.csv",
        "frontier_metrics_by_seed.csv",
        "frontier_best_by_profile.csv",
        "frontier_feasible_candidates.csv",
        "frontier_min_feasible_by_profile.csv",
    ]:
        path = os.path.join(outdir, fn)
        if os.path.exists(path):
            print(f"  {path}")
    print(f"  {os.path.join(outdir, 'frontier_failure_rate_<profile>.png')}")


def main():
    parser = argparse.ArgumentParser(description="A10-RPT Phase 1 v0.9 resource-frontier sweep")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-test", type=int, default=80)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--seed-start", type=int, default=7000)
    parser.add_argument("--outdir", type=str, default="results_v09_frontier")
    parser.add_argument("--profiles", type=str, default="combined_moderate,combined_harsh,heat_150")
    parser.add_argument("--controllers", type=str, default="a10_minimax_v07,a10_supply_priority_v06,a10_adaptive_v06")
    parser.add_argument("--cooling-grid", type=str, default="1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.75,2.00")
    parser.add_argument("--supply-grid", type=str, default="1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.75,2.00")
    parser.add_argument("--impulse-grid", type=str, default="1.00,1.03,1.05")
    parser.add_argument("--target-failure-rate", type=float, default=0.05)
    args = parser.parse_args()

    profiles = parse_list(args.profiles)
    controllers = parse_list(args.controllers)
    cooling_grid = parse_float_grid(args.cooling_grid)
    supply_grid = parse_float_grid(args.supply_grid)
    impulse_grid = parse_float_grid(args.impulse_grid)

    n_test, n_steps = args.n_test, args.n_steps
    if args.quick:
        n_test = 16
        n_steps = 300
        profiles = ["combined_moderate"]
        controllers = ["a10_minimax_v07", "a10_supply_priority_v06"]
        cooling_grid = [1.0, 1.25, 1.5]
        supply_grid = [1.0, 1.25, 1.5]
        impulse_grid = [1.0]

    run(
        outdir=args.outdir,
        profiles=profiles,
        controllers=controllers,
        cooling_grid=cooling_grid,
        supply_grid=supply_grid,
        impulse_grid=impulse_grid,
        n_test=n_test,
        n_steps=n_steps,
        seed_start=args.seed_start,
        target_failure_rate=args.target_failure_rate,
    )


if __name__ == "__main__":
    main()

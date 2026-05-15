#!/usr/bin/env python3
"""
A10-RPT Phase 1 v0.8 architecture-feasibility sweep.

Research-only nondimensional surrogate script. This is not an engine design tool,
not a propulsion-performance predictor, and not a hardware control recipe.

Purpose:
  v0.7 showed that additional controller arbitration can trade thermal failures
  into supply-margin and impulse failures. v0.8 therefore stops tuning a single
  controller and asks a more practical question:

    Which stress regimes are controllable under the current authority envelope,
    and which require extra cooling authority, supply-margin authority, or
    impulse/mission margin?

The script imports the frozen v0.4/v0.6/v0.7 controllers and sweeps simple
architecture/resource multipliers over stress profiles.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import replace
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    v04 = importlib.import_module("a10_rpt_phase1_v04")
except Exception as exc:
    print("ERROR: Could not import a10_rpt_phase1_v04.py")
    print("Place this v0.8 script in the same folder as the earlier scripts.")
    print(f"Original import error: {exc}")
    sys.exit(1)

try:
    v06 = importlib.import_module("a10_rpt_phase1_v06_adaptive")
except Exception as exc:
    print("ERROR: Could not import a10_rpt_phase1_v06_adaptive.py")
    print("Place this v0.8 script in the same folder as a10_rpt_phase1_v06_adaptive.py.")
    print(f"Original import error: {exc}")
    sys.exit(1)

try:
    v07 = importlib.import_module("a10_rpt_phase1_v07_supervisor")
except Exception as exc:
    print("ERROR: Could not import a10_rpt_phase1_v07_supervisor.py")
    print("Place this v0.8 script in the same folder as a10_rpt_phase1_v07_supervisor.py.")
    print(f"Original import error: {exc}")
    sys.exit(1)


# Reuse stress definitions from v0.7 so results are directly comparable.
STRESS_PROFILES = dict(v07.STRESS_PROFILES)

# Architecture/resource variants. These are nondimensional knobs, not hardware
# prescriptions. They ask: how much authority/resource would the surrogate need?
ARCHITECTURE_VARIANTS: Dict[str, Dict[str, float]] = {
    "baseline": {},

    # Cooling authority: improved heat extraction / thermal response reserve.
    "cooling_x125": {"gamma_C": 1.25, "rC": 1.10, "uC0": 1.05},
    "cooling_x150": {"gamma_C": 1.50, "rC": 1.15, "uC0": 1.08},

    # Supply authority: lower supply-margin consumption and better recovery.
    "supply_x125": {"lambda_P": 0.85, "lambda_dP": 0.75, "lambda_phi": 0.85, "rho_m": 1.25},
    "supply_x150": {"lambda_P": 0.75, "lambda_dP": 0.60, "lambda_phi": 0.75, "rho_m": 1.50},

    # Combined modest and strong architecture upgrades.
    "cooling_supply_x125": {
        "gamma_C": 1.25, "rC": 1.10, "uC0": 1.05,
        "lambda_P": 0.85, "lambda_dP": 0.75, "lambda_phi": 0.85, "rho_m": 1.25,
    },
    "cooling_supply_x150": {
        "gamma_C": 1.50, "rC": 1.15, "uC0": 1.08,
        "lambda_P": 0.75, "lambda_dP": 0.60, "lambda_phi": 0.75, "rho_m": 1.50,
    },

    # Effective impulse margin: represents a relaxed normalized impulse target
    # or better thrust-per-pressure margin in this surrogate. The success test
    # still requires final_I >= 1, so this simply gives more integral margin.
    "impulse_margin_x105": {"chi_I": 1.05},
    "cooling_supply_x125_impulse_x105": {
        "gamma_C": 1.25, "rC": 1.10, "uC0": 1.05,
        "lambda_P": 0.85, "lambda_dP": 0.75, "lambda_phi": 0.85, "rho_m": 1.25,
        "chi_I": 1.05,
    },
}


def apply_architecture(base: "v04.Params", arch: Dict[str, float]) -> "v04.Params":
    p = replace(base)
    for attr, mult in arch.items():
        if not hasattr(p, attr):
            raise ValueError(f"Unknown Params attribute in architecture variant: {attr}")
        setattr(p, attr, getattr(p, attr) * float(mult))
    # Maintain valid nominal cooling setting.
    p.uC0 = min(max(p.uC0, 0.0), 1.0)
    return p


def apply_profile_to_scenario(seed: int, base: "v04.Params", profile: Dict[str, float]) -> "v04.Scenario":
    # Same logic as v0.7, but repeated here for self-contained output naming.
    scenario = v04.sample_scenario(seed, base)
    p = replace(scenario.params)
    for attr in [
        "q0", "q_p", "q_a", "sigma0", "sigma_p", "sigma_phi",
        "lambda_P", "lambda_dP", "lambda_phi", "rho_m",
        "tau_p", "tau_h", "tau_phi", "gamma_C", "chi_I",
    ]:
        if attr in profile:
            setattr(p, attr, getattr(p, attr) * float(profile[attr]))
    for attr in ["uP0", "uC0", "rP", "rC", "rD", "rPhi"]:
        if attr in profile:
            setattr(p, attr, getattr(p, attr) * float(profile[attr]))
    disturbance_amp = float(profile.get("disturbance_amp", 1.0))
    disturbance_width = float(profile.get("disturbance_width", 1.0))
    disturbances = [(A * disturbance_amp, center, width * disturbance_width)
                    for A, center, width in scenario.disturbances]
    sensor_noise = scenario.sensor_noise * float(profile.get("sensor_noise", 1.0))
    return v04.Scenario(params=p, disturbances=disturbances, sensor_noise=sensor_noise, seed=seed)


def evaluate(controller_name: str, controller: Callable, profile_name: str, profile: Dict[str, float],
             arch_name: str, arch: Dict[str, float], seeds: List[int], base: "v04.Params", n_steps: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    arch_base = apply_architecture(base, arch)
    rows = []
    for seed in seeds:
        scenario = apply_profile_to_scenario(seed, arch_base, profile)
        out = v04.simulate(controller, scenario, n_steps=n_steps, alpha=None)
        rows.append({
            "profile": profile_name,
            "architecture": arch_name,
            "controller": controller_name,
            "seed": seed,
            "L": out["L"], "V": out["V"], "unsafe_duration": out["unsafe_duration"],
            "impulse_shortfall": out["impulse_shortfall"], "failure": int(out["failure"]),
            "max_p": out["max_p"], "max_a": out["max_a"], "max_h": out["max_h"],
            "max_abs_phi": out["max_abs_phi"], "min_m": out["min_m"], "final_I": out["final_I"],
            "C_U": out["C_U"], "C_R": out["C_R"],
            "fail_p": int(out["max_p"] > 1.0), "fail_a": int(out["max_a"] > 1.0),
            "fail_h": int(out["max_h"] > 1.0), "fail_phi": int(out["max_abs_phi"] > 1.0),
            "fail_m": int(out["min_m"] < scenario.params.m_min), "fail_I": int(out["final_I"] < 1.0),
        })
    df = pd.DataFrame(rows)
    summary = {
        "profile": profile_name, "architecture": arch_name, "controller": controller_name,
        "mean_L": float(df["L"].mean()), "cvar90_L": v04.cvar(df["L"].to_numpy(), 0.9), "worst_L": float(df["L"].max()),
        "mean_V": float(df["V"].mean()), "cvar90_V": v04.cvar(df["V"].to_numpy(), 0.9),
        "failure_rate": float(df["failure"].mean()), "mean_unsafe_duration": float(df["unsafe_duration"].mean()),
        "mean_final_I": float(df["final_I"].mean()), "min_final_I": float(df["final_I"].min()),
        "fail_p_rate": float(df["fail_p"].mean()), "fail_a_rate": float(df["fail_a"].mean()),
        "fail_h_rate": float(df["fail_h"].mean()), "fail_phi_rate": float(df["fail_phi"].mean()),
        "fail_m_rate": float(df["fail_m"].mean()), "fail_I_rate": float(df["fail_I"].mean()),
        "worst_max_a": float(df["max_a"].max()), "worst_max_h": float(df["max_h"].max()),
        "worst_max_p": float(df["max_p"].max()), "worst_min_m": float(df["min_m"].min()),
    }
    return df, summary


def collect_controllers(names: List[str]) -> Dict[str, Callable]:
    controllers: Dict[str, Callable] = {}
    sources = [getattr(v04, "CONTROLLERS", {}), getattr(v06, "LOCAL_CONTROLLERS", {}), getattr(v07, "LOCAL_CONTROLLERS", {})]
    for name in names:
        found = None
        for src in sources:
            if name in src:
                found = src[name]
        if found is None:
            available = sorted(set().union(*[set(src.keys()) for src in sources]))
            raise ValueError(f"Unknown controller {name}. Available: {available}")
        controllers[name] = found
    return controllers


def make_heatmap(best: pd.DataFrame, metric: str, profiles: List[str], outpath: str):
    pivot = best.pivot(index="architecture", columns="profile", values=metric)
    pivot = pivot.reindex(columns=profiles)
    fig, ax = plt.subplots(figsize=(max(9, 1.15 * len(profiles)), 5.5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_title(f"Best-controller {metric} by architecture/profile")
    ax.set_xlabel("profile"); ax.set_ylabel("architecture variant")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=metric)
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()


def run(outdir: str, controllers: List[str], profiles: List[str], architectures: List[str], n_test: int, n_steps: int, seed_start: int):
    base = v04.Params(); seeds = list(range(seed_start, seed_start + n_test))
    controller_map = collect_controllers(controllers)
    os.makedirs(outdir, exist_ok=True)

    metrics_all = []; summary_all = []
    for profile_name in profiles:
        profile = STRESS_PROFILES[profile_name]
        print(f"\n[profile] {profile_name}")
        for arch_name in architectures:
            arch = ARCHITECTURE_VARIANTS[arch_name]
            print(f"  [architecture] {arch_name}")
            for controller_name, controller in controller_map.items():
                print(f"    [run] {controller_name}")
                df, summary = evaluate(controller_name, controller, profile_name, profile, arch_name, arch, seeds, base, n_steps)
                metrics_all.append(df); summary_all.append(summary)

    metrics = pd.concat(metrics_all, ignore_index=True)
    summary = pd.DataFrame(summary_all)
    summary_sorted = summary.sort_values(["profile", "architecture", "failure_rate", "mean_unsafe_duration", "cvar90_V"])

    metrics.to_csv(os.path.join(outdir, "feasibility_metrics_by_seed.csv"), index=False)
    summary_sorted.to_csv(os.path.join(outdir, "feasibility_summary_by_profile_arch_controller.csv"), index=False)

    # Best controller per profile+architecture.
    best_arch = summary_sorted.groupby(["profile", "architecture"], as_index=False).first()
    best_arch.to_csv(os.path.join(outdir, "best_controller_by_profile_architecture.csv"), index=False)

    # Best architecture+controller per profile.
    best_profile = summary.sort_values(["profile", "failure_rate", "mean_unsafe_duration", "cvar90_V"]).groupby("profile", as_index=False).first()
    best_profile.to_csv(os.path.join(outdir, "best_architecture_controller_by_profile.csv"), index=False)

    print("\n=== Best architecture/controller by profile ===")
    cols = ["profile", "architecture", "controller", "failure_rate", "mean_unsafe_duration", "cvar90_V", "mean_final_I", "fail_h_rate", "fail_m_rate", "fail_I_rate", "worst_max_h", "worst_min_m", "min_final_I"]
    print(best_profile[cols].to_string(index=False))

    print("\n=== Best controller for each architecture/profile ===")
    print(best_arch[cols].to_string(index=False))

    make_heatmap(best_arch, "failure_rate", profiles, os.path.join(outdir, "best_failure_rate_heatmap.png"))
    make_heatmap(best_arch, "mean_unsafe_duration", profiles, os.path.join(outdir, "best_unsafe_duration_heatmap.png"))
    make_heatmap(best_arch, "cvar90_V", profiles, os.path.join(outdir, "best_cvar90V_heatmap.png"))

    print("\nSaved:")
    for fn in [
        "feasibility_metrics_by_seed.csv",
        "feasibility_summary_by_profile_arch_controller.csv",
        "best_controller_by_profile_architecture.csv",
        "best_architecture_controller_by_profile.csv",
        "best_failure_rate_heatmap.png",
        "best_unsafe_duration_heatmap.png",
        "best_cvar90V_heatmap.png",
    ]:
        print(f"  {os.path.join(outdir, fn)}")


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="A10-RPT Phase 1 v0.8 architecture-feasibility sweep")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-test", type=int, default=120)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--outdir", type=str, default="results_v08_feasibility")
    parser.add_argument("--controllers", type=str, default="a10_supply_priority_v06,a10_adaptive_v06,a10_minimax_v07,a10_thermal_v04,a10_barrier_v03")
    parser.add_argument("--profiles", type=str, default="combined_moderate,combined_harsh,heat_125,heat_150,supply_stress,noise_disturbance")
    parser.add_argument("--architectures", type=str, default="baseline,cooling_x125,cooling_x150,supply_x125,supply_x150,cooling_supply_x125,cooling_supply_x150,impulse_margin_x105,cooling_supply_x125_impulse_x105")
    args = parser.parse_args()

    controllers = parse_list(args.controllers)
    profiles = parse_list(args.profiles)
    architectures = parse_list(args.architectures)
    for p in profiles:
        if p not in STRESS_PROFILES:
            raise ValueError(f"Unknown profile {p}. Available: {sorted(STRESS_PROFILES)}")
    for a in architectures:
        if a not in ARCHITECTURE_VARIANTS:
            raise ValueError(f"Unknown architecture {a}. Available: {sorted(ARCHITECTURE_VARIANTS)}")

    n_test, n_steps = args.n_test, args.n_steps
    if args.quick:
        n_test = 24; n_steps = 300
        profiles = ["combined_moderate", "supply_stress", "heat_125"]
        architectures = ["baseline", "cooling_x125", "supply_x125", "cooling_supply_x125"]
        controllers = ["a10_supply_priority_v06", "a10_adaptive_v06", "a10_minimax_v07", "a10_thermal_v04"]

    run(args.outdir, controllers, profiles, architectures, n_test, n_steps, args.seed_start)


if __name__ == "__main__":
    main()

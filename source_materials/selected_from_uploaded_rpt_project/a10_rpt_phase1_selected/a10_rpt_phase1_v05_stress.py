#!/usr/bin/env python3
"""
A10-RPT Phase 1 v0.5 stress-validation harness.

This script is a validation wrapper around a10_rpt_phase1_v04.py.
It does not define a new controller. It freezes the v0.4 controller family and tests
whether the zero-failure nominal result survives larger ensembles and off-nominal
stress profiles.

IMPORTANT:
This is a nondimensional surrogate research script. It is not an engine design tool,
not a propulsion-performance predictor, and not a hardware control recipe.
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
except Exception as exc:  # pragma: no cover
    print("ERROR: Could not import a10_rpt_phase1_v04.py")
    print("Place this v0.5 script in the same folder as a10_rpt_phase1_v04.py and run again.")
    print(f"Original import error: {exc}")
    sys.exit(1)


# -----------------------------
# Stress profile definitions
# -----------------------------

STRESS_PROFILES: Dict[str, Dict[str, float]] = {
    # In-distribution retest; useful for checking larger N without changing the model.
    "nominal_retest": {},

    # Thermal loading grows while the controller architecture is frozen.
    "heat_125": {"q0": 1.10, "q_p": 1.25, "q_a": 1.25},
    "heat_150": {"q0": 1.15, "q_p": 1.50, "q_a": 1.50},

    # Combustion-instability growth is increased.
    "growth_125": {"sigma0": 1.25, "sigma_p": 1.25, "disturbance_amp": 1.15},
    "growth_150": {"sigma0": 1.50, "sigma_p": 1.35, "disturbance_amp": 1.25},

    # Supply system is made more fragile: power, dP/dt, and phase actions cost more,
    # and recovery is slower.
    "supply_stress": {
        "lambda_P": 1.35,
        "lambda_dP": 1.70,
        "lambda_phi": 1.35,
        "rho_m": 0.65,
    },

    # Measurement and disturbance stress. This tests whether success is merely
    # a low-noise artifact.
    "noise_disturbance": {"sensor_noise": 2.00, "disturbance_amp": 1.70},

    # A combined moderate stress profile, not an extreme adversarial one.
    "combined_moderate": {
        "q0": 1.10,
        "q_p": 1.25,
        "q_a": 1.25,
        "sigma0": 1.25,
        "sigma_p": 1.20,
        "lambda_P": 1.20,
        "lambda_dP": 1.35,
        "rho_m": 0.80,
        "sensor_noise": 1.50,
        "disturbance_amp": 1.35,
    },

    # A harsher combined profile. This is expected to expose the edge of the v0.4
    # safe region, not necessarily to be passed at zero failure.
    "combined_harsh": {
        "q0": 1.18,
        "q_p": 1.50,
        "q_a": 1.50,
        "sigma0": 1.40,
        "sigma_p": 1.35,
        "lambda_P": 1.35,
        "lambda_dP": 1.70,
        "rho_m": 0.70,
        "sensor_noise": 2.00,
        "disturbance_amp": 1.70,
    },
}

DEFAULT_CONTROLLERS = [
    "no_control",
    "power_only",
    "cooling_only",
    "pid_damping",
    "lqr_like",
    "a10_v01",
    "a10_barrier_v03",
    "a10_thermal_v04",
    "a10_thermal_refined_v04",
]

GRID_CONTROLLERS = [
    "power_only",
    "lqr_like",
    "a10_barrier_v03",
    "a10_thermal_v04",
]


def apply_profile_to_scenario(seed: int, base: "v04.Params", profile: Dict[str, float]) -> "v04.Scenario":
    """Sample a v0.4 scenario, then apply deterministic profile multipliers."""
    scenario = v04.sample_scenario(seed, base)
    p = replace(scenario.params)

    # Multiplicative parameter scales.
    for attr in [
        "q0", "q_p", "q_a", "sigma0", "sigma_p", "sigma_phi",
        "lambda_P", "lambda_dP", "lambda_phi", "rho_m",
        "tau_p", "tau_h", "tau_phi", "gamma_C",
    ]:
        if attr in profile:
            setattr(p, attr, getattr(p, attr) * float(profile[attr]))

    # Optional direct nominal-control scaling, useful for later experiments.
    for attr in ["uP0", "uC0"]:
        if attr in profile:
            setattr(p, attr, getattr(p, attr) * float(profile[attr]))

    # Disturbance and sensor scaling.
    disturbance_amp = float(profile.get("disturbance_amp", 1.0))
    disturbance_width = float(profile.get("disturbance_width", 1.0))
    disturbances = [
        (A * disturbance_amp, center, width * disturbance_width)
        for A, center, width in scenario.disturbances
    ]
    sensor_noise = scenario.sensor_noise * float(profile.get("sensor_noise", 1.0))

    return v04.Scenario(params=p, disturbances=disturbances, sensor_noise=sensor_noise, seed=seed)


def evaluate_controller_custom_scenarios(
    controller_name: str,
    controller: Callable,
    profile_name: str,
    profile: Dict[str, float],
    seeds: List[int],
    base: "v04.Params",
    n_steps: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for seed in seeds:
        scenario = apply_profile_to_scenario(seed, base, profile)
        out = v04.simulate(controller, scenario, n_steps=n_steps, alpha=None)
        rows.append({
            "profile": profile_name,
            "controller": controller_name,
            "seed": seed,
            "L": out["L"],
            "V": out["V"],
            "unsafe_duration": out["unsafe_duration"],
            "impulse_shortfall": out["impulse_shortfall"],
            "failure": int(out["failure"]),
            "max_p": out["max_p"],
            "max_a": out["max_a"],
            "max_h": out["max_h"],
            "max_abs_phi": out["max_abs_phi"],
            "min_m": out["min_m"],
            "final_I": out["final_I"],
            "C_U": out["C_U"],
            "C_R": out["C_R"],
            "fail_p": int(out["max_p"] > 1.0),
            "fail_a": int(out["max_a"] > 1.0),
            "fail_h": int(out["max_h"] > 1.0),
            "fail_phi": int(out["max_abs_phi"] > 1.0),
            "fail_m": int(out["min_m"] < base.m_min),
            "fail_I": int(out["final_I"] < 1.0),
        })
    df = pd.DataFrame(rows)
    summary = {
        "profile": profile_name,
        "controller": controller_name,
        "mean_L": float(df["L"].mean()),
        "cvar90_L": v04.cvar(df["L"].to_numpy(), 0.9),
        "worst_L": float(df["L"].max()),
        "mean_V": float(df["V"].mean()),
        "cvar90_V": v04.cvar(df["V"].to_numpy(), 0.9),
        "failure_rate": float(df["failure"].mean()),
        "mean_unsafe_duration": float(df["unsafe_duration"].mean()),
        "mean_final_I": float(df["final_I"].mean()),
        "mean_C_U": float(df["C_U"].mean()),
        "mean_C_R": float(df["C_R"].mean()),
        "fail_p_rate": float(df["fail_p"].mean()),
        "fail_a_rate": float(df["fail_a"].mean()),
        "fail_h_rate": float(df["fail_h"].mean()),
        "fail_phi_rate": float(df["fail_phi"].mean()),
        "fail_m_rate": float(df["fail_m"].mean()),
        "fail_I_rate": float(df["fail_I"].mean()),
        "worst_max_a": float(df["max_a"].max()),
        "worst_max_h": float(df["max_h"].max()),
        "worst_max_p": float(df["max_p"].max()),
        "worst_min_m": float(df["min_m"].min()),
        "min_final_I": float(df["final_I"].min()),
    }
    return df, summary


def run_stress_suite(
    outdir: str,
    controllers: List[str],
    profiles: List[str],
    n_test: int,
    n_steps: int,
    seed_start: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = v04.Params()
    seeds = list(range(seed_start, seed_start + n_test))
    metrics_all = []
    summary_all = []

    for profile_name in profiles:
        profile = STRESS_PROFILES[profile_name]
        print(f"\n[profile] {profile_name}")
        for controller_name in controllers:
            print(f"  [run] {controller_name}")
            controller = v04.CONTROLLERS[controller_name]
            df, summary = evaluate_controller_custom_scenarios(
                controller_name=controller_name,
                controller=controller,
                profile_name=profile_name,
                profile=profile,
                seeds=seeds,
                base=base,
                n_steps=n_steps,
            )
            metrics_all.append(df)
            summary_all.append(summary)

    metrics = pd.concat(metrics_all, ignore_index=True)
    summary = pd.DataFrame(summary_all).sort_values(
        ["profile", "failure_rate", "mean_unsafe_duration", "cvar90_V", "cvar90_L"]
    )

    metrics.to_csv(os.path.join(outdir, "stress_metrics_by_seed.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "stress_summary_by_profile_controller.csv"), index=False)
    return metrics, summary


def run_grid(
    outdir: str,
    controllers: List[str],
    n_test: int,
    n_steps: int,
    seed_start: int,
) -> pd.DataFrame:
    base = v04.Params()
    seeds = list(range(seed_start, seed_start + n_test))
    heat_scales = [1.00, 1.15, 1.30, 1.45, 1.60]
    growth_scales = [1.00, 1.15, 1.30, 1.45, 1.60]
    rows = []

    print("\n[grid] heat_scale x growth_scale")
    for controller_name in controllers:
        controller = v04.CONTROLLERS[controller_name]
        print(f"  [controller] {controller_name}")
        for heat in heat_scales:
            for growth in growth_scales:
                profile = {
                    "q0": 1.0 + 0.30 * (heat - 1.0),
                    "q_p": heat,
                    "q_a": heat,
                    "sigma0": growth,
                    "sigma_p": 1.0 + 0.70 * (growth - 1.0),
                    "disturbance_amp": 1.0 + 0.40 * (growth - 1.0),
                }
                df, summary = evaluate_controller_custom_scenarios(
                    controller_name=controller_name,
                    controller=controller,
                    profile_name="grid",
                    profile=profile,
                    seeds=seeds,
                    base=base,
                    n_steps=n_steps,
                )
                rows.append({
                    "controller": controller_name,
                    "heat_scale": heat,
                    "growth_scale": growth,
                    **{k: v for k, v in summary.items() if k not in ["profile", "controller"]},
                })
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(outdir, "grid_heat_growth_summary.csv"), index=False)
    return grid


def plot_stress_summary(outdir: str, summary: pd.DataFrame, controllers: List[str]) -> None:
    # Restrict to selected controllers and preserve requested ordering.
    summary = summary[summary["controller"].isin(controllers)].copy()

    for metric, ylabel, fname in [
        ("failure_rate", "failure rate", "stress_failure_rate_by_profile.png"),
        ("mean_unsafe_duration", "mean unsafe duration", "stress_unsafe_duration_by_profile.png"),
        ("cvar90_V", "CVaR90 safe-set violation V", "stress_cvar90V_by_profile.png"),
        ("fail_h_rate", "thermal failure rate", "stress_fail_h_rate_by_profile.png"),
        ("fail_m_rate", "supply-margin failure rate", "stress_fail_m_rate_by_profile.png"),
    ]:
        pivot = summary.pivot(index="profile", columns="controller", values=metric)
        pivot = pivot[[c for c in controllers if c in pivot.columns]]
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        ax.set_ylabel(ylabel)
        ax.set_title(f"A10-RPT Phase 1 v0.5 stress suite: {ylabel}")
        ax.legend(loc="best", fontsize=8)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=180)
        plt.close()


def plot_grid(outdir: str, grid: pd.DataFrame) -> None:
    for controller_name in grid["controller"].unique():
        g = grid[grid["controller"] == controller_name]
        pivot = g.pivot(index="growth_scale", columns="heat_scale", values="failure_rate")
        plt.figure(figsize=(7, 5))
        im = plt.imshow(pivot.values, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar(im, label="failure rate")
        plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
        plt.yticks(range(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])
        plt.xlabel("heat scale")
        plt.ylabel("growth scale")
        plt.title(f"A10-RPT v0.5 stress grid: {controller_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"grid_failure_{controller_name}.png"), dpi=180)
        plt.close()


def print_key_tables(summary: pd.DataFrame) -> None:
    cols = [
        "profile", "controller", "failure_rate", "mean_unsafe_duration", "cvar90_V",
        "mean_final_I", "fail_h_rate", "fail_m_rate", "fail_I_rate",
        "worst_max_h", "worst_min_m", "min_final_I",
    ]
    print("\n=== Stress summary sorted by profile and mission feasibility ===")
    print(summary[cols].to_string(index=False))

    # Focused comparison of v0.4 against v0.3 and LQR.
    focus = summary[summary["controller"].isin(["a10_thermal_v04", "a10_barrier_v03", "lqr_like", "power_only"])].copy()
    print("\n=== Focus comparison: a10_thermal_v04 vs main baselines ===")
    print(focus[cols].to_string(index=False))


def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="A10-RPT Phase 1 v0.5 stress-validation harness")
    parser.add_argument("--outdir", type=str, default="results_v05_stress")
    parser.add_argument("--quick", action="store_true", help="quick run for checking installation")
    parser.add_argument("--n-test", type=int, default=200, help="number of seeds per profile/controller")
    parser.add_argument("--n-steps", type=int, default=600, help="time steps per simulation")
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--controllers", type=str, default=",".join(DEFAULT_CONTROLLERS))
    parser.add_argument("--profiles", type=str, default=",".join(STRESS_PROFILES.keys()))
    parser.add_argument("--grid", action="store_true", help="also run heat/growth 2D stress grid")
    parser.add_argument("--grid-n-test", type=int, default=50)
    parser.add_argument("--grid-controllers", type=str, default=",".join(GRID_CONTROLLERS))
    args = parser.parse_args()

    if args.quick:
        args.n_test = min(args.n_test, 30)
        args.n_steps = min(args.n_steps, 350)
        args.grid_n_test = min(args.grid_n_test, 12)

    controllers = parse_csv_arg(args.controllers)
    profiles = parse_csv_arg(args.profiles)
    missing_controllers = [c for c in controllers if c not in v04.CONTROLLERS]
    missing_profiles = [p for p in profiles if p not in STRESS_PROFILES]
    if missing_controllers:
        raise ValueError(f"Unknown controllers: {missing_controllers}")
    if missing_profiles:
        raise ValueError(f"Unknown profiles: {missing_profiles}")

    os.makedirs(args.outdir, exist_ok=True)

    metrics, summary = run_stress_suite(
        outdir=args.outdir,
        controllers=controllers,
        profiles=profiles,
        n_test=args.n_test,
        n_steps=args.n_steps,
        seed_start=args.seed_start,
    )
    plot_stress_summary(args.outdir, summary, controllers)
    print_key_tables(summary)

    if args.grid:
        grid_controllers = parse_csv_arg(args.grid_controllers)
        missing_grid = [c for c in grid_controllers if c not in v04.CONTROLLERS]
        if missing_grid:
            raise ValueError(f"Unknown grid controllers: {missing_grid}")
        grid = run_grid(
            outdir=args.outdir,
            controllers=grid_controllers,
            n_test=args.grid_n_test,
            n_steps=args.n_steps,
            seed_start=args.seed_start + 20_000,
        )
        plot_grid(args.outdir, grid)

    print("\nSaved:")
    print(f"  {args.outdir}/stress_metrics_by_seed.csv")
    print(f"  {args.outdir}/stress_summary_by_profile_controller.csv")
    print(f"  {args.outdir}/stress_failure_rate_by_profile.png")
    print(f"  {args.outdir}/stress_unsafe_duration_by_profile.png")
    print(f"  {args.outdir}/stress_cvar90V_by_profile.png")
    print(f"  {args.outdir}/stress_fail_h_rate_by_profile.png")
    print(f"  {args.outdir}/stress_fail_m_rate_by_profile.png")
    if args.grid:
        print(f"  {args.outdir}/grid_heat_growth_summary.csv")
        print(f"  {args.outdir}/grid_failure_<controller>.png")


if __name__ == "__main__":
    main()

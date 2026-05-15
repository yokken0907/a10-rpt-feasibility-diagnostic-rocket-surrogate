#!/usr/bin/env python3
"""
A10-RPT Phase 1 v0.6 adaptive barrier stress-validation script.

This script imports a10_rpt_phase1_v04.py, freezes the nondimensional surrogate,
and adds v0.6 controller variants that arbitrate between thermal protection and
supply-margin preservation. It is a research-only nondimensional surrogate script,
not an engine design tool, not a propulsion-performance predictor, and not a
hardware control recipe.
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
    print("Place this v0.6 script in the same folder as a10_rpt_phase1_v04.py and run again.")
    print(f"Original import error: {exc}")
    sys.exit(1)


def pos(x: float) -> float:
    return float(max(x, 0.0))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


# -----------------------------------------------------------------------------
# Stress profiles: same definitions as v0.5 for comparability
# -----------------------------------------------------------------------------
STRESS_PROFILES: Dict[str, Dict[str, float]] = {
    "nominal_retest": {},
    "heat_125": {"q0": 1.10, "q_p": 1.25, "q_a": 1.25},
    "heat_150": {"q0": 1.15, "q_p": 1.50, "q_a": 1.50},
    "growth_125": {"sigma0": 1.25, "sigma_p": 1.25, "disturbance_amp": 1.15},
    "growth_150": {"sigma0": 1.50, "sigma_p": 1.35, "disturbance_amp": 1.25},
    "supply_stress": {"lambda_P": 1.35, "lambda_dP": 1.70, "lambda_phi": 1.35, "rho_m": 0.65},
    "noise_disturbance": {"sensor_noise": 2.00, "disturbance_amp": 1.70},
    "combined_moderate": {
        "q0": 1.10, "q_p": 1.25, "q_a": 1.25,
        "sigma0": 1.25, "sigma_p": 1.20,
        "lambda_P": 1.20, "lambda_dP": 1.35, "rho_m": 0.80,
        "sensor_noise": 1.50, "disturbance_amp": 1.35,
    },
    "combined_harsh": {
        "q0": 1.18, "q_p": 1.50, "q_a": 1.50,
        "sigma0": 1.40, "sigma_p": 1.35,
        "lambda_P": 1.35, "lambda_dP": 1.70, "rho_m": 0.70,
        "sensor_noise": 2.00, "disturbance_amp": 1.70,
    },
}


# -----------------------------------------------------------------------------
# v0.6 controllers
# -----------------------------------------------------------------------------

def controller_a10_adaptive_v06(x, tau, hist, dt, p, alpha=None):
    """
    v0.6 adaptive thermal/supply barrier controller.

    Motivation from v0.5 stress validation:
      - v0.4 thermal controller is excellent under nominal, growth, and heat stress,
        but can spend supply margin under supply/noise stress.
      - v0.3 supply barrier preserves supply margin better, but leaves thermal failures.

    v0.6 therefore uses a three-level architecture:
      1. Main danger gate for oscillation/pressure/heat/impulse.
      2. Thermal barrier, active when heat risk is high and supply margin is adequate.
      3. Supply-priority barrier, active when m approaches reserve; it caps power,
         suppresses phase/composition action, and relaxes impulse chasing.
    """
    pc, a, h, phi, m, I = x
    da = v04.estimate_da(hist, dt)
    sigma_hat = p.sigma0 + p.sigma_p * (pc - p.p_s) + p.sigma_phi * phi
    impulse_lag = pos(tau - I)
    impulse_surplus = pos(I - tau)

    # Smooth main gate, still excluding supply margin from the main gate.
    D_main = float(max(pc, a, h, abs(phi), impulse_lag / (tau + 0.05)))
    g = sigmoid(8.0 * (D_main - 0.755))

    # Risk gates.
    late = sigmoid(12.0 * (tau - 0.58))
    thermal_risk = sigmoid(18.0 * (h - 0.80))
    predictive_heat = sigmoid(10.0 * ((0.58 * h + 0.24 * pc + 0.18 * a * a) - 0.70))
    gT_raw = max(float(thermal_risk), float(late * predictive_heat))

    # Supply risk is deliberately earlier than v0.4. This prevents waiting until m is
    # already close to the hard threshold under supply-stress profiles.
    supply_risk = sigmoid(22.0 * (0.42 - m))
    deep_supply_risk = sigmoid(28.0 * (0.31 - m))

    # When supply risk is high, thermal intervention should be shifted toward cooling
    # and damping, not power-channel correction or phase/composition.
    gT = gT_raw * (1.0 - 0.65 * supply_risk)

    # Slightly more conservative nominal power than v0.4; cooling is free in this
    # surrogate's supply-margin equation, so keep cooling high.
    u = np.array([0.645, 0.0, 0.325, 0.0], dtype=float)

    # Power correction. Only chase impulse when lag is nontrivial and supply is safe.
    effective_lag = pos(impulse_lag - 0.015)
    duP = (
        -0.12 * pos(h - p.h_s)
        -0.08 * pos(pc - p.p_s)
        -0.06 * pos(a - p.a_s)
        +0.075 * effective_lag * (1.0 - 0.85 * supply_risk)
        -0.10 * gT * (0.35 + 0.60 * impulse_surplus)
        -0.10 * supply_risk
        -0.12 * deep_supply_risk
    )

    # Phase/composition is supply-expensive; suppress it aggressively under supply risk.
    duPhi_base = (
        -0.12 * a
        -0.003 * da
        -0.36 * phi
        +0.018 * pos(pc - p.p_s)
    )
    duPhi = duPhi_base * (1.0 - 0.92 * supply_risk)

    # Thermal authority is shifted to cooling and damping.
    duC = (
        +1.05 * pos(h - 0.62)
        +0.27 * a * a
        +0.14 * pos(pc - p.p_s)
        +0.28 * gT_raw
        +0.10 * late * pos(pc - 0.78)
        +0.04 * supply_risk
    )
    duD = (
        +1.00 * a
        +0.012 * pos(da)
        +0.27 * pos(sigma_hat) * a
        +0.08 * gT_raw * a
    )

    u = u + g * np.array([duP, duPhi, duC, duD], dtype=float)

    # Supply-priority power cap. This is stricter than v0.4 but still allows impulse
    # completion in the nominal/growth profiles according to the v0.5 margins.
    uP_cap = 0.50 + 0.34 * sigmoid(13.0 * (m - 0.26))
    uP_cap -= 0.06 * deep_supply_risk
    u[0] = min(u[0], uP_cap)

    # Late thermal cap, but only if there is genuine impulse surplus. Otherwise do not
    # sacrifice the mission variable.
    if tau > 0.62 and h > 0.84 and I > 0.965 * tau:
        u[0] = min(u[0], 0.63 - 0.24 * min(h - 0.84, 0.36))

    # If supply is critically low, prioritize m preservation over extra impulse.
    if m < 0.28:
        u[0] = min(u[0], 0.50)
    if m < 0.22:
        u[0] = min(u[0], 0.46)

    u[1] *= (1.0 - 0.92 * supply_risk)
    return u


def controller_a10_adaptive_refined_v06(x, tau, hist, dt, p, alpha=None):
    """v0.6 adaptive controller with narrow safe-region impulse recovery."""
    u = controller_a10_adaptive_v06(x, tau, hist, dt, p, alpha)
    pc, a, h, phi, m, I = x
    safety_margin = max(pc, a, h, abs(phi), p.m_min / max(m, 1e-6))
    if tau > 0.55 and safety_margin < 0.78 and h < 0.78 and m > 0.50 and I < tau - 0.01:
        u[0] += 0.020 * (tau - I)
    if tau > 0.65 and h > 0.76:
        u[2] += 0.10 * pos(h - 0.76)
        u[3] += 0.04 * a
    return u


def controller_a10_supply_priority_v06(x, tau, hist, dt, p, alpha=None):
    """
    More conservative v0.6 variant for supply/noise stress.

    This is not intended to be the default unless the adaptive controller still fails
    supply-stress validation. It provides a lower-power comparison point.
    """
    u = controller_a10_adaptive_v06(x, tau, hist, dt, p, alpha)
    pc, a, h, phi, m, I = x
    supply_risk = sigmoid(20.0 * (0.46 - m))
    u[0] -= 0.055 * supply_risk
    u[1] *= (1.0 - 0.95 * supply_risk)
    # More cooling/damping to preserve thermal safety after reducing power.
    u[2] += 0.08 * supply_risk + 0.08 * pos(h - 0.72)
    u[3] += 0.04 * supply_risk * a
    return u


LOCAL_CONTROLLERS: Dict[str, Callable] = dict(v04.CONTROLLERS)
LOCAL_CONTROLLERS.update({
    "a10_adaptive_v06": controller_a10_adaptive_v06,
    "a10_adaptive_refined_v06": controller_a10_adaptive_refined_v06,
    "a10_supply_priority_v06": controller_a10_supply_priority_v06,
})

DEFAULT_CONTROLLERS = [
    "no_control",
    "power_only",
    "lqr_like",
    "a10_barrier_v03",
    "a10_thermal_v04",
    "a10_thermal_refined_v04",
    "a10_adaptive_v06",
    "a10_adaptive_refined_v06",
    "a10_supply_priority_v06",
]


# -----------------------------------------------------------------------------
# Stress harness
# -----------------------------------------------------------------------------

def apply_profile_to_scenario(seed: int, base: "v04.Params", profile: Dict[str, float]) -> "v04.Scenario":
    scenario = v04.sample_scenario(seed, base)
    p = replace(scenario.params)

    for attr in [
        "q0", "q_p", "q_a", "sigma0", "sigma_p", "sigma_phi",
        "lambda_P", "lambda_dP", "lambda_phi", "rho_m",
        "tau_p", "tau_h", "tau_phi", "gamma_C",
    ]:
        if attr in profile:
            setattr(p, attr, getattr(p, attr) * float(profile[attr]))

    for attr in ["uP0", "uC0"]:
        if attr in profile:
            setattr(p, attr, getattr(p, attr) * float(profile[attr]))

    disturbance_amp = float(profile.get("disturbance_amp", 1.0))
    disturbance_width = float(profile.get("disturbance_width", 1.0))
    disturbances = [(A * disturbance_amp, center, width * disturbance_width)
                    for A, center, width in scenario.disturbances]
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


def run_stress_suite(outdir: str, controllers: List[str], profiles: List[str], n_test: int, n_steps: int, seed_start: int):
    base = v04.Params()
    seeds = list(range(seed_start, seed_start + n_test))
    metrics_all = []
    summary_all = []

    for profile_name in profiles:
        profile = STRESS_PROFILES[profile_name]
        print(f"\n[profile] {profile_name}")
        for controller_name in controllers:
            print(f"  [run] {controller_name}")
            controller = LOCAL_CONTROLLERS[controller_name]
            df, summary = evaluate_controller_custom_scenarios(controller_name, controller, profile_name, profile, seeds, base, n_steps)
            metrics_all.append(df)
            summary_all.append(summary)

    metrics = pd.concat(metrics_all, ignore_index=True)
    summary = pd.DataFrame(summary_all)

    os.makedirs(outdir, exist_ok=True)
    metrics.to_csv(os.path.join(outdir, "stress_metrics_by_seed.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "stress_summary_by_profile_controller.csv"), index=False)

    print("\n=== Stress summary sorted by profile and mission feasibility ===")
    sort_cols = ["profile", "failure_rate", "mean_unsafe_duration", "cvar90_V"]
    print(summary.sort_values(sort_cols).to_string(index=False))

    focus_names = [c for c in [
        "power_only", "lqr_like", "a10_barrier_v03", "a10_thermal_v04", "a10_adaptive_v06", "a10_adaptive_refined_v06", "a10_supply_priority_v06"
    ] if c in controllers]
    focus = summary[summary["controller"].isin(focus_names)].copy()
    print("\n=== Focus comparison: v0.6 vs main baselines ===")
    cols = [
        "profile", "controller", "failure_rate", "mean_unsafe_duration", "cvar90_V",
        "mean_final_I", "fail_h_rate", "fail_m_rate", "fail_I_rate",
        "worst_max_h", "worst_min_m", "min_final_I"
    ]
    print(focus.sort_values(["profile", "failure_rate", "mean_unsafe_duration", "cvar90_V"])[cols].to_string(index=False))

    make_plots(summary, outdir)
    print("\nSaved:")
    for fn in [
        "stress_metrics_by_seed.csv",
        "stress_summary_by_profile_controller.csv",
        "stress_failure_rate_by_profile.png",
        "stress_unsafe_duration_by_profile.png",
        "stress_cvar90V_by_profile.png",
        "stress_fail_h_rate_by_profile.png",
        "stress_fail_m_rate_by_profile.png",
    ]:
        print(f"  {os.path.join(outdir, fn)}")
    return metrics, summary


def make_grouped_bar(summary: pd.DataFrame, metric: str, ylabel: str, title: str, outpath: str):
    pivot = summary.pivot(index="profile", columns="controller", values=metric)
    pivot = pivot.sort_index()
    ax = pivot.plot(kind="bar", figsize=(16, 6), width=0.82)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlabel("profile")
    ax.legend(loc="best", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def make_plots(summary: pd.DataFrame, outdir: str):
    make_grouped_bar(summary, "failure_rate", "failure rate", "A10-RPT Phase 1 v0.6 stress suite: failure rate", os.path.join(outdir, "stress_failure_rate_by_profile.png"))
    make_grouped_bar(summary, "mean_unsafe_duration", "mean unsafe duration", "A10-RPT Phase 1 v0.6 stress suite: mean unsafe duration", os.path.join(outdir, "stress_unsafe_duration_by_profile.png"))
    make_grouped_bar(summary, "cvar90_V", "CVaR90 safe-set violation V", "A10-RPT Phase 1 v0.6 stress suite: CVaR90 safe-set violation V", os.path.join(outdir, "stress_cvar90V_by_profile.png"))
    make_grouped_bar(summary, "fail_h_rate", "thermal failure rate", "A10-RPT Phase 1 v0.6 stress suite: thermal failure rate", os.path.join(outdir, "stress_fail_h_rate_by_profile.png"))
    make_grouped_bar(summary, "fail_m_rate", "supply-margin failure rate", "A10-RPT Phase 1 v0.6 stress suite: supply-margin failure rate", os.path.join(outdir, "stress_fail_m_rate_by_profile.png"))


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="A10-RPT Phase 1 v0.6 adaptive barrier stress validation")
    parser.add_argument("--quick", action="store_true", help="Small fast run for smoke testing")
    parser.add_argument("--n-test", type=int, default=120)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--outdir", type=str, default="results_v06_stress")
    parser.add_argument("--controllers", type=str, default=",".join(DEFAULT_CONTROLLERS))
    parser.add_argument("--profiles", type=str, default=",".join(STRESS_PROFILES.keys()))
    args = parser.parse_args()

    controllers = parse_list(args.controllers)
    profiles = parse_list(args.profiles)
    for c in controllers:
        if c not in LOCAL_CONTROLLERS:
            raise ValueError(f"Unknown controller: {c}. Available: {sorted(LOCAL_CONTROLLERS)}")
    for p in profiles:
        if p not in STRESS_PROFILES:
            raise ValueError(f"Unknown profile: {p}. Available: {sorted(STRESS_PROFILES)}")

    n_test = args.n_test
    n_steps = args.n_steps
    if args.quick:
        n_test = 24
        n_steps = 300
        profiles = ["nominal_retest", "growth_125", "supply_stress", "combined_moderate"]
        controllers = ["power_only", "lqr_like", "a10_barrier_v03", "a10_thermal_v04", "a10_adaptive_v06", "a10_supply_priority_v06"]

    run_stress_suite(args.outdir, controllers, profiles, n_test, n_steps, args.seed_start)


if __name__ == "__main__":
    main()

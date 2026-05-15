#!/usr/bin/env python3
"""
A10-RPT Phase 1 v0.7 supervised barrier stress-validation script.

Research-only nondimensional surrogate script. This is not an engine design tool,
not a propulsion-performance predictor, and not a hardware control recipe.

v0.7 freezes the v0.4/v0.6 surrogate and adds a supervised barrier architecture:
- thermal and supply risks are not merely blended;
- a mission supervisor allocates authority among power, cooling, damping, and
  phase/composition according to remaining impulse margin and reserve state;
- the controller is intentionally more conservative on base power and uses
  cooling/damping first because they are not supply-margin-expensive in this
  surrogate.
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
    print("Place this v0.7 script in the same folder as a10_rpt_phase1_v04.py and run again.")
    print(f"Original import error: {exc}")
    sys.exit(1)

try:
    v06 = importlib.import_module("a10_rpt_phase1_v06_adaptive")
except Exception:
    v06 = None


def pos(x: float) -> float:
    return float(max(x, 0.0))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


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


def _risk_terms(x, tau, hist, dt, p):
    pc, a, h, phi, m, I = x
    da = v04.estimate_da(hist, dt)
    sigma_hat = p.sigma0 + p.sigma_p * (pc - p.p_s) + p.sigma_phi * phi
    impulse_lag = pos(tau - I)
    impulse_surplus = pos(I - tau)
    # Predictive heat proxy; tuned conservatively for heat/supply combined profiles.
    Hpred = 0.62 * h + 0.22 * pc + 0.11 * a * a + 0.05 * pos(sigma_hat) * a
    thermal_risk = max(sigmoid(18.0 * (h - 0.74)), sigmoid(14.0 * (Hpred - 0.68)))
    late = sigmoid(12.0 * (tau - 0.55))
    thermal_late = max(thermal_risk, late * sigmoid(15.0 * (h - 0.70)))
    supply_risk = sigmoid(24.0 * (0.48 - m))
    deep_supply = sigmoid(30.0 * (0.34 - m))
    # If impulse is comfortably ahead, permit stricter power reduction.
    surplus_gate = sigmoid(16.0 * (impulse_surplus - 0.025))
    # If impulse is late, recover only if both heat and supply are acceptable.
    recovery_permission = (1.0 - 0.92 * supply_risk) * (1.0 - 0.82 * thermal_late)
    return dict(pc=pc, a=a, h=h, phi=phi, m=m, I=I, da=da, sigma_hat=sigma_hat,
                impulse_lag=impulse_lag, impulse_surplus=impulse_surplus,
                thermal_risk=thermal_risk, thermal_late=thermal_late,
                supply_risk=supply_risk, deep_supply=deep_supply,
                surplus_gate=surplus_gate, recovery_permission=recovery_permission)


def controller_a10_supervisor_v07(x, tau, hist, dt, p, alpha=None):
    """v0.7 supervised thermal/supply barrier controller."""
    r = _risk_terms(x, tau, hist, dt, p)
    pc, a, h, phi, m, I = r["pc"], r["a"], r["h"], r["phi"], r["m"], r["I"]
    da, sigma_hat = r["da"], r["sigma_hat"]
    gT, gS, gSdeep = r["thermal_late"], r["supply_risk"], r["deep_supply"]
    lag, surplus = r["impulse_lag"], r["impulse_surplus"]
    rec = r["recovery_permission"]

    # Main gate excludes m; m is handled by the supervisory barrier.
    D_main = float(max(pc, a, h, abs(phi), lag / (tau + 0.05)))
    g = sigmoid(8.5 * (D_main - 0.73))

    # Conservative base. v0.6 showed excess final impulse even under stress.
    u = np.array([0.600, 0.0, 0.405, 0.0], dtype=float)

    # Power is the expensive channel. It is allowed to recover impulse only when
    # recovery_permission is high. Otherwise thermal/supply protection dominates.
    effective_lag = pos(lag - 0.018)
    duP = (
        +0.110 * effective_lag * rec
        -0.110 * pos(pc - 0.76)
        -0.150 * pos(h - 0.68)
        -0.080 * pos(a - 0.42)
        -0.130 * gT * (0.35 + 0.55 * r["surplus_gate"])
        -0.120 * gS
        -0.160 * gSdeep
    )

    # Phase/composition is supply-expensive. Keep it small; use it mostly to damp
    # oscillatory tendencies when supply risk is low.
    duPhi = (-0.08 * a - 0.0025 * da - 0.34 * phi + 0.012 * pos(pc - p.p_s))
    duPhi *= (1.0 - 0.96 * gS)
    duPhi *= (1.0 - 0.70 * gT)

    # Cooling and damping are preferred risk channels in this surrogate.
    duC = (
        +0.98 * pos(h - 0.54)
        +0.36 * a * a
        +0.14 * pos(pc - 0.72)
        +0.36 * gT
        +0.08 * gS
    )
    duD = (
        +0.94 * a
        +0.014 * pos(da)
        +0.30 * pos(sigma_hat) * a
        +0.10 * gT * a
        +0.04 * gS * a
    )

    u = u + g * np.array([duP, duPhi, duC, duD], dtype=float)

    # Dynamic power cap based on m, heat, and impulse surplus.
    cap = 0.68
    cap -= 0.18 * gS
    cap -= 0.13 * gSdeep
    cap -= 0.10 * gT * (0.45 + 0.55 * r["surplus_gate"])
    if tau > 0.55 and h > 0.72 and I > 0.965 * tau:
        cap -= 0.10 * min(h - 0.72, 0.35)
    if m < 0.34:
        cap = min(cap, 0.48)
    if m < 0.26:
        cap = min(cap, 0.43)
    # Avoid too-low power in early burn unless supply is genuinely endangered.
    floor = 0.44 if (tau < 0.35 and m > 0.42 and h < 0.88) else 0.35
    u[0] = min(max(u[0], floor), cap)

    # Additional late thermal cap with impulse-aware release.
    if tau > 0.66 and h > 0.82 and I > 0.98 * tau:
        u[0] = min(u[0], 0.56 - 0.22 * min(h - 0.82, 0.35))

    # Suppress phase further under deep supply or high thermal risk.
    u[1] *= (1.0 - 0.95 * gSdeep)
    return u


def controller_a10_supervisor_refined_v07(x, tau, hist, dt, p, alpha=None):
    u = controller_a10_supervisor_v07(x, tau, hist, dt, p, alpha)
    pc, a, h, phi, m, I = x
    # Only recover impulse in a very safe window.
    if tau > 0.62 and I < tau - 0.018 and h < 0.72 and m > 0.55 and pc < 0.82:
        u[0] += 0.018 * (tau - I)
    # Extra cooling buffer late in burn.
    if tau > 0.50:
        u[2] += 0.12 * pos(h - 0.66)
        u[3] += 0.035 * a
    return u


def controller_a10_minimax_v07(x, tau, hist, dt, p, alpha=None):
    """More conservative minimax variant for combined stress. May sacrifice margin/cost."""
    u = controller_a10_supervisor_v07(x, tau, hist, dt, p, alpha)
    pc, a, h, phi, m, I = x
    r = _risk_terms(x, tau, hist, dt, p)
    gT, gS = r["thermal_late"], r["supply_risk"]
    # Stronger power cap under either risk, with conditional release for impulse lag.
    cap = 0.62 - 0.15 * gT - 0.16 * gS
    if I < tau - 0.04 and h < 0.76 and m > 0.48:
        cap += 0.06
    u[0] = min(u[0], cap)
    u[2] += 0.10 + 0.15 * gT + 0.06 * gS
    u[3] += 0.06 * a
    u[1] *= (1.0 - 0.90 * max(gT, gS))
    return u


LOCAL_CONTROLLERS: Dict[str, Callable] = dict(v04.CONTROLLERS)
# Pull v0.6 controllers when available for direct comparison.
if v06 is not None:
    for name in ["a10_adaptive_v06", "a10_adaptive_refined_v06", "a10_supply_priority_v06"]:
        if hasattr(v06, "LOCAL_CONTROLLERS") and name in v06.LOCAL_CONTROLLERS:
            LOCAL_CONTROLLERS[name] = v06.LOCAL_CONTROLLERS[name]
LOCAL_CONTROLLERS.update({
    "a10_supervisor_v07": controller_a10_supervisor_v07,
    "a10_supervisor_refined_v07": controller_a10_supervisor_refined_v07,
    "a10_minimax_v07": controller_a10_minimax_v07,
})

DEFAULT_CONTROLLERS = [
    "power_only",
    "lqr_like",
    "a10_barrier_v03",
    "a10_thermal_v04",
    "a10_adaptive_v06",
    "a10_supply_priority_v06",
    "a10_supervisor_v07",
    "a10_supervisor_refined_v07",
    "a10_minimax_v07",
]


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


def evaluate_controller_custom_scenarios(controller_name: str, controller: Callable, profile_name: str, profile: Dict[str, float], seeds: List[int], base: "v04.Params", n_steps: int):
    rows = []
    for seed in seeds:
        scenario = apply_profile_to_scenario(seed, base, profile)
        out = v04.simulate(controller, scenario, n_steps=n_steps, alpha=None)
        rows.append({
            "profile": profile_name,
            "controller": controller_name,
            "seed": seed,
            "L": out["L"], "V": out["V"], "unsafe_duration": out["unsafe_duration"],
            "impulse_shortfall": out["impulse_shortfall"], "failure": int(out["failure"]),
            "max_p": out["max_p"], "max_a": out["max_a"], "max_h": out["max_h"],
            "max_abs_phi": out["max_abs_phi"], "min_m": out["min_m"], "final_I": out["final_I"],
            "C_U": out["C_U"], "C_R": out["C_R"],
            "fail_p": int(out["max_p"] > 1.0), "fail_a": int(out["max_a"] > 1.0),
            "fail_h": int(out["max_h"] > 1.0), "fail_phi": int(out["max_abs_phi"] > 1.0),
            "fail_m": int(out["min_m"] < base.m_min), "fail_I": int(out["final_I"] < 1.0),
        })
    df = pd.DataFrame(rows)
    summary = {
        "profile": profile_name, "controller": controller_name,
        "mean_L": float(df["L"].mean()), "cvar90_L": v04.cvar(df["L"].to_numpy(), 0.9), "worst_L": float(df["L"].max()),
        "mean_V": float(df["V"].mean()), "cvar90_V": v04.cvar(df["V"].to_numpy(), 0.9),
        "failure_rate": float(df["failure"].mean()), "mean_unsafe_duration": float(df["unsafe_duration"].mean()),
        "mean_final_I": float(df["final_I"].mean()), "mean_C_U": float(df["C_U"].mean()), "mean_C_R": float(df["C_R"].mean()),
        "fail_p_rate": float(df["fail_p"].mean()), "fail_a_rate": float(df["fail_a"].mean()), "fail_h_rate": float(df["fail_h"].mean()),
        "fail_phi_rate": float(df["fail_phi"].mean()), "fail_m_rate": float(df["fail_m"].mean()), "fail_I_rate": float(df["fail_I"].mean()),
        "worst_max_a": float(df["max_a"].max()), "worst_max_h": float(df["max_h"].max()), "worst_max_p": float(df["max_p"].max()),
        "worst_min_m": float(df["min_m"].min()), "min_final_I": float(df["final_I"].min()),
    }
    return df, summary


def make_grouped_bar(summary: pd.DataFrame, metric: str, ylabel: str, title: str, outpath: str):
    pivot = summary.pivot(index="profile", columns="controller", values=metric)
    pivot = pivot.sort_index()
    ax = pivot.plot(kind="bar", figsize=(16, 6), width=0.82)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_xlabel("profile")
    ax.legend(loc="best", fontsize=8)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()


def make_plots(summary: pd.DataFrame, outdir: str):
    make_grouped_bar(summary, "failure_rate", "failure rate", "A10-RPT Phase 1 v0.7 stress suite: failure rate", os.path.join(outdir, "stress_failure_rate_by_profile.png"))
    make_grouped_bar(summary, "mean_unsafe_duration", "mean unsafe duration", "A10-RPT Phase 1 v0.7 stress suite: mean unsafe duration", os.path.join(outdir, "stress_unsafe_duration_by_profile.png"))
    make_grouped_bar(summary, "cvar90_V", "CVaR90 safe-set violation V", "A10-RPT Phase 1 v0.7 stress suite: CVaR90 safe-set violation V", os.path.join(outdir, "stress_cvar90V_by_profile.png"))
    make_grouped_bar(summary, "fail_h_rate", "thermal failure rate", "A10-RPT Phase 1 v0.7 stress suite: thermal failure rate", os.path.join(outdir, "stress_fail_h_rate_by_profile.png"))
    make_grouped_bar(summary, "fail_m_rate", "supply-margin failure rate", "A10-RPT Phase 1 v0.7 stress suite: supply-margin failure rate", os.path.join(outdir, "stress_fail_m_rate_by_profile.png"))


def run_stress_suite(outdir: str, controllers: List[str], profiles: List[str], n_test: int, n_steps: int, seed_start: int):
    base = v04.Params(); seeds = list(range(seed_start, seed_start + n_test))
    metrics_all = []; summary_all = []
    for profile_name in profiles:
        profile = STRESS_PROFILES[profile_name]
        print(f"\n[profile] {profile_name}")
        for controller_name in controllers:
            print(f"  [run] {controller_name}")
            df, summary = evaluate_controller_custom_scenarios(controller_name, LOCAL_CONTROLLERS[controller_name], profile_name, profile, seeds, base, n_steps)
            metrics_all.append(df); summary_all.append(summary)
    metrics = pd.concat(metrics_all, ignore_index=True); summary = pd.DataFrame(summary_all)
    os.makedirs(outdir, exist_ok=True)
    metrics.to_csv(os.path.join(outdir, "stress_metrics_by_seed.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "stress_summary_by_profile_controller.csv"), index=False)
    print("\n=== Stress summary sorted by profile and mission feasibility ===")
    print(summary.sort_values(["profile", "failure_rate", "mean_unsafe_duration", "cvar90_V"]).to_string(index=False))
    focus_names = [c for c in [
        "power_only", "lqr_like", "a10_barrier_v03", "a10_thermal_v04", "a10_adaptive_v06", "a10_supply_priority_v06",
        "a10_supervisor_v07", "a10_supervisor_refined_v07", "a10_minimax_v07"
    ] if c in controllers]
    focus = summary[summary["controller"].isin(focus_names)].copy()
    print("\n=== Focus comparison: v0.7 vs main baselines ===")
    cols = ["profile", "controller", "failure_rate", "mean_unsafe_duration", "cvar90_V", "mean_final_I", "fail_h_rate", "fail_m_rate", "fail_I_rate", "worst_max_h", "worst_min_m", "min_final_I"]
    print(focus.sort_values(["profile", "failure_rate", "mean_unsafe_duration", "cvar90_V"])[cols].to_string(index=False))
    make_plots(summary, outdir)
    print("\nSaved:")
    for fn in ["stress_metrics_by_seed.csv", "stress_summary_by_profile_controller.csv", "stress_failure_rate_by_profile.png", "stress_unsafe_duration_by_profile.png", "stress_cvar90V_by_profile.png", "stress_fail_h_rate_by_profile.png", "stress_fail_m_rate_by_profile.png"]:
        print(f"  {os.path.join(outdir, fn)}")
    return metrics, summary


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="A10-RPT Phase 1 v0.7 supervised barrier stress validation")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-test", type=int, default=120)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--outdir", type=str, default="results_v07_stress")
    parser.add_argument("--controllers", type=str, default=",".join(DEFAULT_CONTROLLERS))
    parser.add_argument("--profiles", type=str, default=",".join(STRESS_PROFILES.keys()))
    args = parser.parse_args()
    controllers = parse_list(args.controllers); profiles = parse_list(args.profiles)
    for c in controllers:
        if c not in LOCAL_CONTROLLERS:
            raise ValueError(f"Unknown controller: {c}. Available: {sorted(LOCAL_CONTROLLERS)}")
    for p in profiles:
        if p not in STRESS_PROFILES:
            raise ValueError(f"Unknown profile: {p}. Available: {sorted(STRESS_PROFILES)}")
    n_test, n_steps = args.n_test, args.n_steps
    if args.quick:
        n_test = 24; n_steps = 300
        profiles = ["nominal_retest", "growth_125", "supply_stress", "combined_moderate"]
        controllers = ["power_only", "lqr_like", "a10_thermal_v04", "a10_adaptive_v06", "a10_supply_priority_v06", "a10_supervisor_v07", "a10_minimax_v07"]
    run_stress_suite(args.outdir, controllers, profiles, n_test, n_steps, args.seed_start)


if __name__ == "__main__":
    main()

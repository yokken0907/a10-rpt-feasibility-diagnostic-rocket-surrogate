#!/usr/bin/env python3
"""
A10-RPT Phase 1 v0.3
Mission-variable-preserving structured-prior control
for a nondimensional rocket-combustion-stability surrogate.

v0.3 changes:
- Keeps v0.1 and v0.2 A10 baselines.
- Adds a supply-barrier A10 controller that separates supply-margin preservation from the main danger gate.
- Smooths the A10 gate and reduces power-channel chattering, which was the dominant cause of supply-margin loss.
- Separates mission-feasibility diagnostics from aggregate loss.
- Reports failure reasons and violation-only CVaR.

IMPORTANT:
This is a toy nondimensional surrogate for theory testing.
It is not an engine design tool, not a propulsion-performance predictor,
and not a hardware control recipe.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utility functions
# -----------------------------

def pos(x: float | np.ndarray) -> float | np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def cvar(values: np.ndarray, beta: float = 0.9) -> float:
    """Mean of the worst (1-beta) fraction. Higher loss = worse."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    q = np.quantile(values, beta)
    tail = values[values >= q]
    return float(np.mean(tail)) if len(tail) else float(np.max(values))


# -----------------------------
# Model parameters
# -----------------------------

@dataclass
class Params:
    tau_p: float = 0.04
    tau_h: float = 0.18
    tau_phi: float = 0.06

    p0: float = 0.35
    k_P: float = 0.75
    k_phi: float = 0.10
    eta_a: float = 0.06

    sigma0: float = 1.10
    sigma_p: float = 1.60
    sigma_phi: float = 0.85
    sigma_D: float = 1.50
    sigma_C: float = 0.45
    kappa_a: float = 0.70

    q0: float = 0.25
    q_p: float = 0.95
    q_a: float = 0.90
    q_phi: float = 0.20
    gamma_C: float = 0.42

    b_phi: float = 0.65
    c_p: float = 0.20

    lambda_P: float = 0.12
    lambda_dP: float = 0.04
    lambda_phi: float = 0.03
    rho_m: float = 0.08

    chi_I: float = 1.45
    cphi_imp: float = 0.12
    ca_imp: float = 0.10

    p_s: float = 0.72
    a_s: float = 0.65
    h_s: float = 0.72
    m_min: float = 0.15
    m_guard: float = 0.32

    # Gate
    D0: float = 0.82
    rho_gate: float = 12.0

    # Nominal controls
    uP0: float = 0.72
    uPhi0: float = 0.0
    uC0: float = 0.25
    uD0: float = 0.0

    # Slew-rate limits per nondimensional time
    rP: float = 6.0
    rPhi: float = 10.0
    rC: float = 5.0
    rD: float = 12.0

    # Loss weights
    wV: float = 100.0
    wI: float = 80.0
    wU: float = 0.20
    wR: float = 0.05

    # Initial state: p, a, h, phi, m, I
    x0: Tuple[float, float, float, float, float, float] = (0.55, 0.12, 0.25, 0.0, 1.0, 0.0)


@dataclass
class Scenario:
    params: Params
    disturbances: List[Tuple[float, float, float]]  # (A, center, width)
    sensor_noise: float
    seed: int


def sample_scenario(seed: int, base: Params) -> Scenario:
    rng = np.random.default_rng(seed)
    p = replace(base)

    # Parametric uncertainty
    p.sigma0 *= 1.0 + rng.uniform(-0.25, 0.25)
    p.q_p *= 1.0 + rng.uniform(-0.20, 0.20)
    p.q_a *= 1.0 + rng.uniform(-0.20, 0.20)
    p.tau_p *= 1.0 + rng.uniform(-0.20, 0.20)
    p.tau_h *= 1.0 + rng.uniform(-0.20, 0.20)
    p.tau_phi *= 1.0 + rng.uniform(-0.20, 0.20)
    p.b_phi *= 1.0 + rng.uniform(-0.15, 0.15)

    n_pulses = int(rng.integers(1, 4))
    disturbances = []
    for _ in range(n_pulses):
        A = float(rng.uniform(0.05, 0.18))
        center = float(rng.uniform(0.15, 0.85))
        width = float(rng.uniform(0.01, 0.04))
        disturbances.append((A, center, width))

    sensor_noise = float(rng.uniform(0.01, 0.03))
    return Scenario(p, disturbances, sensor_noise, seed)


def disturbance_a(tau: float, scenario: Scenario) -> float:
    val = 0.0
    for A, center, width in scenario.disturbances:
        val += A * math.exp(-0.5 * ((tau - center) / width) ** 2)
    return val


# -----------------------------
# Dynamics, safety, metrics
# -----------------------------

def rhs(tau: float, x: np.ndarray, u: np.ndarray, du_dt: np.ndarray, scenario: Scenario) -> np.ndarray:
    """Continuous-time RHS for the nondimensional surrogate."""
    p = scenario.params
    pc, a, h, phi, m, I = x
    uP, uPhi, uC, uD = u
    duP_dt = du_dt[0]

    p_cmd = p.p0 + p.k_P * uP + p.k_phi * phi
    dpc = (p_cmd - pc) / p.tau_p + p.eta_a * a * a

    growth = (
        p.sigma0
        + p.sigma_p * (pc - p.p_s)
        + p.sigma_phi * phi
        - p.sigma_D * uD
        - p.sigma_C * uC
    )
    da = growth * a - p.kappa_a * a**3 + disturbance_a(tau, scenario)

    q = p.q0 + p.q_p * pc**2 + p.q_a * a**2 + p.q_phi * abs(phi)
    dh = (q - h) / p.tau_h - p.gamma_C * uC

    dphi = -phi / p.tau_phi + p.b_phi * uPhi + p.c_p * (pc - p.p_s)

    dm = (
        -p.lambda_P * uP**2
        -p.lambda_dP * duP_dt**2
        -p.lambda_phi * uPhi**2
        + p.rho_m * (1.0 - m)
    )

    impulse_rate = p.chi_I * max(pc - p.cphi_imp * phi**2 - p.ca_imp * a**2, 0.0)
    dI = impulse_rate

    return np.array([dpc, da, dh, dphi, dm, dI], dtype=float)


def apply_limits(u_raw: np.ndarray, u_prev: np.ndarray, dt: float, p: Params) -> Tuple[np.ndarray, np.ndarray]:
    """Apply saturation and slew-rate limits."""
    lower = np.array([0.0, -1.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0, 1.0])  # uD positive = active damping
    r = np.array([p.rP, p.rPhi, p.rC, p.rD])

    u_target = np.clip(u_raw, lower, upper)
    delta = np.clip(u_target - u_prev, -r * dt, r * dt)
    u = u_prev + delta
    du_dt = delta / dt
    return u, du_dt


def safe_violation_state(x: np.ndarray, p: Params) -> float:
    pc, a, h, phi, m, I = x
    return float(
        pos(pc - 1.0) ** 2
        + pos(a - 1.0) ** 2
        + pos(h - 1.0) ** 2
        + pos(abs(phi) - 1.0) ** 2
        + pos(p.m_min - m) ** 2
    )


def is_unsafe(x: np.ndarray, p: Params) -> bool:
    pc, a, h, phi, m, I = x
    return bool(pc > 1.0 or a > 1.0 or h > 1.0 or abs(phi) > 1.0 or m < p.m_min)


def danger_index(x: np.ndarray, tau: float, p: Params) -> float:
    pc, a, h, phi, m, I = x
    eps = 1e-6
    # I_req(tau)=tau; if I lags the schedule, danger rises.
    impulse_lag = pos(tau - I) / (tau + 0.05)
    return float(max(pc, a, h, abs(phi), p.m_min / (m + eps), impulse_lag))


def estimate_da(history: List[np.ndarray], dt: float) -> float:
    if len(history) < 2:
        return 0.0
    return float((history[-1][1] - history[-2][1]) / dt)


# -----------------------------
# Controllers
# -----------------------------

def nominal_u(p: Params) -> np.ndarray:
    return np.array([p.uP0, p.uPhi0, p.uC0, p.uD0], dtype=float)


def controller_no_control(x, tau, hist, dt, p, alpha=None):
    return nominal_u(p)


def controller_pid_damping(x, tau, hist, dt, p, alpha=None):
    da = estimate_da(hist, dt)
    u = nominal_u(p)
    u[3] = 0.75 * x[1] + 0.04 * pos(da)
    return u


def controller_power_only(x, tau, hist, dt, p, alpha=None):
    pc, a, h, phi, m, I = x
    u = nominal_u(p)
    u[0] += -0.45 * pos(pc - p.p_s) - 0.35 * pos(a - p.a_s) - 0.35 * pos(h - p.h_s) + 0.25 * pos(tau - I)
    return u


def controller_cooling_only(x, tau, hist, dt, p, alpha=None):
    pc, a, h, phi, m, I = x
    u = nominal_u(p)
    u[2] += 0.70 * pos(h - p.h_s) + 0.25 * a * a + 0.20 * pos(pc - p.p_s)
    return u


def controller_lqr_like(x, tau, hist, dt, p, alpha=None):
    # Simple hand-tuned linear feedback around a safe operating point.
    pc, a, h, phi, m, I = x
    da = estimate_da(hist, dt)
    u = nominal_u(p)
    u[0] += -0.25 * (pc - 0.75) - 0.20 * (a - 0.35) - 0.20 * (h - 0.55) + 0.20 * pos(tau - I)
    u[1] += -0.55 * a - 0.03 * da - 0.45 * phi
    u[2] += 0.35 * (h - 0.50) + 0.15 * a
    u[3] += 0.65 * a + 0.02 * pos(da)
    return u


DEFAULT_A10_ALPHA = np.array([
    0.45, 0.35, 0.35, 0.30,  # power correction
    0.50, 0.035, 0.60, 0.15, # phase/composition correction
    0.75, 0.30, 0.25,        # cooling correction
    0.85, 0.035, 0.35        # damping correction
], dtype=float)


def controller_a10(x, tau, hist, dt, p, alpha=None):
    """A10-RPT structured prior. Positive uD = active damping."""
    if alpha is None:
        alpha = DEFAULT_A10_ALPHA
    pc, a, h, phi, m, I = x
    da = estimate_da(hist, dt)
    sigma_hat = p.sigma0 + p.sigma_p * (pc - p.p_s) + p.sigma_phi * phi

    D = danger_index(x, tau, p)
    g = sigmoid(p.rho_gate * (D - p.D0))

    duP = (
        -alpha[0] * pos(a - p.a_s)
        -alpha[1] * pos(h - p.h_s)
        -alpha[2] * pos(pc - p.p_s)
        +alpha[3] * pos(tau - I)
    )
    duPhi = (
        -alpha[4] * a
        -alpha[5] * da
        -alpha[6] * phi
        +alpha[7] * pos(pc - p.p_s)
    )
    duC = (
        +alpha[8] * pos(h - p.h_s)
        +alpha[9] * a * a
        +alpha[10] * pos(pc - p.p_s)
    )
    duD = (
        +alpha[11] * a
        +alpha[12] * pos(da)
        +alpha[13] * pos(sigma_hat) * a
    )
    return nominal_u(p) + g * np.array([duP, duPhi, duC, duD], dtype=float)


def controller_a10_refined(x, tau, hist, dt, p, alpha=None):
    """v0.1 A10 + small in-sector refinement, kept as a historical baseline."""
    u = controller_a10(x, tau, hist, dt, p, alpha)
    pc, a, h, phi, m, I = x
    # Small scheduled correction to recover impulse after mid-burn if safe.
    safety_margin = max(pc, a, h, abs(phi), p.m_min / max(m, 1e-6))
    if tau > 0.35 and safety_margin < 0.85 and I < tau:
        u[0] += 0.08 * (tau - I)
    # Small extra damping in late-burn if oscillation persists.
    if tau > 0.55:
        u[3] += 0.10 * a
    return u


def controller_a10_guarded(x, tau, hist, dt, p, alpha=None):
    """
    v0.2 A10-RPT: supply-margin-aware structured prior.

    The v0.1 controller reduced unsafe duration but tended to spend the pump/supply
    margin too aggressively. This version keeps the same gate-and-prior logic, but
    adds an explicit guard that suppresses margin-expensive power/phase actions when
    m approaches the reserve threshold.
    """
    if alpha is None:
        alpha = DEFAULT_A10_ALPHA
    pc, a, h, phi, m, I = x
    da = estimate_da(hist, dt)
    sigma_hat = p.sigma0 + p.sigma_p * (pc - p.p_s) + p.sigma_phi * phi

    D = danger_index(x, tau, p)
    g = sigmoid(p.rho_gate * (D - p.D0))
    m_deficit = pos(p.m_guard - m)
    m_risk = sigmoid(18.0 * (p.m_guard - m))
    impulse_lag = pos(tau - I)

    duP = (
        -0.85 * alpha[0] * pos(a - p.a_s)
        -1.05 * alpha[1] * pos(h - p.h_s)
        -0.90 * alpha[2] * pos(pc - p.p_s)
        +0.65 * alpha[3] * impulse_lag * (1.0 - 0.85 * m_risk)
        -0.90 * m_deficit
    )

    phase_raw = (
        -0.75 * alpha[4] * a
        -0.50 * alpha[5] * da
        -0.90 * alpha[6] * phi
        +0.40 * alpha[7] * pos(pc - p.p_s)
    )
    duPhi = (1.0 - 0.80 * m_risk) * phase_raw

    duC = (
        +1.00 * alpha[8] * pos(h - p.h_s)
        +0.85 * alpha[9] * a * a
        +0.80 * alpha[10] * pos(pc - p.p_s)
        +0.12 * m_risk
    )
    duD = (
        +0.90 * alpha[11] * a
        +0.55 * alpha[12] * pos(da)
        +0.90 * alpha[13] * pos(sigma_hat) * a
        +0.08 * m_risk * a
    )

    u = nominal_u(p) + g * np.array([duP, duPhi, duC, duD], dtype=float)

    if m < p.m_guard:
        u[0] = min(u[0], p.uP0 - 0.55 * (p.m_guard - m))
    return u


def controller_a10_guarded_refined(x, tau, hist, dt, p, alpha=None):
    """v0.2 A10 with a narrow in-sector impulse-recovery correction."""
    u = controller_a10_guarded(x, tau, hist, dt, p, alpha)
    pc, a, h, phi, m, I = x
    safety_margin = max(pc, a, h, abs(phi), p.m_min / max(m, 1e-6))
    if tau > 0.40 and safety_margin < 0.82 and m > p.m_guard and I < tau:
        u[0] += 0.05 * (tau - I)
    if tau > 0.55 and a > 0.25:
        u[3] += 0.07 * a
    return u


def controller_a10_barrier_v03(x, tau, hist, dt, p, alpha=None):
    """
    v0.3 A10-RPT: supply-barrier structured prior.

    Diagnostic motivation:
    v0.1 reduced unsafe duration but spent the supply margin.
    v0.2 tried to include supply risk in the A10 gate, but the coupled gate
    could still create power-channel chattering and worsen m failures.

    v0.3 separates the two roles:
      1. A smoother danger gate acts on pressure/oscillation/thermal/impulse risk.
      2. A supply barrier caps the power channel and suppresses phase action when
         m approaches the reserve threshold.

    This is still a nondimensional toy controller, not a hardware control law.
    """
    pc, a, h, phi, m, I = x
    da = estimate_da(hist, dt)
    sigma_hat = p.sigma0 + p.sigma_p * (pc - p.p_s) + p.sigma_phi * phi
    impulse_lag = pos(tau - I)

    # Smooth main gate. Supply margin is deliberately excluded here and handled
    # by a separate barrier below. This reduces gate chattering from noisy m.
    D_main = float(max(pc, a, h, abs(phi), impulse_lag / (tau + 0.05)))
    g = sigmoid(8.0 * (D_main - 0.78))

    # Slightly lower nominal power and slightly higher nominal cooling.
    # This preserves impulse in the tested surrogate while creating supply reserve.
    u = np.array([0.68, 0.0, 0.28, 0.0], dtype=float)

    # Low-gain power correction: avoid aggressive throttle oscillations.
    duP = (
        -0.18 * pos(h - p.h_s)
        -0.12 * pos(pc - p.p_s)
        -0.08 * pos(a - p.a_s)
        +0.08 * impulse_lag
    )

    # Phase/composition is useful but supply-expensive in this surrogate, so keep
    # it weaker than v0.1/v0.2.
    duPhi = (
        -0.18 * a
        -0.006 * da
        -0.30 * phi
        +0.04 * pos(pc - p.p_s)
    )

    # Shift authority toward cooling and damping, which do not directly deplete
    # m in the present surrogate.
    duC = (
        +0.85 * pos(h - 0.66)
        +0.22 * a * a
        +0.12 * pos(pc - p.p_s)
    )
    duD = (
        +0.95 * a
        +0.015 * pos(da)
        +0.25 * pos(sigma_hat) * a
    )

    u = u + g * np.array([duP, duPhi, duC, duD], dtype=float)

    # Supply barrier. As m approaches the reserve region, cap power and suppress
    # phase/composition action. This is the key v0.3 mechanism.
    m_risk = sigmoid(24.0 * (0.34 - m))
    uP_cap = 0.52 + 0.36 * sigmoid(14.0 * (m - 0.23))
    u[0] = min(u[0], uP_cap)
    u[1] *= (1.0 - 0.85 * m_risk)
    return u


def controller_a10_barrier_refined_v03(x, tau, hist, dt, p, alpha=None):
    """v0.3 A10 barrier plus very narrow in-sector impulse/damping refinement."""
    u = controller_a10_barrier_v03(x, tau, hist, dt, p, alpha)
    pc, a, h, phi, m, I = x
    safety_margin = max(pc, a, h, abs(phi), p.m_min / max(m, 1e-6))

    # Recover impulse only when the trajectory is comfortably inside the safe sector
    # and supply margin is ample.
    if tau > 0.45 and safety_margin < 0.82 and m > 0.45 and I < tau:
        u[0] += 0.035 * (tau - I)
    if tau > 0.55 and a > 0.25:
        u[3] += 0.05 * a
    return u


CONTROLLERS: Dict[str, Callable] = {
    "no_control": controller_no_control,
    "pid_damping": controller_pid_damping,
    "power_only": controller_power_only,
    "cooling_only": controller_cooling_only,
    "lqr_like": controller_lqr_like,
    "a10_v01": controller_a10,
    "a10_v01_refined": controller_a10_refined,
    "a10_rpt_v02": controller_a10_guarded,
    "a10_refined_v02": controller_a10_guarded_refined,
    "a10_barrier_v03": controller_a10_barrier_v03,
    "a10_barrier_refined_v03": controller_a10_barrier_refined_v03,
}


# -----------------------------
# Simulation and evaluation
# -----------------------------

def simulate(
    controller: Callable,
    scenario: Scenario,
    n_steps: int = 600,
    alpha: np.ndarray | None = None,
) -> Dict[str, np.ndarray | float | bool]:
    p = scenario.params
    dt = 1.0 / n_steps
    x = np.array(p.x0, dtype=float)
    u_prev = nominal_u(p)
    hist: List[np.ndarray] = [x.copy()]

    T = np.zeros(n_steps + 1)
    X = np.zeros((n_steps + 1, 6))
    U = np.zeros((n_steps + 1, 4))
    X[0] = x
    U[0] = u_prev

    # Deterministic sensor noise stream per scenario/controller call.
    rng = np.random.default_rng(scenario.seed + 10_000)

    for k in range(n_steps):
        tau = k * dt
        # Controller sees a noisy measurement, not the exact state.
        y = x + rng.normal(0.0, scenario.sensor_noise, size=6)
        y[4] = max(y[4], -1.0)  # avoid extreme nonsensical measurement for m
        u_raw = controller(y, tau, hist, dt, p, alpha=alpha)
        u, du_dt = apply_limits(u_raw, u_prev, dt, p)

        # RK4 with constant u over the step.
        k1 = rhs(tau, x, u, du_dt, scenario)
        k2 = rhs(tau + 0.5 * dt, x + 0.5 * dt * k1, u, du_dt, scenario)
        k3 = rhs(tau + 0.5 * dt, x + 0.5 * dt * k2, u, du_dt, scenario)
        k4 = rhs(tau + dt, x + dt * k3, u, du_dt, scenario)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Physical non-negativity for amplitude-like and accumulated quantities.
        x[0] = max(x[0], 0.0)
        x[1] = max(x[1], 0.0)
        x[2] = max(x[2], 0.0)
        x[5] = max(x[5], 0.0)

        T[k + 1] = tau + dt
        X[k + 1] = x
        U[k + 1] = u
        u_prev = u
        hist.append(x.copy())

    violation_series = np.array([safe_violation_state(row, p) for row in X])
    unsafe_series = np.array([is_unsafe(row, p) for row in X], dtype=float)
    V = float(np.trapezoid(violation_series, T))
    unsafe_duration = float(np.trapezoid(unsafe_series, T))
    impulse_shortfall = float(pos(1.0 - X[-1, 5]) ** 2)
    C_U = float(np.trapezoid(np.sum(U * U, axis=1), T))
    dU = np.gradient(U, T, axis=0)
    C_R = float(np.trapezoid(np.sum(dU * dU, axis=1), T))
    L = p.wV * V + p.wI * impulse_shortfall + p.wU * C_U + p.wR * C_R

    failure = bool((V > 1e-8) or (X[-1, 5] < 1.0))

    return {
        "T": T,
        "X": X,
        "U": U,
        "V": V,
        "unsafe_duration": unsafe_duration,
        "impulse_shortfall": impulse_shortfall,
        "C_U": C_U,
        "C_R": C_R,
        "L": L,
        "failure": failure,
        "max_p": float(np.max(X[:, 0])),
        "max_a": float(np.max(X[:, 1])),
        "max_h": float(np.max(X[:, 2])),
        "max_abs_phi": float(np.max(np.abs(X[:, 3]))),
        "min_m": float(np.min(X[:, 4])),
        "final_I": float(X[-1, 5]),
    }


def evaluate_controller(
    name: str,
    controller: Callable,
    seeds: List[int],
    base: Params,
    n_steps: int,
    alpha: np.ndarray | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for seed in seeds:
        scenario = sample_scenario(seed, base)
        out = simulate(controller, scenario, n_steps=n_steps, alpha=alpha)
        rows.append({
            "controller": name,
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
        "controller": name,
        "mean_L": float(df["L"].mean()),
        "cvar90_L": cvar(df["L"].to_numpy(), 0.9),
        "worst_L": float(df["L"].max()),
        "mean_V": float(df["V"].mean()),
        "cvar90_V": cvar(df["V"].to_numpy(), 0.9),
        "mean_C_U": float(df["C_U"].mean()),
        "mean_C_R": float(df["C_R"].mean()),
        "failure_rate": float(df["failure"].mean()),
        "fail_p_rate": float(df["fail_p"].mean()),
        "fail_a_rate": float(df["fail_a"].mean()),
        "fail_h_rate": float(df["fail_h"].mean()),
        "fail_phi_rate": float(df["fail_phi"].mean()),
        "fail_m_rate": float(df["fail_m"].mean()),
        "fail_I_rate": float(df["fail_I"].mean()),
        "mean_unsafe_duration": float(df["unsafe_duration"].mean()),
        "mean_final_I": float(df["final_I"].mean()),
        "worst_max_a": float(df["max_a"].max()),
        "worst_max_h": float(df["max_h"].max()),
        "worst_max_p": float(df["max_p"].max()),
        "worst_min_m": float(df["min_m"].min()),
    }
    return df, summary


def random_search_a10(
    base: Params,
    train_seeds: List[int],
    n_steps: int,
    n_iter: int,
    rng_seed: int = 1234,
) -> np.ndarray:
    """Simple derivative-free search over A10 alpha. No scipy required."""
    rng = np.random.default_rng(rng_seed)
    best_alpha = DEFAULT_A10_ALPHA.copy()
    _, best_summary = evaluate_controller("a10_train", controller_a10_guarded, train_seeds, base, n_steps, best_alpha)
    best_score = best_summary["mean_L"] + 2.0 * best_summary["cvar90_L"]

    print(f"[opt] initial score={best_score:.4f}")

    for i in range(n_iter):
        scale = 0.18 if i < n_iter // 2 else 0.08
        cand = best_alpha * np.exp(rng.normal(0.0, scale, size=best_alpha.shape))
        cand = np.clip(cand, 0.0, 2.5)
        _, summary = evaluate_controller("a10_train", controller_a10_guarded, train_seeds, base, n_steps, cand)
        score = summary["mean_L"] + 2.0 * summary["cvar90_L"]
        if score < best_score:
            best_score = score
            best_alpha = cand
            print(f"[opt] iter={i+1:03d} improved score={best_score:.4f}")
    return best_alpha


# -----------------------------
# Plotting
# -----------------------------

def save_trajectory_plots(outdir: str, base: Params, n_steps: int, alpha: np.ndarray | None = None) -> None:
    seed = 20260424
    scenario = sample_scenario(seed, base)
    selected = ["no_control", "power_only", "pid_damping", "lqr_like", "a10_v01", "a10_rpt_v02", "a10_barrier_v03", "a10_barrier_refined_v03"]

    trajectories = {}
    for name in selected:
        trajectories[name] = simulate(CONTROLLERS[name], scenario, n_steps=n_steps, alpha=alpha)

    labels = {
        "p": (0, "chamber-pressure proxy p", "trajectory_p.png", 1.0),
        "a": (1, "pressure-oscillation amplitude a", "trajectory_a.png", 1.0),
        "h": (2, "thermal-load proxy h", "trajectory_h.png", 1.0),
        "I": (5, "cumulative impulse I", "trajectory_I.png", 1.0),
    }
    for key, (idx, ylabel, filename, threshold) in labels.items():
        plt.figure(figsize=(9, 5))
        for name, out in trajectories.items():
            plt.plot(out["T"], out["X"][:, idx], label=name)
        plt.axhline(threshold, linestyle="--", linewidth=1.0)
        plt.xlabel("nondimensional burn time tau")
        plt.ylabel(ylabel)
        plt.title(f"A10-RPT Phase 1 representative trajectory: {key}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=180)
        plt.close()

    # Damping/control channels for A10 refined
    out = trajectories["a10_barrier_refined_v03"]
    control_names = ["uP", "uPhi", "uC", "uD"]
    for idx, cname in enumerate(control_names):
        plt.figure(figsize=(9, 5))
        plt.plot(out["T"], out["U"][:, idx], label="a10_barrier_refined_v03")
        plt.xlabel("nondimensional burn time tau")
        plt.ylabel(cname)
        plt.title(f"A10-RPT Phase 1 control trace: {cname}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"control_{cname}.png"), dpi=180)
        plt.close()


def save_summary_plots(outdir: str, metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    # CVaR bar
    plt.figure(figsize=(10, 5))
    plt.bar(summary["controller"], summary["cvar90_L"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("CVaR90 loss")
    plt.title("A10-RPT Phase 1: CVaR90 loss by controller")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cvar90_loss_by_controller.png"), dpi=180)
    plt.close()

    # Failure rate bar
    plt.figure(figsize=(10, 5))
    plt.bar(summary["controller"], summary["failure_rate"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("failure rate")
    plt.title("A10-RPT Phase 1: failure rate by controller")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "failure_rate_by_controller.png"), dpi=180)
    plt.close()

    # Unsafe duration bar
    plt.figure(figsize=(10, 5))
    plt.bar(summary["controller"], summary["mean_unsafe_duration"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("mean unsafe duration")
    plt.title("A10-RPT Phase 1: unsafe duration by controller")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "unsafe_duration_by_controller.png"), dpi=180)
    plt.close()

    # Violation-only CVaR bar, separated from control cost.
    plt.figure(figsize=(10, 5))
    plt.bar(summary["controller"], summary["cvar90_V"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("CVaR90 safe-set violation V")
    plt.title("A10-RPT Phase 1: violation-only CVaR by controller")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cvar90_violation_by_controller.png"), dpi=180)
    plt.close()

    reason_cols = ["fail_p_rate", "fail_a_rate", "fail_h_rate", "fail_phi_rate", "fail_m_rate", "fail_I_rate"]
    available = [c for c in reason_cols if c in summary.columns]
    if available:
        summary[["controller"] + available].to_csv(os.path.join(outdir, "failure_reason_rates.csv"), index=False)

    # Scatter: final impulse vs max oscillation
    plt.figure(figsize=(8, 6))
    for name, group in metrics.groupby("controller"):
        plt.scatter(group["final_I"], group["max_a"], label=name, alpha=0.7)
    plt.axvline(1.0, linestyle="--", linewidth=1.0)
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("final impulse I(1)")
    plt.ylabel("max pressure-oscillation amplitude max(a)")
    plt.title("A10-RPT Phase 1: mission attainment vs oscillation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_finalI_vs_maxa.png"), dpi=180)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="A10-RPT Phase 1 nondimensional surrogate test")
    parser.add_argument("--outdir", type=str, default="results_a10_rpt_phase1", help="output directory")
    parser.add_argument("--quick", action="store_true", help="quick run: fewer seeds and steps")
    parser.add_argument("--n-test", type=int, default=120, help="number of test scenarios")
    parser.add_argument("--n-steps", type=int, default=600, help="time steps per simulation")
    parser.add_argument("--optimize-a10", action="store_true", help="run simple random search for A10 alpha")
    parser.add_argument("--n-train", type=int, default=24, help="training scenarios for A10 alpha search")
    parser.add_argument("--n-opt", type=int, default=40, help="random-search iterations for A10 alpha")
    args = parser.parse_args()

    if args.quick:
        args.n_test = min(args.n_test, 30)
        args.n_steps = min(args.n_steps, 350)
        args.n_train = min(args.n_train, 10)
        args.n_opt = min(args.n_opt, 8)

    os.makedirs(args.outdir, exist_ok=True)

    base = Params()
    test_seeds = list(range(1000, 1000 + args.n_test))
    train_seeds = list(range(2000, 2000 + args.n_train))

    alpha = DEFAULT_A10_ALPHA.copy()
    if args.optimize_a10:
        alpha = random_search_a10(base, train_seeds, args.n_steps, args.n_opt)
        pd.DataFrame({"alpha": alpha}).to_csv(os.path.join(args.outdir, "optimized_a10_alpha.csv"), index=False)

    all_metrics = []
    summaries = []

    for name, controller in CONTROLLERS.items():
        print(f"[run] {name}")
        df, summary = evaluate_controller(name, controller, test_seeds, base, args.n_steps, alpha=alpha)
        all_metrics.append(df)
        summaries.append(summary)

    metrics = pd.concat(all_metrics, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values(["failure_rate", "mean_unsafe_duration", "cvar90_V", "cvar90_L"])

    metrics_path = os.path.join(args.outdir, "metrics_by_seed.csv")
    summary_path = os.path.join(args.outdir, "summary_by_controller.csv")
    metrics.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    save_trajectory_plots(args.outdir, base, args.n_steps, alpha=alpha)
    save_summary_plots(args.outdir, metrics, summary_df)

    print("\n=== Summary sorted by mission feasibility: failure_rate, unsafe_duration, cvar90_V ===")
    cols = [
        "controller", "mean_L", "cvar90_L", "cvar90_V", "failure_rate",
        "mean_unsafe_duration", "mean_final_I", "fail_h_rate", "fail_m_rate", "fail_I_rate",
        "worst_max_a", "worst_max_h", "worst_max_p", "worst_min_m"
    ]
    print(summary_df[cols].to_string(index=False))

    print(f"\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {args.outdir}/trajectory_a.png")
    print(f"  {args.outdir}/trajectory_h.png")
    print(f"  {args.outdir}/trajectory_p.png")
    print(f"  {args.outdir}/trajectory_I.png")
    print(f"  {args.outdir}/cvar90_loss_by_controller.png")
    print(f"  {args.outdir}/failure_rate_by_controller.png")
    print(f"  {args.outdir}/unsafe_duration_by_controller.png")
    print(f"  {args.outdir}/cvar90_violation_by_controller.png")
    print(f"  {args.outdir}/failure_reason_rates.csv")
    print(f"  {args.outdir}/scatter_finalI_vs_maxa.png")


if __name__ == "__main__":
    main()

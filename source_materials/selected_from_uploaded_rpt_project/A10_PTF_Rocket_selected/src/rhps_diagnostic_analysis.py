import jax
import jax.numpy as jnp
from jax import lax, value_and_grad, jit, vjp
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

jax.config.update("jax_enable_x64", True)

def define_params(tau_f=0.002):
    omega_a = 2.0 * jnp.pi * 150.0
    params = {
        'A_a': jnp.array([[0.0, 1.0], [-omega_a**2, -2.0*0.05*omega_a]]),
        'A_f': jnp.array([[0.0, 1.0], [-1.0/tau_f**2, -2.0/tau_f]]),
        'B_f': jnp.array([[0.0, 0.0], [0.0, 5000.0]]),
        'B_u': jnp.array([[0.0], [100.0]]),
        'B_fa': jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        'B_f_out': jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        'A_N': jnp.array([[-1000.0, 0.0], [0.0, -666.0]]),
        'B_Np': jnp.array([[1.0], [1.0]]),
        'B_N': jnp.array([[0.0, 0.0], [-100.0, -50.0]]),
        'kappa': 10.0, 'omega_a': omega_a, 'U_max': 50.0
    }
    return params

def rhps_step(state, u, params, dt):
    a, z_f, z_N = state
    q_sat = (jnp.dot(params['B_f_out'], z_f)) / (1.0 + params['kappa'] * jnp.abs(jnp.dot(params['B_f_out'], z_f)))
    da = jnp.dot(params['A_a'], a) + jnp.dot(params['B_f'], q_sat) + jnp.dot(params['B_N'], z_N) + jnp.dot(params['B_u'], u)
    dz_f = jnp.dot(params['A_f'], z_f) + jnp.dot(params['B_fa'], a)
    dz_N = jnp.dot(params['A_N'], z_N) + jnp.dot(params['B_Np'], jnp.array([a[0]]))
    return (a + dt * da, z_f + dt * dz_f, z_N + dt * dz_N)

@jit
def simulate(theta, time_seq, params, dt):
    carrier = theta[0] * jnp.sin(params['omega_a'] * time_seq) + theta[1] * jnp.cos(params['omega_a'] * time_seq)
    # 【修正】 .reshape(-1, 1) を追加して明示的に列ベクトル化
    u_seq = (params['U_max'] * jnp.tanh((carrier * jnp.exp(-theta[2] * time_seq)) / params['U_max'])).reshape(-1, 1)
    
    def scan_fn(state, u):
        next_state = rhps_step(state, u, params, dt)
        # 次元崩壊を防ぐため jnp.sum(u**2) を明示
        return next_state, (next_state[0], jnp.sum(next_state[0]**2) + 1e-4 * jnp.sum(u**2))
    
    init_state = (jnp.array([0.05, 0.0]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]))
    _, (a_seq, costs) = lax.scan(scan_fn, init_state, u_seq)
    return jnp.sum(costs) * dt, a_seq, u_seq

def analyze_diagnostics():
    params = define_params()
    dt, N_steps = 1e-4, 500
    time_seq = jnp.arange(N_steps) * dt
    # Phase 3で得られた最適パラメータを使用
    theta_opt = jnp.array([2.72e6, 1.88e2, 7.68e2])

    # 1. 随伴変数の抽出 (VJPを使用)
    def cost_from_u(u_in):
        def scan_fn(state, u):
            next_state = rhps_step(state, u, params, dt)
            return next_state, jnp.sum(next_state[0]**2) + 1e-4 * jnp.sum(u**2)
        _, costs = lax.scan(scan_fn, (jnp.array([0.05, 0.0]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0])), u_in)
        return jnp.sum(costs) * dt

    u_opt_seq = params['U_max'] * jnp.tanh(( (theta_opt[0] * jnp.sin(params['omega_a'] * time_seq) + theta_opt[1] * jnp.cos(params['omega_a'] * time_seq)) * jnp.exp(-theta_opt[2] * time_seq) ) / params['U_max']).reshape(-1, 1)
    
    val, vjp_fun = vjp(cost_from_u, u_opt_seq)
    adjoint_u = vjp_fun(1.0)[0] 

    # 2. 火炎遅れ tau_f に対する感度スキャン
    tau_range = np.linspace(0.001, 0.005, 20)
    costs_scan = [simulate(theta_opt, time_seq, define_params(tf), dt)[0] for tf in tau_range]

    # 3. 可視化
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    # 【修正】 raw string (r'') を使用して LaTeX 文字の警告を回避
    plt.plot(tau_range, costs_scan, 'o-', label=r'Cost vs Flame Delay $\tau_f$')
    plt.axvline(0.002, color='r', linestyle='--', label='Design Point')
    plt.ylabel('Cost (Instability)')
    plt.legend(); plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_seq, u_opt_seq, label='Saturated Control $u(t)$')
    plt.ylabel('Control'); plt.legend(); plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_seq, adjoint_u, label=r'Adjoint Sensitivity $\partial J/\partial u$', color='purple')
    plt.xlabel('Time (s)'); plt.ylabel('Adjoint'); plt.legend(); plt.grid(True)
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/diagnostic_analysis.png')
    print("--- 診断解析結果を output/diagnostic_analysis.png に保存しました ---")

if __name__ == "__main__":
    analyze_diagnostics()
import jax
import jax.numpy as jnp
from jax import lax, jacfwd, jit, value_and_grad
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

jax.config.update("jax_enable_x64", True)

def define_params_strict(tau_f=0.002):
    omega_a = 2.0 * jnp.pi * 150.0
    return {
        'A_a': jnp.array([[0.0, 1.0], [-omega_a**2, -2.0*0.01*omega_a]]),
        'A_f': jnp.array([[0.0, 1.0], [-1.0/tau_f**2, -2.0/tau_f]]),
        'B_f': jnp.array([[0.0, 0.0], [0.0, 8000.0]]),
        'B_u': jnp.array([0.0, 100.0]),
        'B_fa': jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        'B_f_out': jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        'A_N': jnp.array([[-1000.0, 0.0], [0.0, -666.0]]),
        'B_Np': jnp.array([[1.0], [1.0]]),
        'B_N': jnp.array([[0.0, 0.0], [-100.0, -50.0]]),
        'kappa': 10.0, 'omega_a': omega_a, 'U_max': 50.0, 'dt': 1e-4
    }

def Phi(x, u, params):
    a, z_f, z_N = x[0:2], x[2:4], x[4:6]
    q_sat = (jnp.dot(params['B_f_out'], z_f)) / (1.0 + params['kappa'] * jnp.abs(jnp.dot(params['B_f_out'], z_f)))
    da = jnp.dot(params['A_a'], a) + jnp.dot(params['B_f'], q_sat) + jnp.dot(params['B_N'], z_N) + (params['B_u'] * u)
    dz_f = jnp.dot(params['A_f'], z_f) + jnp.dot(params['B_fa'], a)
    dz_N = jnp.dot(params['A_N'], z_N) + jnp.dot(params['B_Np'], jnp.array([a[0]]))
    return jnp.concatenate([a + params['dt'] * da, z_f + params['dt'] * dz_f, z_N + params['dt'] * dz_N])

@jit
def u_a10(theta, t, params):
    carrier = theta[0] * jnp.sin(params['omega_a'] * t) + theta[1] * jnp.cos(params['omega_a'] * t)
    return params['U_max'] * jnp.tanh((carrier * jnp.exp(-theta[2] * t)) / params['U_max'])

@jit
def u_vanilla(theta, t, params):
    raw = theta[0] * jnp.sin(params['omega_a'] * t) + theta[1] * jnp.sin(2*params['omega_a'] * t) + theta[2] * jnp.sin(3*params['omega_a'] * t)
    return params['U_max'] * jnp.tanh(raw / params['U_max'])

def run_sim(theta, u_func, tau_f, steps=1000): # 時間を2倍(0.1s)に延ばして不安定化を待つ
    params = define_params_strict(tau_f)
    dt = params['dt']
    time_seq = jnp.arange(steps) * dt
    def scan_fn(state, t):
        u = u_func(theta, t, params)
        next_state = Phi(state, u, params)
        # コストは状態の二乗和 + 制御エネルギー
        return next_state, jnp.sum(next_state[0:2]**2) + 1e-3 * u**2
    init_state = jnp.concatenate([jnp.array([0.05, 0.0]), jnp.zeros(4)])
    _, costs = lax.scan(scan_fn, init_state, time_seq)
    return jnp.sum(costs) * dt

def run_test():
    print("--- 未知領域(OOD) 生存能力比較テスト ---")
    th_init = np.array([1.0, 1.0, 1.0])
    
    # 設計点 (0.002) での最適化
    print("Optimizing at Design Point (tau_f = 0.002)...")
    obj_a10 = jit(value_and_grad(lambda th: run_sim(th, u_a10, 0.002)))
    res_a10 = minimize(lambda th: tuple(np.array(v) for v in obj_a10(jnp.array(th))), th_init, method='L-BFGS-B', jac=True)

    obj_vanilla = jit(value_and_grad(lambda th: run_sim(th, u_vanilla, 0.002)))
    res_vanilla = minimize(lambda th: tuple(np.array(v) for v in obj_vanilla(jnp.array(th))), th_init, method='L-BFGS-B', jac=True)

    # 想定外の広範囲スキャン (0.001 ～ 0.010)
    tau_range = np.linspace(0.001, 0.010, 20)
    print("OOD Scanning...")
    costs_a10 = [run_sim(jnp.array(res_a10.x), u_a10, tf) for tf in tau_range]
    costs_vanilla = [run_sim(jnp.array(res_vanilla.x), u_vanilla, tf) for tf in tau_range]

    plt.figure(figsize=(10, 6))
    plt.plot(tau_range, costs_a10, 'o-', label='A10-Ansatz', color='blue', linewidth=2)
    plt.plot(tau_range, costs_vanilla, 's--', label='Vanilla-Policy', color='gray')
    plt.axvline(0.002, color='red', linestyle=':', label='Training Point')
    plt.yscale('log')
    plt.xlabel(r'Flame Delay $\tau_f$')
    plt.ylabel('Total OOD Cost (Log Scale)')
    plt.title('Out-of-Distribution Survival: A10 vs Vanilla')
    plt.legend(); plt.grid(True)
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/ood_survival_test.png')
    
    # 生存境界の評価（コストが10倍以上に跳ね上がる点）
    def find_break_point(costs):
        base = costs[np.argmin(np.abs(tau_range - 0.002))]
        for i, c in enumerate(costs):
            if c > base * 10: return tau_range[i]
        return tau_range[-1]

    print(f"\nA10 生存限界 tau_f: {find_break_point(costs_a10):.4f}")
    print(f"Vanilla 生存限界 tau_f: {find_break_point(costs_vanilla):.4f}")

if __name__ == "__main__":
    run_test()
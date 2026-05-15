import jax
import jax.numpy as jnp
from jax import lax, jacfwd, jit, value_and_grad
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

jax.config.update("jax_enable_x64", True)

def define_params(tau_f=0.002):
    omega_a = 2.0 * jnp.pi * 150.0
    return {
        'A_a': jnp.array([[0.0, 1.0], [-omega_a**2, -2.0*0.05*omega_a]]),
        'A_f': jnp.array([[0.0, 1.0], [-1.0/tau_f**2, -2.0/tau_f]]),
        'B_f': jnp.array([[0.0, 0.0], [0.0, 5000.0]]),
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
    raw = carrier * jnp.exp(-theta[2] * t)
    return params['U_max'] * jnp.tanh(raw / params['U_max'])

@jit
def u_vanilla(theta, t, params):
    raw = theta[0] * jnp.sin(params['omega_a'] * t) + theta[1] * jnp.sin(2*params['omega_a'] * t) + theta[2] * jnp.sin(3*params['omega_a'] * t)
    return params['U_max'] * jnp.tanh(raw / params['U_max'])

def run_simulation(theta, u_func, tau_f):
    params = define_params(tau_f)
    dt, N_steps = params['dt'], 500
    time_seq = jnp.arange(N_steps) * dt
    
    def scan_fn(state, t):
        u = u_func(theta, t, params)
        next_state = Phi(state, u, params)
        return next_state, jnp.sum(next_state[0:2]**2) + 1e-4 * u**2
    
    init_state = jnp.concatenate([jnp.array([0.05, 0.0]), jnp.zeros(2), jnp.zeros(2)])
    _, costs = lax.scan(scan_fn, init_state, time_seq)
    return jnp.sum(costs) * dt

def get_cost_and_grad(theta, u_func):
    val, grad = value_and_grad(lambda th: run_simulation(th, u_func, 0.002))(theta)
    return val, grad

def run_comparison():
    print("--- 動的コスト感度の比較検証 ---")
    th_init = np.array([1.0, 1.0, 1.0])
    
    print("A10 最適化中...")
    res_a10 = minimize(lambda th: tuple(np.array(v) for v in get_cost_and_grad(jnp.array(th), u_a10)), 
                       th_init, method='L-BFGS-B', jac=True)
    
    print("Vanilla 最適化中...")
    res_vanilla = minimize(lambda th: tuple(np.array(v) for v in get_cost_and_grad(jnp.array(th), u_vanilla)), 
                          th_init, method='L-BFGS-B', jac=True)

    tau_range = np.linspace(0.001, 0.005, 20)
    costs_a10 = [run_simulation(jnp.array(res_a10.x), u_a10, tf) for tf in tau_range]
    costs_vanilla = [run_simulation(jnp.array(res_vanilla.x), u_vanilla, tf) for tf in tau_range]

    plt.figure(figsize=(10, 6))
    plt.plot(tau_range, costs_a10, 'o-', label='A10-Ansatz (Geometric Prior)', color='blue')
    plt.plot(tau_range, costs_vanilla, 's--', label='Vanilla-Policy (No Structure)', color='gray')
    plt.axvline(0.002, color='red', linestyle=':', label='Design Point')
    plt.xlabel(r'Flame Delay $\tau_f$')
    plt.ylabel('Total Cost $J$')
    plt.title('Dynamic Robustness Comparison: A10 vs Vanilla')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/dynamic_cost_comparison.png')
    
    # 頑健性指数（感度の平均）の算出
    sens_a10 = np.mean(np.abs(np.diff(costs_a10) / np.diff(tau_range)))
    sens_vanilla = np.mean(np.abs(np.diff(costs_vanilla) / np.diff(tau_range)))
    
    print(f"\nA10 平均コスト感度: {sens_a10:.5e}")
    print(f"Vanilla 平均コスト感度: {sens_vanilla:.5e}")
    print(f"ロバスト性向上倍率: {sens_vanilla / sens_a10:.2f}x")

if __name__ == "__main__":
    run_comparison()
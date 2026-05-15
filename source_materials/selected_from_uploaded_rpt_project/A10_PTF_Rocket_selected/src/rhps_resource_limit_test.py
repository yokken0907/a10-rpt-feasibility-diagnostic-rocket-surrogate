import jax
import jax.numpy as jnp
from jax import lax, jacfwd, jit, value_and_grad
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

jax.config.update("jax_enable_x64", True)

def define_params_restricted(tau_f=0.002, u_limit=15.0): # 物理限界を大幅に下げる
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
        'kappa': 10.0, 'omega_a': omega_a, 'U_max': u_limit, 'dt': 1e-4
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

def run_constrained_sim(theta, u_func, u_limit):
    params = define_params_restricted(0.002, u_limit)
    dt, N_steps = params['dt'], 1000
    time_seq = jnp.arange(N_steps) * dt
    def scan_fn(state, t):
        u = u_func(theta, t, params)
        next_state = Phi(state, u, params)
        return next_state, jnp.sum(next_state[0:2]**2)
    init_state = jnp.concatenate([jnp.array([0.05, 0.0]), jnp.zeros(4)])
    _, costs = lax.scan(scan_fn, init_state, time_seq)
    return jnp.sum(costs) * dt

def run_resource_test():
    u_limits = [50.0, 30.0, 20.0, 15.0, 10.0]
    results_a10 = []
    results_vanilla = []
    
    th_init = np.array([1.0, 1.0, 1.0])

    print("--- リソース制限下の性能比較テスト ---")
    for lim in u_limits:
        print(f"Testing U_max = {lim}...")
        
        # A10 最適化
        obj_a10 = jit(value_and_grad(lambda th: run_constrained_sim(th, u_a10, lim)))
        res_a10 = minimize(lambda th: tuple(np.array(v) for v in obj_a10(jnp.array(th))), th_init, method='L-BFGS-B', jac=True)
        results_a10.append(res_a10.fun)

        # Vanilla 最適化
        obj_vanilla = jit(value_and_grad(lambda th: run_constrained_sim(th, u_vanilla, lim)))
        res_vanilla = minimize(lambda th: tuple(np.array(v) for v in obj_vanilla(jnp.array(th))), th_init, method='L-BFGS-B', jac=True)
        results_vanilla.append(res_vanilla.fun)

    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(u_limits, results_a10, 'o-', label='A10-Ansatz', color='blue')
    plt.plot(u_limits, results_vanilla, 's--', label='Vanilla-Policy', color='gray')
    plt.gca().invert_xaxis() # 左に行くほど制約が厳しい
    plt.xlabel(r'Physical Limit $U_{\max}$')
    plt.ylabel('Minimum Attainable Cost $J$')
    plt.title('Efficiency under Resource Scarcity: A10 vs Vanilla')
    plt.legend(); plt.grid(True)
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/resource_limit_comparison.png')
    
    # 改善率の計算（最も厳しい条件）
    imp = results_vanilla[-1] / results_a10[-1]
    print(f"\n[最終結果] 最も厳しい制約 (U_max=10.0) におけるコスト比 (Vanilla/A10): {imp:.2f}x")

if __name__ == "__main__":
    run_resource_test()
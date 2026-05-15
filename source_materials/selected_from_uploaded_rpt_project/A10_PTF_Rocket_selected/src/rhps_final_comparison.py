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
        'B_u': jnp.array([0.0, 100.0]), # (2,) ベクトルに修正
        'B_fa': jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        'B_f_out': jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        'A_N': jnp.array([[-1000.0, 0.0], [0.0, -666.0]]),
        'B_Np': jnp.array([[1.0], [1.0]]),
        'B_N': jnp.array([[0.0, 0.0], [-100.0, -50.0]]),
        'kappa': 10.0, 'omega_a': omega_a, 'U_max': 50.0, 'dt': 1e-4
    }

def pack_state(a, z_f, z_N):
    return jnp.concatenate([a, z_f, z_N])

def Phi(x, u, params):
    a = x[0:2]
    z_f = x[2:4]
    z_N = x[4:6]
    q_sat = (jnp.dot(params['B_f_out'], z_f)) / (1.0 + params['kappa'] * jnp.abs(jnp.dot(params['B_f_out'], z_f)))
    
    # スカラー u との演算を安全に行う
    da = jnp.dot(params['A_a'], a) + jnp.dot(params['B_f'], q_sat) + jnp.dot(params['B_N'], z_N) + (params['B_u'] * u)
    dz_f = jnp.dot(params['A_f'], z_f) + jnp.dot(params['B_fa'], a)
    dz_N = jnp.dot(params['A_N'], z_N) + jnp.dot(params['B_Np'], jnp.array([a[0]]))
    
    return jnp.concatenate([a + params['dt'] * da, z_f + params['dt'] * dz_f, z_N + params['dt'] * dz_N])

Phi_x = jit(jacfwd(Phi, argnums=0))

@jit
def u_a10(theta, t, omega_a, u_max):
    carrier = theta[0] * jnp.sin(omega_a * t) + theta[1] * jnp.cos(omega_a * t)
    raw = carrier * jnp.exp(-theta[2] * t)
    return u_max * jnp.tanh(raw / u_max)

@jit
def u_vanilla(theta, t, omega_a, u_max):
    # 物理的構造を持たない比較用ポリシー
    raw = theta[0] * jnp.sin(omega_a * t) + theta[1] * jnp.sin(2*omega_a * t) + theta[2] * jnp.sin(3*omega_a * t)
    return u_max * jnp.tanh(raw / u_max)

def get_cost_and_grad(theta, u_func):
    params = define_params(0.002)
    dt, N_steps = params['dt'], 500
    time_seq = jnp.arange(N_steps) * dt
    
    @jit
    def cost_fn(th):
        def scan_fn(state, t):
            u = u_func(th, t, params['omega_a'], params['U_max'])
            next_state = Phi(state, u, params)
            return next_state, jnp.sum(next_state[0:2]**2) + 1e-4 * u**2
        init_state = pack_state(jnp.array([0.05, 0.0]), jnp.zeros(2), jnp.zeros(2))
        _, costs = lax.scan(scan_fn, init_state, time_seq)
        return jnp.sum(costs) * dt

    val, grad = value_and_grad(cost_fn)(theta)
    return val, grad

def get_sensitivity(theta, u_func):
    dt = 1e-4
    tau_range = np.linspace(0.0015, 0.0035, 10)
    
    re_eigs = []
    for tf in tau_range:
        p = define_params(tf)
        def closed_loop_Phi(x):
            # 安定性を評価するための摂動点（x=0, t=0）でのヤコビアン
            u = u_func(theta, 0.0, p['omega_a'], p['U_max'])
            return Phi(x, u, p)
        
        J = jacfwd(closed_loop_Phi)(jnp.zeros(6))
        eig = jnp.max(jnp.log(jnp.abs(jnp.linalg.eigvals(J)))) / dt
        re_eigs.append(eig)
    
    return np.mean(np.abs(np.diff(re_eigs) / np.diff(tau_range)))

def run_final_comparison():
    print("--- 比較検証開始: A10-Ansatz vs Vanilla-Policy ---")
    th_init = np.array([1.0, 1.0, 1.0])
    
    print("Optimizing A10...")
    res_a10 = minimize(lambda th: tuple(np.array(v) for v in get_cost_and_grad(jnp.array(th), u_a10)), 
                       th_init, method='L-BFGS-B', jac=True, options={'maxiter': 50})
    
    print("Optimizing Vanilla...")
    res_vanilla = minimize(lambda th: tuple(np.array(v) for v in get_cost_and_grad(jnp.array(th), u_vanilla)), 
                          th_init, method='L-BFGS-B', jac=True, options={'maxiter': 50})

    sens_a10 = get_sensitivity(jnp.array(res_a10.x), u_a10)
    sens_vanilla = get_sensitivity(jnp.array(res_vanilla.x), u_vanilla)

    print(f"\n--- 最終証明結果 ---")
    print(f"A10-Ansatz Sensitivity: {sens_a10:.5e}")
    print(f"Vanilla-Policy Sensitivity: {sens_vanilla:.5e}")
    print(f"Improvement Ratio (Robustness): {sens_vanilla / sens_a10:.2f}x")

    if sens_a10 < sens_vanilla:
        print("\n結論: A10アンスァッツはVanilla政策に比べ、物理変動に対して能動的な低感度化を実現している。")
    else:
        print("\n結論: 感度差が有意に認められない。アンスァッツの構成的優位性を主張するには更なる検討が必要。")

if __name__ == "__main__":
    run_final_comparison()
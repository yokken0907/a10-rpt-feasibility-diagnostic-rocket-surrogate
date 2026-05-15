import jax
import jax.numpy as jnp
from jax import lax, value_and_grad, jit
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from functools import partial  # 追加: 静的引数を指定するためのモジュール

jax.config.update("jax_enable_x64", True)

def define_rhps_parameters():
    omega_a = 2.0 * jnp.pi * 150.0
    zeta_a  = 0.05
    A_a = jnp.array([[0.0, 1.0], [-omega_a**2, -2.0*zeta_a*omega_a]])
    
    tau_f = 0.002
    A_f = jnp.array([[0.0, 1.0], [-1.0/tau_f**2, -2.0/tau_f]])
    B_fa = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    B_f_out = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    B_f = jnp.array([[0.0, 0.0], [0.0, 5000.0]])
    
    tau_N = 0.001
    A_N = jnp.array([[-1.0/tau_N, 0.0], [0.0, -1.0/(tau_N*1.5)]])
    B_Np = jnp.array([[1.0], [1.0]])
    B_N = jnp.array([[0.0, 0.0], [-100.0, -50.0]])
    
    B_u = jnp.array([[0.0], [1.0]])

    params = {
        'A_a': A_a, 'B_f': B_f, 'B_N': B_N, 'B_u': B_u,
        'A_f': A_f, 'B_fa': B_fa, 'B_f_out': B_f_out,
        'A_N': A_N, 'B_Np': B_Np,
        'kappa': 10.0
    }
    return params

def smooth_saturation(x, kappa):
    return x / (1.0 + kappa * jnp.abs(x))

def rhps_step(state, u, params, dt):
    a, z_f, z_N = state
    q_prime = jnp.dot(params['B_f_out'], z_f)
    q_prime_sat = smooth_saturation(q_prime, params['kappa'])
    p_prime = jnp.array([a[0]])
    
    da = jnp.dot(params['A_a'], a) + jnp.dot(params['B_f'], q_prime_sat) + jnp.dot(params['B_N'], z_N) + jnp.dot(params['B_u'], u)
    dz_f = jnp.dot(params['A_f'], z_f) + jnp.dot(params['B_fa'], a)
    dz_N = jnp.dot(params['A_N'], z_N) + jnp.dot(params['B_Np'], p_prime)
    
    return (a + dt * da, z_f + dt * dz_f, z_N + dt * dz_N)

@jit
def simulate_trajectory(u_seq, init_state, params, dt):
    def scan_fn(state, u):
        next_state = rhps_step(state, u, params, dt)
        a, _, _ = next_state
        cost = jnp.sum(a**2) + 0.1 * jnp.sum(u**2)
        return next_state, (a, cost)
    
    final_state, (a_seq, costs) = lax.scan(scan_fn, init_state, u_seq)
    total_cost = jnp.sum(costs) * dt
    return total_cost, a_seq

# 修正箇所: N_steps (5番目の引数, index 4) は静的な整数であるとJAXに明示する
@partial(jit, static_argnums=(4,))
def objective_with_grad(u_seq_flat, init_state, params, dt, N_steps):
    u_seq = u_seq_flat.reshape((N_steps, 1))
    val, grad = value_and_grad(lambda u: simulate_trajectory(u, init_state, params, dt)[0])(u_seq)
    return val, grad.flatten()

def run_optimization():
    params = define_rhps_parameters()
    dt = 1e-4
    N_steps = 500
    
    init_state = (jnp.array([1.0, 0.0]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]))
    u_initial = np.zeros(N_steps)
    
    def scipy_wrapper(u_flat):
        val, grad = objective_with_grad(u_flat, init_state, params, dt, N_steps)
        return np.array(val), np.array(grad)

    print("--- 制御なし (No Control) の軌道を計算中 ---")
    cost_no_control, a_seq_no_control = simulate_trajectory(u_initial.reshape(-1, 1), init_state, params, dt)

    print("--- L-BFGS-B による最適化を実行中 ---")
    # 修正箇所: 警告を消すためにオプションから 'disp': True を削除
    res = minimize(scipy_wrapper, u_initial, method='L-BFGS-B', jac=True, options={'maxiter': 100})
    
    u_opt = res.x.reshape((N_steps, 1))
    cost_opt, a_seq_opt = simulate_trajectory(u_opt, init_state, params, dt)
    
    print(f"初期コスト (制御なし): {cost_no_control:.5e}")
    print(f"最適化後コスト: {cost_opt:.5e}")
    
    time_axis = np.arange(N_steps) * dt
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, a_seq_no_control[:, 0], label='No Control (Pressure Perturbation)', linestyle='--')
    plt.plot(time_axis, a_seq_opt[:, 0], label='Optimized Control', linewidth=2)
    plt.ylabel('Pressure Perturbation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, u_opt[:, 0], label='Control Input (u)', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Actuator Effort')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/optimization_result.png')
    print("--- グラフを output/optimization_result.png に保存しました ---")

if __name__ == "__main__":
    run_optimization()
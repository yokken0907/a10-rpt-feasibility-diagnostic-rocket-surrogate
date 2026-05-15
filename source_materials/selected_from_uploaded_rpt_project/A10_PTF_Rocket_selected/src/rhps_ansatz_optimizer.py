import jax
import jax.numpy as jnp
from jax import lax, value_and_grad, jit
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

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
        'kappa': 10.0,
        'omega_a': omega_a
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

# A10アンスァッツ: 低次元構造化事前分布による制御生成
@jit
def generate_ansatz_u(theta, time_seq, omega_a):
    # theta[0]: sin成分の振幅
    # theta[1]: cos成分の振幅 (位相シフトを表現)
    # theta[2]: 減衰エンベロープ
    carrier = theta[0] * jnp.sin(omega_a * time_seq) + theta[1] * jnp.cos(omega_a * time_seq)
    envelope = jnp.exp(-theta[2] * time_seq)
    return (carrier * envelope).reshape(-1, 1)

@jit
def simulate_trajectory_ansatz(theta, time_seq, init_state, params, dt):
    # 3次元パラメータ theta から 500ステップの u_seq を生成
    u_seq = generate_ansatz_u(theta, time_seq, params['omega_a'])
    
    def scan_fn(state, u):
        next_state = rhps_step(state, u, params, dt)
        a, _, _ = next_state
        # ペナルティ付き音響エネルギーコスト
        cost = jnp.sum(a**2) + 1e-4 * jnp.sum(u**2)
        return next_state, (a, cost)
    
    final_state, (a_seq, costs) = lax.scan(scan_fn, init_state, u_seq)
    total_cost = jnp.sum(costs) * dt
    return total_cost, a_seq, u_seq

# 勾配計算の対象を 500次元の u_seq ではなく、3次元の theta に変更
@jit
def objective_with_grad_theta(theta, time_seq, init_state, params, dt):
    val, grad = value_and_grad(lambda th: simulate_trajectory_ansatz(th, time_seq, init_state, params, dt)[0])(theta)
    return val, grad

def run_ansatz_optimization():
    params = define_rhps_parameters()
    dt = 1e-4
    N_steps = 500
    time_seq = jnp.arange(N_steps) * dt
    
    init_state = (jnp.array([1.0, 0.0]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]))
    
    # パラメータの初期推定値 [sin振幅, cos振幅, 減衰率]
    theta_initial = np.array([0.0, 0.0, 10.0])
    
    def scipy_wrapper(theta_np):
        val, grad = objective_with_grad_theta(jnp.array(theta_np), time_seq, init_state, params, dt)
        return np.array(val), np.array(grad)

    print("--- A10アンスァッツ最適化を実行中（探索空間: 3次元） ---")
    res = minimize(scipy_wrapper, theta_initial, method='L-BFGS-B', jac=True, options={'maxiter': 100})
    
    theta_opt = res.x
    cost_opt, a_seq_opt, u_seq_opt = simulate_trajectory_ansatz(jnp.array(theta_opt), time_seq, init_state, params, dt)
    
    # 比較用：制御なし
    cost_no_control, a_seq_no_control, _ = simulate_trajectory_ansatz(jnp.array([0.0, 0.0, 0.0]), time_seq, init_state, params, dt)
    
    print(f"最適化されたパラメータ (theta): {theta_opt}")
    print(f"初期コスト (制御なし): {cost_no_control:.5e}")
    print(f"最適化後コスト: {cost_opt:.5e}")
    print(f"改善率: {(cost_no_control - cost_opt) / cost_no_control * 100:.2f}%")
    
    time_axis = np.array(time_seq)
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, a_seq_no_control[:, 0], label='No Control', linestyle='--')
    plt.plot(time_axis, a_seq_opt[:, 0], label='A10-Ansatz Optimized Control', linewidth=2)
    plt.ylabel('Pressure Perturbation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, u_seq_opt[:, 0], label='Ansatz Input $u(t)$', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Actuator Effort')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/ansatz_optimization_result.png')
    print("--- グラフを output/ansatz_optimization_result.png に保存しました ---")

if __name__ == "__main__":
    run_ansatz_optimization()
import jax
import jax.numpy as jnp
from jax import lax, value_and_grad, jit, vmap
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from functools import partial

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
    
    B_u = jnp.array([[0.0], [100.0]])  # Phase 2.1で適正化された制御ゲイン

    params = {
        'A_a': A_a, 'B_f': B_f, 'B_N': B_N, 'B_u': B_u,
        'A_f': A_f, 'B_fa': B_fa, 'B_f_out': B_f_out,
        'A_N': A_N, 'B_Np': B_Np,
        'kappa': 10.0,
        'omega_a': omega_a,
        'U_max': 50.0,
        'noise_std': 1000.0  # 注入する乱流ノイズの強度
    }
    return params

def smooth_saturation(x, kappa):
    return x / (1.0 + kappa * jnp.abs(x))

def rhps_step_noisy(state, u, noise, params, dt):
    a, z_f, z_N = state
    q_prime = jnp.dot(params['B_f_out'], z_f)
    q_prime_sat = smooth_saturation(q_prime, params['kappa'])
    p_prime = jnp.array([a[0]])
    
    # ノイズを音響方程式（加速度成分）に注入
    da = jnp.dot(params['A_a'], a) + jnp.dot(params['B_f'], q_prime_sat) + jnp.dot(params['B_N'], z_N) + jnp.dot(params['B_u'], u)
    da = da + jnp.array([0.0, noise[0]]) 
    
    dz_f = jnp.dot(params['A_f'], z_f) + jnp.dot(params['B_fa'], a)
    dz_N = jnp.dot(params['A_N'], z_N) + jnp.dot(params['B_Np'], p_prime)
    
    return (a + dt * da, z_f + dt * dz_f, z_N + dt * dz_N)

@jit
def generate_ansatz_u_saturated(theta, time_seq, omega_a, U_max):
    carrier = theta[0] * jnp.sin(omega_a * time_seq) + theta[1] * jnp.cos(omega_a * time_seq)
    envelope = jnp.exp(-theta[2] * time_seq)
    raw_u = carrier * envelope
    return U_max * jnp.tanh(raw_u / U_max).reshape(-1, 1)

# 単一のノイズ軌道をシミュレーションする関数
def simulate_single_trajectory(theta, time_seq, init_state, params, dt, noise_seq):
    u_seq = generate_ansatz_u_saturated(theta, time_seq, params['omega_a'], params['U_max'])
    
    def scan_fn(state, carry):
        u, noise = carry
        next_state = rhps_step_noisy(state, u, noise, params, dt)
        a, _, _ = next_state
        cost = jnp.sum(a**2) + 1e-4 * jnp.sum(u**2)
        return next_state, (a, cost)
    
    final_state, (a_seq, costs) = lax.scan(scan_fn, init_state, (u_seq, noise_seq))
    return jnp.sum(costs) * dt, a_seq, u_seq

# vmapを使って、複数のノイズ軌道を一括処理するバッチ関数を生成
batch_simulate = vmap(simulate_single_trajectory, in_axes=(None, None, None, None, None, 0))

@partial(jit, static_argnums=(4,))
def objective_with_grad_robust(theta, time_seq, init_state, params, dt, noise_matrix):
    def robust_cost_fn(th):
        # 20パターンの軌道を一括計算
        costs, _, _ = batch_simulate(th, time_seq, init_state, params, dt, noise_matrix)
        
        # リスク考慮型目的関数: 平均コスト + 1.0 * 標準偏差 (テールリスクの抑制)
        mean_cost = jnp.mean(costs)
        std_cost = jnp.std(costs)
        return mean_cost + 1.0 * std_cost
        
    val, grad = value_and_grad(robust_cost_fn)(theta)
    return val, grad

def run_robust_optimization():
    params = define_rhps_parameters()
    dt = 1e-4
    N_steps = 500
    N_batch = 20  # 20個の並行宇宙（アンサンブル）
    time_seq = jnp.arange(N_steps) * dt
    
    init_state = (jnp.array([0.05, 0.0]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]))
    theta_initial = np.array([1.0, 0.0, 1.0])
    
    # 乱数シードを固定し、N_batch個の異なるノイズシーケンスを生成
    key = jax.random.PRNGKey(42)
    noise_matrix = jax.random.normal(key, (N_batch, N_steps, 1)) * params['noise_std']
    
    def scipy_wrapper(theta_np):
        val, grad = objective_with_grad_robust(jnp.array(theta_np), time_seq, init_state, params, dt, noise_matrix)
        return np.array(val), np.array(grad)

    print(f"--- 乱気流ノイズ環境下でのロバスト・アンサンブル最適化を実行中 ---")
    print(f"--- アンサンブル数: {N_batch}, 物理限界: U_max={params['U_max']} ---")
    
    res = minimize(scipy_wrapper, theta_initial, method='L-BFGS-B', jac=True, options={'maxiter': 200})
    theta_opt = res.x
    
    # 結果の取得（最適化後と制御なしの両方をバッチ計算）
    costs_opt, a_seqs_opt, u_seqs_opt = batch_simulate(jnp.array(theta_opt), time_seq, init_state, params, dt, noise_matrix)
    costs_no, a_seqs_no, _ = batch_simulate(jnp.array([0.0, 0.0, 0.0]), time_seq, init_state, params, dt, noise_matrix)
    
    mean_no = np.mean(costs_no)
    mean_opt = np.mean(costs_opt)
    worst_opt = np.max(costs_opt)
    
    print(f"最適化パラメータ: {theta_opt}")
    print(f"制御なし 平均コスト: {mean_no:.5e}")
    print(f"最適化後 平均コスト: {mean_opt:.5e}")
    print(f"最適化後 最悪コスト (Worst-case): {worst_opt:.5e}")
    print(f"平均改善率: {(mean_no - mean_opt) / mean_no * 100:.2f}%")
    
    # プロット（20軌道をすべて描画して、ロバスト性を視覚化）
    time_axis = np.array(time_seq)
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    for i in range(N_batch):
        plt.plot(time_axis, a_seqs_no[i, :, 0], color='gray', alpha=0.2)
        plt.plot(time_axis, a_seqs_opt[i, :, 0], color='blue', alpha=0.3)
    plt.plot(time_axis, np.mean(a_seqs_opt[:, :, 0], axis=0), color='cyan', linewidth=2, label='Mean Saturated A10 Control')
    plt.ylabel('Pressure Perturbation')
    plt.title('Ensemble Robust Control vs Noisy Environment')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, u_seqs_opt[0, :, 0], label=f'Actuator Input (Saturated)', color='red')
    plt.axhline(params['U_max'], color='black', linestyle=':')
    plt.axhline(-params['U_max'], color='black', linestyle=':')
    plt.xlabel('Time (s)')
    plt.ylabel('Actuator Effort')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/robust_optimization_result.png')
    print("--- グラフを output/robust_optimization_result.png に保存しました ---")

if __name__ == "__main__":
    run_robust_optimization()
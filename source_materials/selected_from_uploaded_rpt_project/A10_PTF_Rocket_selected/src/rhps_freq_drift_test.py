import jax
import jax.numpy as jnp
from jax import lax, jacfwd, jit, value_and_grad
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

jax.config.update("jax_enable_x64", True)

# システムパラメータ定義（固有周波数を変動させられるように設定）
def define_params_drift(omega_actual_hz):
    omega_a = 2.0 * jnp.pi * omega_actual_hz
    return {
        'A_a': jnp.array([[0.0, 1.0], [-omega_a**2, -2.0*0.01*omega_a]]),
        'A_f': jnp.array([[0.0, 1.0], [-1.0/0.002**2, -2.0/0.002]]),
        'B_f': jnp.array([[0.0, 0.0], [0.0, 8000.0]]), # 不安定化設定
        'B_u': jnp.array([0.0, 100.0]),
        'B_fa': jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        'B_f_out': jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        'A_N': jnp.array([[-1000.0, 0.0], [0.0, -666.0]]),
        'B_Np': jnp.array([[1.0], [1.0]]),
        'B_N': jnp.array([[0.0, 0.0], [-100.0, -50.0]]),
        'kappa': 10.0, 'omega_a': omega_a, 'U_max': 25.0, 'dt': 1e-4
    }

# 状態遷移
def Phi(x, u, params):
    a, z_f, z_N = x[0:2], x[2:4], x[4:6]
    q_sat = (jnp.dot(params['B_f_out'], z_f)) / (1.0 + params['kappa'] * jnp.abs(jnp.dot(params['B_f_out'], z_f)))
    da = jnp.dot(params['A_a'], a) + jnp.dot(params['B_f'], q_sat) + jnp.dot(params['B_N'], z_N) + (params['B_u'] * u)
    dz_f = jnp.dot(params['A_f'], z_f) + jnp.dot(params['B_fa'], a)
    dz_N = jnp.dot(params['A_N'], z_N) + jnp.dot(params['B_Np'], jnp.array([a[0]]))
    return jnp.concatenate([a + params['dt'] * da, z_f + params['dt'] * dz_f, z_N + params['dt'] * dz_N])

# A10アンスァッツ (設計周波数 150Hz に固定)
@jit
def u_a10_fixed(theta, t, params):
    omega_design = 2.0 * jnp.pi * 150.0
    carrier = theta[0] * jnp.sin(omega_design * t) + theta[1] * jnp.cos(omega_design * t)
    raw = carrier * jnp.exp(-theta[2] * t)
    return params['U_max'] * jnp.tanh(raw / params['U_max'])

# Vanillaポリシー (設計周波数 150Hz に固定)
@jit
def u_vanilla_fixed(theta, t, params):
    omega_design = 2.0 * jnp.pi * 150.0
    raw = theta[0] * jnp.sin(omega_design * t) + theta[1] * jnp.sin(2*omega_design * t) + theta[2] * jnp.sin(3*omega_design * t)
    return params['U_max'] * jnp.tanh(raw / params['U_max'])

# シミュレーション実行関数
def run_sim(theta, u_func, omega_actual_hz):
    params = define_params_drift(omega_actual_hz)
    dt, N_steps = params['dt'], 800
    time_seq = jnp.arange(N_steps) * dt
    def scan_fn(state, t):
        u = u_func(theta, t, params)
        next_state = Phi(state, u, params)
        return next_state, jnp.sum(next_state[0:2]**2)
    init_state = jnp.concatenate([jnp.array([0.05, 0.0]), jnp.zeros(4)])
    _, costs = lax.scan(scan_fn, init_state, time_seq)
    return jnp.sum(costs) * dt

def run_drift_test():
    print("--- 周波数変動(Frequency Drift)耐性テスト ---")
    th_init = np.array([1.0, 1.0, 1.0])
    
    # 1. 設計点 (150Hz) での最適化
    print("150Hzの設計環境で最適化を実行中...")
    obj_a10 = jit(value_and_grad(lambda th: run_sim(th, u_a10_fixed, 150.0)))
    res_a10 = minimize(lambda th: tuple(np.array(v) for v in obj_a10(jnp.array(th))), th_init, method='L-BFGS-B', jac=True)

    obj_vanilla = jit(value_and_grad(lambda th: run_sim(th, u_vanilla_fixed, 150.0)))
    res_vanilla = minimize(lambda th: tuple(np.array(v) for v in obj_vanilla(jnp.array(th))), th_init, method='L-BFGS-B', jac=True)

    # 2. 周波数を変動させて、固定パラメータで評価
    freq_range = np.linspace(140, 160, 21)
    costs_a10 = []
    costs_vanilla = []

    print("周波数ドリフト(140Hz-160Hz)に対する耐性をスキャン中...")
    for f in freq_range:
        costs_a10.append(run_sim(jnp.array(res_a10.x), u_a10_fixed, f))
        costs_vanilla.append(run_sim(jnp.array(res_vanilla.x), u_vanilla_fixed, f))

    # 3. 結果の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(freq_range, costs_a10, 'o-', label='A10-Ansatz (Fixed at 150Hz)', color='blue')
    plt.plot(freq_range, costs_vanilla, 's--', label='Vanilla-Policy (Fixed at 150Hz)', color='gray')
    plt.axvline(150, color='red', linestyle=':', label='Design Frequency (150Hz)')
    plt.yscale('log')
    plt.xlabel('Actual System Frequency (Hz)')
    plt.ylabel('Total Cost (Log Scale)')
    plt.title('Robustness to Frequency Drift: A10 vs Vanilla')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/freq_drift_comparison.png')
    
    # 定量評価
    drift_loss_a10 = np.mean(costs_a10)
    drift_loss_vanilla = np.mean(costs_vanilla)
    
    print(f"\n--- 最終結果 ---")
    print(f"A10 平均ドリフト損失: {drift_loss_a10:.5e}")
    print(f"Vanilla 平均ドリフト損失: {drift_loss_vanilla:.5e}")
    print(f"ロバスト性向上倍率 (Vanilla / A10): {drift_loss_vanilla / drift_loss_a10:.2f}x")
    print("グラフを output/freq_drift_comparison.png に保存しました。")

if __name__ == "__main__":
    run_drift_test()
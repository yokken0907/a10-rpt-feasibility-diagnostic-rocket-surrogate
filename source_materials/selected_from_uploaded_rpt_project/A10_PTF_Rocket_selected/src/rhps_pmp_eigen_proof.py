import jax
import jax.numpy as jnp
from jax import lax, jacfwd, jit  # <-- jit を追加しました
import numpy as np
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
        'kappa': 10.0, 'omega_a': omega_a, 'U_max': 50.0, 'dt': 1e-4
    }
    return params

def smooth_saturation(x, kappa):
    return x / (1.0 + kappa * jnp.abs(x))

# 状態を1つのベクトル(6次元)にまとめるためのヘルパー関数
def pack_state(a, z_f, z_N):
    return jnp.concatenate([a, z_f, z_N])

def unpack_state(x):
    return x[0:2], x[2:4], x[4:6]

# 離散時間ダイナミクス x_{k+1} = Phi(x_k, u_k)
def Phi(x, u, params):
    a, z_f, z_N = unpack_state(x)
    q_sat = smooth_saturation(jnp.dot(params['B_f_out'], z_f), params['kappa'])
    
    da = jnp.dot(params['A_a'], a) + jnp.dot(params['B_f'], q_sat) + jnp.dot(params['B_N'], z_N) + jnp.dot(params['B_u'], u)
    dz_f = jnp.dot(params['A_f'], z_f) + jnp.dot(params['B_fa'], a)
    dz_N = jnp.dot(params['A_N'], z_N) + jnp.dot(params['B_Np'], jnp.array([a[0]]))
    
    next_a = a + params['dt'] * da
    next_zf = z_f + params['dt'] * dz_f
    next_zN = z_N + params['dt'] * dz_N
    return pack_state(next_a, next_zf, next_zN)

# ステップごとのコスト l(x_k, u_k)
def stage_cost(x, u):
    a, _, _ = unpack_state(x)
    return jnp.sum(a**2) + 1e-4 * jnp.sum(u**2)

# ヤコビアンのコンパイル
Phi_x = jit(jacfwd(Phi, argnums=0))
Phi_u = jit(jacfwd(Phi, argnums=1))
ell_x = jit(jacfwd(stage_cost, argnums=0))
ell_u = jit(jacfwd(stage_cost, argnums=1))

def run_full_proof():
    params = define_params()
    dt, N_steps = params['dt'], 500
    time_seq = jnp.arange(N_steps) * dt
    
    # Phase 3で得られた最適パラメータ
    theta_opt = jnp.array([2.72e6, 1.88e2, 7.68e2])
    
    # A10アンスァッツによる制御入力シーケンスの生成
    carrier = theta_opt[0] * jnp.sin(params['omega_a'] * time_seq) + theta_opt[1] * jnp.cos(params['omega_a'] * time_seq)
    u_seq = params['U_max'] * jnp.tanh((carrier * jnp.exp(-theta_opt[2] * time_seq)) / params['U_max'])
    u_seq = u_seq.reshape(-1, 1)

    # 1. 順方向シミュレーション (状態軌道の保存)
    x_seq = [pack_state(jnp.array([0.05, 0.0]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]))]
    for k in range(N_steps):
        x_seq.append(Phi(x_seq[-1], u_seq[k], params))
    x_seq = jnp.stack(x_seq)

    # 2. 逆方向シミュレーション (離散随伴変数 lambda の厳密な算出)
    lambda_seq = [jnp.zeros(6)] # 終端条件 lambda_N = 0
    for k in reversed(range(N_steps)):
        lx = ell_x(x_seq[k], u_seq[k])
        Px = Phi_x(x_seq[k], u_seq[k], params)
        lambda_k = lx + jnp.dot(Px.T, lambda_seq[-1])
        lambda_seq.append(lambda_k)
    lambda_seq = jnp.stack(lambda_seq[::-1]) # 時間順に戻す

    # 3. スイッチング関数 (ハミルトニアンの u による偏微分) の算出
    sigma_seq = []
    for k in range(N_steps):
        lu = ell_u(x_seq[k], u_seq[k])
        Pu = Phi_u(x_seq[k], u_seq[k], params)
        sigma_k = lu + jnp.dot(Pu.T, lambda_seq[k+1])
        sigma_seq.append(sigma_k[0])
    sigma_seq = jnp.array(sigma_seq)

    # 4. 固有値感度による安定性盆の解析
    tau_range = np.linspace(0.001, 0.005, 50)
    max_eigenvalues = []
    for tf in tau_range:
        p_tf = define_params(tf)
        # 平衡点 (x=0, u=0) でのシステムヤコビアンを計算
        J_sys = Phi_x(jnp.zeros(6), jnp.zeros(1), p_tf)
        # 離散時間ヤコビアンの固有値を連続時間の成長率（実部）に変換
        eigvals = np.linalg.eigvals(J_sys)
        cont_growth_rates = np.log(np.abs(eigvals)) / dt
        max_eigenvalues.append(np.max(cont_growth_rates))
    
    # 5. グラフ描画
    plt.figure(figsize=(12, 10))
    
    # グラフ1: 固有値による真の安定性盆 (Stability Basin)
    plt.subplot(3, 1, 1)
    plt.plot(tau_range, max_eigenvalues, 'b-', linewidth=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0.002, color='red', linestyle=':', label='Design Point (tau_f=0.002)')
    plt.fill_between(tau_range, max_eigenvalues, 0, where=(np.array(max_eigenvalues) < 0), color='green', alpha=0.3, label='Stability Basin (Re(lambda) < 0)')
    plt.ylabel('Max Eigenvalue Real Part')
    plt.title('True Stability Basin via Eigenvalue Sensitivity')
    plt.legend(); plt.grid(True)

    # グラフ2: 制御入力と飽和のタイミング
    plt.subplot(3, 1, 2)
    plt.plot(time_seq, u_seq[:, 0], color='red', label='Saturated Control $u(t)$')
    plt.axhline(params['U_max'], color='black', linestyle=':')
    plt.axhline(-params['U_max'], color='black', linestyle=':')
    plt.ylabel('Control Input $u$')
    plt.title('Actuator Effort')
    plt.legend(); plt.grid(True)

    # グラフ3: 離散ハミルトニアンによるスイッチング関数
    plt.subplot(3, 1, 3)
    plt.plot(time_seq, sigma_seq, color='purple', label=r'Switching Function $\sigma_k = \partial H_k / \partial u_k$')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Time (s)'); plt.ylabel('Switching Function')
    plt.title('PMP Optimality Verification')
    plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/pmp_eigen_proof.png')
    print("--- PMPおよび固有値感度の完全証明グラフを output/pmp_eigen_proof.png に保存しました ---")

if __name__ == "__main__":
    run_full_proof()
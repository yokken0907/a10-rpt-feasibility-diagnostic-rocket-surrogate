import jax
import jax.numpy as jnp
from jax import lax

# 倍精度演算の強制（Taylor testの厳密な検証に必須）
jax.config.update("jax_enable_x64", True)

def define_rhps_parameters():
    """
    RHPSの系を構成する線形行列および非線形パラメータを定義する。
    本プロトタイプでは、各状態を2次元(N_a=1, N_f=2, N_N=2)とする。
    """
    # 音響モードダイナミクス (減衰振動系)
    omega_a = 2.0 * jnp.pi * 150.0  # 150 Hz
    zeta_a  = 0.05
    A_a = jnp.array([[0.0, 1.0],
                     [-omega_a**2, -2.0*zeta_a*omega_a]])
    
    # 火炎内部状態 (2次遅れ系を想定)
    tau_f = 0.002
    A_f = jnp.array([[0.0, 1.0],
                     [-1.0/tau_f**2, -2.0/tau_f]])
    B_fa = jnp.array([[0.0, 0.0],
                      [1.0, 0.0]]) # 圧力摂動(a[0])に結合
    B_f_out = jnp.array([[1.0, 0.0],
                         [0.0, 0.0]]) # 熱放出変動への出力行列
    B_f = jnp.array([[0.0, 0.0],
                     [0.0, 5000.0]]) # 熱放出が音響を駆動
    
    # ノズルメモリ状態 (安定なインピーダンス境界)
    tau_N = 0.001
    A_N = jnp.array([[-1.0/tau_N, 0.0],
                     [0.0, -1.0/(tau_N*1.5)]])
    B_Np = jnp.array([[1.0], [1.0]])
    B_N = jnp.array([[0.0, 0.0],
                     [-100.0, -50.0]]) # 境界からの反射
    
    # 制御入力行列
    B_u = jnp.array([[0.0], [1.0]]) # 例えば音響系へ直接介入

    params = {
        'A_a': A_a, 'B_f': B_f, 'B_N': B_N, 'B_u': B_u,
        'A_f': A_f, 'B_fa': B_fa, 'B_f_out': B_f_out,
        'A_N': A_N, 'B_Np': B_Np,
        'kappa': 10.0  # 非線形飽和のスケール
    }
    return params

def smooth_saturation(x, kappa):
    """
    可微分な非線形飽和関数（Softsign関数）。
    ハードな最大値制約を避け、随伴勾配の破綻を防ぐ。
    """
    return x / (1.0 + kappa * jnp.abs(x))

def rhps_step(state, u, params, dt):
    """
    1ステップのオイラー積分（またはルンゲクッタへの拡張用）。
    """
    a, z_f, z_N = state
    
    # 火炎からの熱放出（非線形飽和を含む）
    q_prime = jnp.dot(params['B_f_out'], z_f)
    q_prime_sat = smooth_saturation(q_prime, params['kappa'])
    
    # ノズルからのフィードバック
    p_prime = jnp.array([a[0]])
    
    # 時間発展方程式
    da = jnp.dot(params['A_a'], a) + jnp.dot(params['B_f'], q_prime_sat) + jnp.dot(params['B_N'], z_N) + jnp.dot(params['B_u'], u)
    dz_f = jnp.dot(params['A_f'], z_f) + jnp.dot(params['B_fa'], a)
    dz_N = jnp.dot(params['A_N'], z_N) + jnp.dot(params['B_Np'], p_prime)
    
    next_a = a + dt * da
    next_z_f = z_f + dt * dz_f
    next_z_N = z_N + dt * dz_N
    
    return (next_a, next_z_f, next_z_N)

def objective_function(u_seq, init_state, params, dt):
    """
    制御入力系列 u_seq に対する系全体のコスト（軌道積分）を計算する。
    """
    def scan_fn(state, u):
        next_state = rhps_step(state, u, params, dt)
        a, _, _ = next_state
        # 音響エネルギー（圧力変動と速度変動の二乗和）の抑制を目的とする
        cost = jnp.sum(a**2) + 0.1 * jnp.sum(u**2)
        return next_state, cost
    
    final_state, costs = lax.scan(scan_fn, init_state, u_seq)
    total_cost = jnp.sum(costs) * dt
    return total_cost

def run_taylor_test():
    """
    離散随伴勾配の正確性を証明するTaylor Test。
    | J(u + alpha*du) - J(u) - alpha * <grad_J, du> | が O(alpha^2) に従うか確認する。
    """
    params = define_rhps_parameters()
    dt = 1e-4
    N_steps = 1000
    
    # 初期状態と制御入力の初期化
    init_state = (jnp.array([1.0, 0.0]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]))
    u_seq = jnp.zeros((N_steps, 1))
    
    # ランダムな摂動方向 du を生成
    key = jax.random.PRNGKey(42)
    du_seq = jax.random.normal(key, u_seq.shape)
    
    # 1. 基準となるコスト J(u) の計算
    J0 = objective_function(u_seq, init_state, params, dt)
    
    # 2. 自動微分（離散随伴法）による勾配 grad_J の計算
    grad_fn = jax.grad(objective_function, argnums=0)
    grad_J = grad_fn(u_seq, init_state, params, dt)
    
    # 方向微分 <grad_J, du>
    directional_deriv = jnp.sum(grad_J * du_seq)
    
    # 3. 様々な alpha に対する残差の評価
    alphas = jnp.logspace(-2, -8, 7)
    
    print("--- Taylor Test Results ---")
    print(f"{'alpha':<12} | {'Residual':<15} | {'Expected O(alpha^2) ratio'}")
    
    prev_residual = None
    for alpha in alphas:
        J_perturbed = objective_function(u_seq + alpha * du_seq, init_state, params, dt)
        # テイラー展開の1次近似との残差
        residual = jnp.abs(J_perturbed - J0 - alpha * directional_deriv)
        
        ratio = (prev_residual / residual) if prev_residual is not None else 0.0
        print(f"{alpha:.1e}    | {residual:.5e}     | {ratio:.2f} (Target ~ 100.0)")
        prev_residual = residual

# 実行
run_taylor_test()
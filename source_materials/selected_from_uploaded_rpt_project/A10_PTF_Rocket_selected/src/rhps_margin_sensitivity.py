import jax
import jax.numpy as jnp
from jax import lax, jacfwd, jit
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

def pack_state(a, z_f, z_N):
    return jnp.concatenate([a, z_f, z_N])

def unpack_state(x):
    return x[0:2], x[2:4], x[4:6]

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

def stage_cost(x, u):
    a, _, _ = unpack_state(x)
    return jnp.sum(a**2) + 1e-4 * jnp.sum(u**2)

Phi_x = jit(jacfwd(Phi, argnums=0))
Phi_u = jit(jacfwd(Phi, argnums=1))
ell_x = jit(jacfwd(stage_cost, argnums=0))
ell_u = jit(jacfwd(stage_cost, argnums=1))

def get_max_re_eig(tf, dt):
    p_tf = define_params(tf)
    J_sys = Phi_x(jnp.zeros(6), jnp.zeros(1), p_tf)
    eigvals = np.linalg.eigvals(J_sys)
    cont_growth_rates = np.log(np.abs(eigvals)) / dt
    return np.max(cont_growth_rates)

def run_margin_analysis():
    params = define_params()
    dt, N_steps = params['dt'], 500
    time_seq = jnp.arange(N_steps) * dt
    
    theta_opt = jnp.array([2.72e6, 1.88e2, 7.68e2])
    
    carrier = theta_opt[0] * jnp.sin(params['omega_a'] * time_seq) + theta_opt[1] * jnp.cos(params['omega_a'] * time_seq)
    u_seq = params['U_max'] * jnp.tanh((carrier * jnp.exp(-theta_opt[2] * time_seq)) / params['U_max'])
    u_seq = u_seq.reshape(-1, 1)

    x_seq = [pack_state(jnp.array([0.05, 0.0]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]))]
    for k in range(N_steps):
        x_seq.append(Phi(x_seq[-1], u_seq[k], params))
    x_seq = jnp.stack(x_seq)

    lambda_seq = [jnp.zeros(6)]
    for k in reversed(range(N_steps)):
        lx = ell_x(x_seq[k], u_seq[k])
        Px = Phi_x(x_seq[k], u_seq[k], params)
        lambda_k = lx + jnp.dot(Px.T, lambda_seq[-1])
        lambda_seq.append(lambda_k)
    lambda_seq = jnp.stack(lambda_seq[::-1])

    sigma_seq = []
    for k in range(N_steps):
        lu = ell_u(x_seq[k], u_seq[k])
        Pu = Phi_u(x_seq[k], u_seq[k], params)
        sigma_k = lu + jnp.dot(Pu.T, lambda_seq[k+1])
        sigma_seq.append(sigma_k[0])
    sigma_seq = jnp.array(sigma_seq)

    # 1. スイッチングマージンの計算 (飽和区間における |sigma_k| の最小値)
    sat_indices = np.where(np.abs(u_seq[:, 0]) > 0.99 * params['U_max'])[0]
    if len(sat_indices) > 0:
        m_sigma = np.min(np.abs(sigma_seq[sat_indices]))
    else:
        m_sigma = 0.0
    
    print(f"--- 定量評価指標 ---")
    print(f"Switching Margin (m_sigma): {m_sigma:.5e}")

    # 2. 固有値感度の計算 (有限差分法)
    tau_range = np.linspace(0.001, 0.005, 50)
    dtau = 1e-6
    max_eigenvalues = []
    eigen_sensitivities = []
    
    for tf in tau_range:
        eig_center = get_max_re_eig(tf, dt)
        eig_plus = get_max_re_eig(tf + dtau, dt)
        eig_minus = get_max_re_eig(tf - dtau, dt)
        
        max_eigenvalues.append(eig_center)
        # 中心差分による微分 d(Re lambda) / d(tau_f)
        sensitivity = (eig_plus - eig_minus) / (2 * dtau)
        eigen_sensitivities.append(sensitivity)
        
    print(f"Max Eigenvalue Sensitivity: {np.max(np.abs(eigen_sensitivities)):.5e}")

    # グラフ描画
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Flame Delay $\\tau_f$ (s)')
    ax1.set_ylabel('Dominant Eigenvalue $\\Re(\\lambda_{\\max})$', color=color)
    ax1.plot(tau_range, max_eigenvalues, color=color, linewidth=2, label='$\\Re(\\lambda_{\\max})$')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(0, color='black', linestyle='--')
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Sensitivity $d\\Re(\\lambda_{\\max})/d\\tau_f$', color=color)
    ax2.plot(tau_range, eigen_sensitivities, color=color, linestyle='-.', label='Sensitivity')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Spectral Stability and Sensitivity over Delay Interval')
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/spectral_sensitivity.png')
    print("--- グラフを output/spectral_sensitivity.png に保存しました ---")

if __name__ == "__main__":
    run_margin_analysis()
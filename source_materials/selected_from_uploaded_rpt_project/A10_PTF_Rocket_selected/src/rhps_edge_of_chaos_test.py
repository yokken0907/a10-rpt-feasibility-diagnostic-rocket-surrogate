import jax
import jax.numpy as jnp
from jax import lax, jacfwd, jit, value_and_grad
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

jax.config.update("jax_enable_x64", True)

def define_params_unstable(tau_f=0.002):
    omega_a = 2.0 * jnp.pi * 150.0
    return {
        'A_a': jnp.array([[0.0, 1.0], [-omega_a**2, -2.0*0.01*omega_a]]), # 減衰を0.05→0.01へ（より不安定に）
        'A_f': jnp.array([[0.0, 1.0], [-1.0/tau_f**2, -2.0/tau_f]]),
        'B_f': jnp.array([[0.0, 0.0], [0.0, 8000.0]]), # 火炎ゲインを5000→8000へ（発散しやすく）
        'B_u': jnp.array([0.0, 100.0]),
        'B_fa': jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        'B_f_out': jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        'A_N': jnp.array([[-1000.0, 0.0], [0.0, -666.0]]),
        'B_Np': jnp.array([[1.0], [1.0]]),
        'B_N': jnp.array([[0.0, 0.0], [-100.0, -50.0]]),
        'kappa': 10.0, 'omega_a': omega_a, 'U_max': 50.0, 'dt': 1e-4,
        'noise_std': 2000.0 # 強力なノイズを継続注入
    }

def Phi_noisy(x, u, noise, params):
    a, z_f, z_N = x[0:2], x[2:4], x[4:6]
    q_sat = (jnp.dot(params['B_f_out'], z_f)) / (1.0 + params['kappa'] * jnp.abs(jnp.dot(params['B_f_out'], z_f)))
    
    # 乱流ノイズを常に加速度項へ注入
    da = jnp.dot(params['A_a'], a) + jnp.dot(params['B_f'], q_sat) + jnp.dot(params['B_N'], z_N) + (params['B_u'] * u) + jnp.array([0.0, noise])
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

def run_noisy_simulation(theta, u_func, tau_f, noise_seq):
    params = define_params_unstable(tau_f)
    dt, N_steps = params['dt'], 500
    time_seq = jnp.arange(N_steps) * dt
    
    def scan_fn(state, carry):
        t, noise = carry
        u = u_func(theta, t, params)
        next_state = Phi_noisy(state, u, noise, params)
        return next_state, jnp.sum(next_state[0:2]**2)
    
    init_state = jnp.concatenate([jnp.array([0.05, 0.0]), jnp.zeros(4)])
    _, costs = lax.scan(scan_fn, init_state, (time_seq, noise_seq))
    return jnp.sum(costs) * dt

def run_comparison():
    print("--- 限界領域（Edge of Chaos）でのロバスト性比較 ---")
    key = jax.random.PRNGKey(123)
    noise_seq = jax.random.normal(key, (500,)) * 2000.0
    
    th_init = np.array([1.0, 1.0, 1.0])
    
    # 設計点 (0.002) での最適化
    print("Optimizing...")
    obj_a10 = jit(value_and_grad(lambda th: run_noisy_simulation(th, u_a10, 0.002, noise_seq)))
    res_a10 = minimize(lambda th: tuple(np.array(v) for v in obj_a10(jnp.array(th))), th_init, method='L-BFGS-B', jac=True)

    obj_vanilla = jit(value_and_grad(lambda th: run_noisy_simulation(th, u_vanilla, 0.002, noise_seq)))
    res_vanilla = minimize(lambda th: tuple(np.array(v) for v in obj_vanilla(jnp.array(th))), th_init, method='L-BFGS-B', jac=True)

    # 広い範囲で tau_f をスキャン
    tau_range = np.linspace(0.0005, 0.0045, 15)
    costs_a10 = [run_noisy_simulation(jnp.array(res_a10.x), u_a10, tf, noise_seq) for tf in tau_range]
    costs_vanilla = [run_noisy_simulation(jnp.array(res_vanilla.x), u_vanilla, tf, noise_seq) for tf in tau_range]

    plt.figure(figsize=(10, 6))
    plt.plot(tau_range, costs_a10, 'o-', label='A10-Ansatz', color='blue')
    plt.plot(tau_range, costs_vanilla, 's--', label='Vanilla-Policy', color='gray')
    plt.yscale('log') # コスト差が激しいため対数表示
    plt.xlabel(r'Flame Delay $\tau_f$')
    plt.ylabel('Total Cost (Log Scale)')
    plt.title('Survival Test: A10 vs Vanilla at Edge of Chaos')
    plt.legend(); plt.grid(True)
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/edge_of_chaos_comparison.png')
    
    print(f"\n[結果]")
    print(f"A10 平均感度: {np.mean(np.abs(np.diff(costs_a10))):.2e}")
    print(f"Vanilla 平均感度: {np.mean(np.abs(np.diff(costs_vanilla))):.2e}")

if __name__ == "__main__":
    run_comparison()
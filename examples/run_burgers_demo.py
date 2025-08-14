# examples/run_burgers_demo.py
import os, sys
# Add the PROJECT ROOT (parent of 'src/') to sys.path so `import src...` works.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np

from src.python.finite_difference_solver import solve_burgers
from src.python.visualization import plot_solution_heatmap

# --- 1) Parameters (small but presentable) ---
nx = 256
nt = 500
L = 2.0
T = 1.0
nu = 1e-2
u0 = lambda x: np.sin(2 * np.pi * x / L)  # smooth sine IC

# --- 2) Solve PDE ---
x, t, U = solve_burgers(nx=nx, nt=nt, L=L, T=T, nu=nu, u0=u0)

# --- 3) Ensure examples/ exists for outputs ---
os.makedirs("examples", exist_ok=True)

# --- 4) Save a space–time heatmap ---
heatmap_path = "examples/burgers_heatmap.png"
plot_solution_heatmap(U, x, t, title="Burgers' Equation (Lax–Friedrichs, periodic)", save_path=heatmap_path)

# --- 5) Also save an initial vs final snapshot line plot (quick visual) ---
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.plot(x, U[0], label="t = 0", linewidth=2)
ax.plot(x, U[-1], label=f"t = {T:.2f}", linewidth=2)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Initial vs Final Snapshot")
ax.grid(True)
ax.legend()
fig.tight_layout()
snap_path = "examples/burgers_snapshots.png"
fig.savefig(snap_path, dpi=150)
plt.close(fig)

print(f"Saved:\n  {heatmap_path}\n  {snap_path}")

"""Figure for the report: ground normal force vs. penetration depth.

Run from the repo root:  python3 Reports/figures/make_ground_figure.py
Uses the real normal_force() from Gym/stance_env/ground.py (static case, no damping).
"""
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
# Load ground.py directly (it only needs numpy) so MuJoCo/gymnasium aren't required.
spec = importlib.util.spec_from_file_location("ground", ROOT / "Gym" / "stance_env" / "ground.py")
ground = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ground)
normal_force = ground.normal_force

K0, F_YIELD = 3e4, 1000.0      # representative train-range ground
depth = np.linspace(0, 0.06, 400)

fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)
for alpha, color in [(0.0, "#1f77b4"), (2.0, "#d95f02")]:
    f = [normal_force(d, 0.0, K0, 0.0, alpha, F_YIELD)[0] for d in depth]
    ax.plot(depth * 1000, f, color=color, lw=1.5,
            label=rf"$\alpha$ = {alpha:g}" + (" (linear)" if alpha == 0 else " (stiffening)"))

ax.axhline(F_YIELD, color="gray", ls="--", lw=0.8)
ax.text(1, F_YIELD + 40, r"$f_{yield}$: force saturates, surface dents", fontsize=6.5, color="gray")
ax.set_xlabel("Penetration depth (mm)", fontsize=8)
ax.set_ylabel("Normal force (N)", fontsize=8)
ax.set_title(r"Ground force law ($k_0$ = 30 kN/m, static)", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.set_xlim(0, 60)
ax.set_ylim(0, 1250)
fig.tight_layout()
out = Path(__file__).with_name("ground_force_law.png")
fig.savefig(out)
print("saved", out)

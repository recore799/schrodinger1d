import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import conj
import math

import sys
import os
# Add src directory to path (not needed when files are in the same dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from helpers import harmonic_oscillator_wf

# ---------------------------
# User settings
# ---------------------------
n_states_to_load = 6      # how many eigenstates to load (0..n_states_to_load-1)
fps = 30
duration = 6.0            # seconds
hbar = 1.0                # if your energies are in hbar*omega units already set accordingly

# ---------------------------
# Load states
# ---------------------------
x_ref = None
psi_n_list = []
E_list = []

for n in range(n_states_to_load):
    x, psi, e = harmonic_oscillator_wf(state=n)   # uses your provided function
    # convert to numpy arrays (defensive)
    x = np.asarray(x)
    psi = np.asarray(psi, dtype=float)

    # ensure we have the same x grid for all states
    if x_ref is None:
        x_ref = x.copy()
    else:
        if not np.allclose(x_ref, x):
            raise RuntimeError("x grid mismatch between states; ensure same mesh/xmax used each call")

    # numeric normalization on full domain (just in case)
    norm = np.sqrt(np.trapz(psi**2, x))
    if norm == 0:
        raise RuntimeError(f"state {n} normalization zero")
    psi /= norm

    psi_n_list.append(psi)
    E_list.append(e)

psi_n_list = np.array(psi_n_list)   # shape (n_states, nx)
E_list = np.array(E_list)

print(f"Loaded {len(psi_n_list)} states, grid length = {x_ref.size}")

# optional: upsample / smooth the grid (uses scipy if available)
upsample_factor = 1   # set to e.g. 3 or 5 to get smoother curves; 1 = keep original grid
if upsample_factor > 1:
    try:
        from scipy.interpolate import InterpolatedUnivariateSpline
        x_dense = np.linspace(x_ref.min(), x_ref.max(), upsample_factor * x_ref.size)
        psi_n_dense = []
        for psi in psi_n_list:
            s = InterpolatedUnivariateSpline(x_ref, psi, k=3)
            psi_n_dense.append(s(x_dense))
        psi_n_list = np.array(psi_n_dense)
        x_ref = x_dense
        print("Used scipy to upsample -> new grid length:", x_ref.size)
    except Exception as exc:
        print("scipy not available or upsample failed:", exc, " -> continuing without upsampling")

# ---------------------------
# Choose coefficients for superposition
# ---------------------------
# Example 1: two-state equal superposition using states 0 and 1
c = np.zeros(len(psi_n_list), dtype=complex)
c[0] = 1/np.sqrt(2)
if len(psi_n_list) > 1:
    c[1] = 1/np.sqrt(2)

# Example 2: coherent-like state (uncomment to use)
# alpha = 2.0
# maxn = len(psi_n_list)
# c = np.array([np.exp(-abs(alpha)**2/2) * alpha**n / math.sqrt(math.factorial(n)) for n in range(maxn)], dtype=complex)

# normalize coefficients
c = c / np.linalg.norm(c)

# ---------------------------
# Precompute time frames
# ---------------------------
nframes = int(fps * duration)
t_array = np.linspace(0, duration, nframes, endpoint=False)

nx = x_ref.size
n_states = psi_n_list.shape[0]

# convert psi_n_list to shape (n_states, nx)
psi_n = psi_n_list

Psi_t = np.zeros((nframes, nx), dtype=complex)
for i, t in enumerate(t_array):
    phases = np.exp(-1j * E_list * t / hbar)[:, None]   # (n_states,1)
    Psi = np.sum((c[:, None] * phases) * psi_n, axis=0)  # (nx,)
    Psi_t[i] = Psi

density = np.abs(Psi_t)**2
realpart = Psi_t.real

# ---------------------------
# Plot + animate
# ---------------------------
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax0, ax1 = ax
line_density, = ax0.plot(x_ref, density[0], lw=2)
ax0.set_ylabel(r"$|\Psi(x,t)|^2$")
line_real, = ax1.plot(x_ref, realpart[0], lw=2)
ax1.set_ylabel(r"$\Re\Psi(x,t)$")
ax1.set_xlabel("x")

# autoscale y-limits with small margin
ax0.set_ylim(0, density.max() * 1.05)
rp_min, rp_max = realpart.min(), realpart.max()
ax1.set_ylim(rp_min * 1.05, rp_max * 1.05)

time_text = ax0.text(0.02, 0.9, "", transform=ax0.transAxes)

def update(k):
    line_density.set_ydata(density[k])
    line_real.set_ydata(realpart[k])
    time_text.set_text(f"t = {t_array[k]:.3f}")
    return line_density, line_real, time_text

anim = FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=True)

# save the movie (optional)
save_filename = "ho_superposition.mp4"
anim.save(save_filename, dpi=150, fps=fps)
print("Saved animation to", save_filename)

plt.show()

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

# ho_vertical_travel.py
import matplotlib.gridspec as gridspec

# -----------------------
# Settings
# -----------------------
n_states = 7             # number of eigenstates to load (0..n_states-1)
fps = 30
duration = 6.0           # seconds for the full travel top->bottom
hbar = 1.0               # units for time-evolution (match your energy units)
upsample_factor = 1      # set >1 if you want smoother lines (requires scipy)

# Vertical travel extents in the right panel (data coords)
vertical_top = 2.5       # y-value where the waveform starts (top)
vertical_bottom = -2.5   # y-value where the waveform ends (bottom)

# -----------------------
# Load eigenstates
# -----------------------
x_ref = None
psi_n_list = []
E_list = []

for n in range(n_states):
    x, psi, e = harmonic_oscillator_wf(state=n)
    x = np.asarray(x)
    psi = np.asarray(psi, dtype=float)

    if x_ref is None:
        x_ref = x.copy()
    else:
        if not np.allclose(x_ref, x):
            raise RuntimeError("x grid mismatch across states. Ensure harmonic_oscillator_wf uses same mesh/xmax each call.")

    # ensure psi normalized on full domain
    norm = np.sqrt(np.trapezoid(psi**2, x))
    if norm == 0:
        raise RuntimeError(f"state {n} has zero norm")
    psi /= norm

    psi_n_list.append(psi)
    E_list.append(e)

psi_n_list = np.array(psi_n_list)   # shape (n_states, nx)
E_list = np.array(E_list)
nx = x_ref.size

# optional upsampling for smoother curves
if upsample_factor > 1:
    try:
        from scipy.interpolate import InterpolatedUnivariateSpline
        x_dense = np.linspace(x_ref.min(), x_ref.max(), upsample_factor * nx)
        psi_dense = []
        for psi in psi_n_list:
            s = InterpolatedUnivariateSpline(x_ref, psi, k=3)
            psi_dense.append(s(x_dense))
        psi_n_list = np.array(psi_dense)
        x_ref = x_dense
        nx = x_ref.size
        print("Upsampled grid length:", nx)
    except Exception as exc:
        print("Upsample skipped (scipy missing or error):", exc)

print(f"Loaded {n_states} states, grid length = {nx}")

# -----------------------
# Choose coefficients for the superposition (coherent-like -> localized packet)
# -----------------------
alpha = 2.0
c = np.array([np.exp(-abs(alpha)**2/2) * alpha**n / math.sqrt(math.factorial(n)) for n in range(n_states)], dtype=complex)
c /= np.linalg.norm(c)   # normalize coefficients

# If you prefer a simple two-state superposition uncomment:
# c = np.zeros(n_states, dtype=complex)
# c[0] = 1/np.sqrt(2); c[1] = 1/np.sqrt(2)

# -----------------------
# Precompute time frames of the superposition
# -----------------------
nframes = int(fps * duration)
t_array = np.linspace(0, duration, nframes, endpoint=False)

psi_n = psi_n_list  # (n_states, nx)
Psi_t = np.zeros((nframes, nx), dtype=complex)

for i, t in enumerate(t_array):
    phases = np.exp(-1j * E_list * t / hbar)[:, None]   # (n_states,1)
    Psi = np.sum((c[:, None] * phases) * psi_n, axis=0)  # (nx,)
    Psi_t[i] = Psi

density_t = np.abs(Psi_t)**2
real_t = Psi_t.real

# -----------------------
# Build Matplotlib figure: left column stacked states, right column large panel
# -----------------------
fig = plt.figure(figsize=(8.5, 9))
gs = gridspec.GridSpec(n_states, 2, width_ratios=[1, 2], height_ratios=[1]*n_states, wspace=0.12, hspace=0.6)

# Left: one small axis per state (plot real(psi) or psi itself)
left_axes = []
for i in range(n_states):
    ax = fig.add_subplot(gs[i, 0])
    left_axes.append(ax)
    psi = psi_n_list[i]
    ax.plot(x_ref, psi, lw=1.5)
    ax.set_ylabel(f"n={i}")
    ax.set_xticks([])
    # unify x-limits
    ax.set_xlim(x_ref.min(), x_ref.max())

# adjust y-limits so they are comparable across states
ymin = psi_n_list.min()
ymax = psi_n_list.max()
for ax in left_axes:
    ax.set_ylim(ymin*1.05, ymax*1.05)

# Right: merged axis spanning all rows
ax_right = fig.add_subplot(gs[:, 1])
ax_right.set_xlim(x_ref.min(), x_ref.max())
ax_right.set_ylim(vertical_bottom, vertical_top)
ax_right.set_xlabel("x")
ax_right.set_ylabel("vertical position")
title = ax_right.set_title("Superposition")

# Initialize the waveform line on the right panel
# We'll draw Re(Ψ) and a translucent filled density below it for visual clarity
line_re, = ax_right.plot(x_ref, real_t[0] + vertical_top, lw=2)
fill_density = ax_right.fill_between(x_ref, vertical_top, np.abs(Psi_t[0])**2 + vertical_top, alpha=0.25)

time_text = ax_right.text(0.02, 0.95, "", transform=ax_right.transAxes)

# Precompute vertical offsets for frames (linear travel from top->bottom)
offsets = np.linspace(vertical_top, vertical_bottom, nframes)

# Update function for animation
def update(k):
    y_offset = offsets[k]
    # Real part shifted vertically
    ydata = real_t[k] + y_offset
    line_re.set_ydata(ydata)

    # update filled density region: remove and redraw (simpler approach)
    global fill_density
    try:
        fill_density.remove()
    except Exception:
        pass
    fill_density = ax_right.fill_between(x_ref, y_offset, np.abs(Psi_t[k])**2 + y_offset, alpha=0.25)

    time_text.set_text(f"t = {t_array[k]:.3f}")
    return (line_re, fill_density, time_text)

anim = FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=False)

# Save output (optional; comment out if you only want interactive)
outname = "ho_vertical_travel.mp4"
print("Rendering animation (this may take a moment)...")


# prefer this DPI (what you had)
desired_dpi = 150

# compute pixel dims and bump DPI until both dims are even integers
fig_inches = fig.get_size_inches()
dpi_try = int(desired_dpi)
def pixel_even(dpi_val):
    w = int(round(fig_inches[0] * dpi_val))
    h = int(round(fig_inches[1] * dpi_val))
    return (w % 2 == 0) and (h % 2 == 0), w, h

is_even, w_px, h_px = pixel_even(dpi_try)
while not is_even:
    dpi_try += 1
    is_even, w_px, h_px = pixel_even(dpi_try)

print(f"Saving with dpi={dpi_try} -> size = {w_px}x{h_px} (even)")

anim.save(outname, dpi=dpi_try, fps=fps)


print("Saved animation to", outname)

plt.show()

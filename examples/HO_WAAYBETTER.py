# ho_vertical_with_phase_bg.py
"""
Vertical-wave animation with phase-background behind each pure state (left column).
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


import sys
import os
# Add src directory to path (not needed when files are in the same dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from helpers import harmonic_oscillator_wf



# -----------------------
# User settings
# -----------------------
n_states = 7
fps = 30
duration = 6.0
hbar = 1.0
upsample_factor = 1
coherent_alpha = 2.0

# Phase background visibility (0 = invisible, 1 = fully opaque)
phase_alpha = 0.32

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

    norm = np.sqrt(np.trapezoid(psi**2, x))
    if norm == 0:
        raise RuntimeError(f"state {n} has zero norm")
    psi /= norm

    psi_n_list.append(psi)
    E_list.append(e)

psi_n_list = np.array(psi_n_list)
E_list = np.array(E_list)
nx = x_ref.size
print(f"Loaded {n_states} states, grid length = {nx}")

# optional upsample
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

# -----------------------
# Superposition coefficients
# -----------------------
c = np.array([np.exp(-abs(coherent_alpha)**2/2) * coherent_alpha**n / math.sqrt(math.factorial(n))
              for n in range(n_states)], dtype=complex)
c /= np.linalg.norm(c)

# -----------------------
# Precompute time evolution
# -----------------------
nframes = int(fps * duration)
t_array = np.linspace(0, duration, nframes, endpoint=False)

psi_n = psi_n_list
phases = np.exp(-1j * np.outer(E_list, t_array) / hbar)
psi_n_t_complex = psi_n[:, None, :] * phases[:, :, None]   # (n_states, nframes, nx)
psi_n_real_t = psi_n_t_complex.real

Psi_t = np.sum((c[:, None, None] * psi_n_t_complex), axis=0)  # (nframes, nx)
Psi_density_t = np.abs(Psi_t)**2
Psi_real_t = Psi_t.real

# -----------------------
# Plot ranges and figure
# -----------------------
max_amp_left = np.max(np.abs(psi_n_real_t))
xlim_left = (-1.05 * max_amp_left, 1.05 * max_amp_left)
max_density = np.max(Psi_density_t)
xlim_right = (-0.02*max_density, 1.05 * max_density)
ymin, ymax = x_ref.min(), x_ref.max()

fig = plt.figure(figsize=(7, 11))
gs = gridspec.GridSpec(n_states, 2, width_ratios=[1, 2], height_ratios=[1]*n_states,
                       left=0.07, right=0.95, top=0.96, bottom=0.03, hspace=0.6, wspace=0.12)

left_axes = []
left_lines = []
left_phase_images = []   # AxesImage objects for dynamic phase backgrounds

# We'll create a narrow 2D array for the phase image: shape (ny, ncols)
# ncols can be small since extent stretches it; choose 4 for quality
ncols_img = 4

for i in range(n_states):
    ax = fig.add_subplot(gs[i, 0])
    left_axes.append(ax)

    # initial line (amplitude vs vertical spatial axis)
    line, = ax.plot(psi_n_real_t[i, 0, :], x_ref, lw=1.5, color='white', zorder=3)
    left_lines.append(line)
    ax.set_xlim(xlim_left)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.yaxis.set_label_position("left")
    ax.set_ylabel(f"n={i}", rotation=0, labelpad=18, va="center")
    ax.grid(False)

    # build initial phase image from the phase at frame 0
    phase0 = np.angle(psi_n_t_complex[i, 0, :])  # shape (nx,)
    # create image data shape (nx, ncols_img)
    img_data = np.tile(phase0[:, None], (1, ncols_img))

    # imshow with extent = (xleft, xright, ymin, ymax) so the image fills the subplot
    img = ax.imshow(img_data,
                    cmap='hsv',
                    vmin=-np.pi, vmax=np.pi,
                    aspect='auto',
                    extent=(xlim_left[0], xlim_left[1], ymin, ymax),
                    origin='lower',
                    alpha=phase_alpha,
                    zorder=1)
    left_phase_images.append(img)

# Right big axis
ax_right = fig.add_subplot(gs[:, 1])
ax_right.set_xlim(xlim_right)
ax_right.set_ylim(ymin, ymax)
ax_right.set_xlabel("probability density / amplitude")
ax_right.set_ylabel("x (spatial)")
ax_right.set_title("Superposition: |Ψ(x,t)|^2")

line_super_real, = ax_right.plot(Psi_real_t[0], x_ref, lw=1.6, color='k', label="Re(Ψ)", zorder=3)
fill_super = ax_right.fill_betweenx(x_ref, np.zeros_like(x_ref), Psi_density_t[0], alpha=0.28, label="|Ψ|^2", zorder=2)
ax_right.legend(loc="upper right", framealpha=0.9)
time_text = ax_right.text(0.02, 0.95, "", transform=ax_right.transAxes)

# -----------------------
# Update function
# -----------------------
def update(frame):
    # update left lines and phase images
    for i, (line, img) in enumerate(zip(left_lines, left_phase_images)):
        # update amplitude line (real part)
        line.set_xdata(psi_n_real_t[i, frame, :])

        # compute phase array for this state & frame and update image
        phase_arr = np.angle(psi_n_t_complex[i, frame, :])   # length nx
        new_img = np.tile(phase_arr[:, None], (1, ncols_img))  # (nx, ncols_img)
        img.set_data(new_img)

    # update right panel
    line_super_real.set_xdata(Psi_real_t[frame])
    global fill_super
    try:
        fill_super.remove()
    except Exception:
        pass
    fill_super = ax_right.fill_betweenx(x_ref, np.zeros_like(x_ref), Psi_density_t[frame], alpha=0.28)

    time_text.set_text(f"t = {t_array[frame]:.3f}")

    artists = left_lines + [line_super_real, fill_super, time_text] + left_phase_images
    return artists

# -----------------------
# Save helper (ensures even pixel dims)
# -----------------------
def save_animation(anim_obj, filename, desired_dpi=120):
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
    anim_obj.save(filename, dpi=dpi_try, fps=fps)

anim = FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=False)

outname = "ho_vertical_phase_bg.mp4"
print("Rendering animation (this may take a moment)...")
save_animation(anim, outname)
print("Saved animation to", outname)

plt.show()

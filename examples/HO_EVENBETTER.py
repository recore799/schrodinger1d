import sys
import os
# Add src directory to path (not needed when files are in the same dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from helpers import harmonic_oscillator_wf

# ho_vertical_final.py
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# -----------------------
# User-adjustable settings
# -----------------------
n_states = 7              # load states 0..6
fps = 30
duration = 6.0            # seconds for the whole animation
hbar = 1.0                # change if your energies use different units
upsample_factor = 1       # set >1 to smooth (requires scipy)
coherent_alpha = 2.0      # if you want coherent-like coefficients for superposition

# -----------------------
# Load eigenstates
# -----------------------
x_ref = None
psi_n_list = []
E_list = []

for n in range(n_states):
    x, psi, e = harmonic_oscillator_wf(state=n)   # must return x_full, psi_full, e
    x = np.asarray(x)
    psi = np.asarray(psi, dtype=float)

    if x_ref is None:
        x_ref = x.copy()
    else:
        if not np.allclose(x_ref, x):
            raise RuntimeError("x grid mismatch across states. Ensure harmonic_oscillator_wf uses same mesh/xmax each call.")

    # normalize on full domain using trapezoid
    norm = np.sqrt(np.trapezoid(psi**2, x))
    if norm == 0:
        raise RuntimeError(f"state {n} has zero norm")
    psi /= norm

    psi_n_list.append(psi)
    E_list.append(e)

psi_n_list = np.array(psi_n_list)   # shape (n_states, nx)
E_list = np.array(E_list)
nx = x_ref.size
print(f"Loaded {n_states} states, grid length = {nx}")

# optional upsample for smoother curves
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
# Coefficients for superposition (coherent-like)
# -----------------------
c = np.array([np.exp(-abs(coherent_alpha)**2/2) * coherent_alpha**n / math.sqrt(math.factorial(n))
              for n in range(n_states)], dtype=complex)
c /= np.linalg.norm(c)

# alternatively use a simple two-state superposition:
# c = np.zeros(n_states, dtype=complex); c[0]=1/np.sqrt(2); c[1]=1/np.sqrt(2)

# -----------------------
# Precompute time evolution frames
# -----------------------
nframes = int(fps * duration)
t_array = np.linspace(0, duration, nframes, endpoint=False)

# psi_n: (n_states, nx)
psi_n = psi_n_list

# phases array: (n_states, nframes)
phases = np.exp(-1j * np.outer(E_list, t_array) / hbar)

# pure-state time-dependent real parts: shape (n_states, nframes, nx)
psi_n_t_complex = psi_n[:, None, :] * phases[:, :, None]   # shape (n_states, nframes, nx), complex
psi_n_real_t = psi_n_t_complex.real

# superposition complex wave per frame: (nframes, nx)
Psi_t = np.sum((c[:, None, None] * psi_n_t_complex), axis=0)
Psi_density_t = np.abs(Psi_t)**2             # (nframes, nx)
Psi_real_t = Psi_t.real                      # (nframes, nx)

# -----------------------
# Plot ranges & layout (avoid squish)
# -----------------------
# amplitude ranges for left panels (across all states & times)
max_amp_left = np.max(np.abs(psi_n_real_t))
xlim_left = (-1.05 * max_amp_left, 1.05 * max_amp_left)

# right panel density horizontal range
max_density = np.max(Psi_density_t)
xlim_right = (-0.02*max_density, 1.05 * max_density)  # small left margin

# vertical (spatial) limits = x_ref range (same in left & right)
ymin, ymax = x_ref.min(), x_ref.max()

# Build figure with a tall aspect to keep vertical axis readable
fig = plt.figure(figsize=(7, 11))   # adjust width/height if you want different proportions
gs = gridspec.GridSpec(n_states, 2, width_ratios=[1, 2], height_ratios=[1]*n_states,
                       left=0.07, right=0.95, top=0.96, bottom=0.03, hspace=0.6, wspace=0.12)

# Left small stacked axes
left_axes = []
left_lines = []
for i in range(n_states):
    ax = fig.add_subplot(gs[i, 0])
    left_axes.append(ax)
    # initial data: amplitude vs y (vertical spatial)
    line, = ax.plot(psi_n_real_t[i, 0, :], x_ref, lw=1.5)
    left_lines.append(line)
    ax.set_xlim(xlim_left)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([])
    ax.set_xlabel("")   # no label to keep tight
    # label on the left of the subplot
    ax.yaxis.set_label_position("left")
    ax.set_ylabel(f"n={i}", rotation=0, labelpad=18, va="center")
    ax.grid(False)

# Right big axis spanning all rows
ax_right = fig.add_subplot(gs[:, 1])
ax_right.set_xlim(xlim_right)
ax_right.set_ylim(ymin, ymax)
ax_right.set_xlabel("probability density / amplitude")
ax_right.set_ylabel("x (spatial)")
ax_right.set_title("Superposition: |Ψ(x,t)|^2 (travels in y as t evolves)")

# initial right plot: density (horizontal extent) vs y
line_super_real, = ax_right.plot(Psi_real_t[0], x_ref, lw=1.6, label="Re(Ψ)")
fill_super = ax_right.fill_betweenx(x_ref, np.zeros_like(x_ref), Psi_density_t[0], alpha=0.28, label="|Ψ|^2")

# small legend on the right
ax_right.legend(loc="upper right", framealpha=0.9)

time_text = ax_right.text(0.02, 0.95, "", transform=ax_right.transAxes)

# -----------------------
# Animation update function
# -----------------------
def update(frame):
    # update left small panels: each state's real part vs vertical x_ref
    for i, line in enumerate(left_lines):
        line.set_xdata(psi_n_real_t[i, frame, :])

    # update right: density and real part (no forced translation; density will "move" because frame changes)
    line_super_real.set_xdata(Psi_real_t[frame])
    # remove previous fill and draw new
    global fill_super
    try:
        fill_super.remove()
    except Exception:
        pass
    fill_super = ax_right.fill_betweenx(x_ref, np.zeros_like(x_ref), Psi_density_t[frame], alpha=0.28)

    time_text.set_text(f"t = {t_array[frame]:.3f}")
    # return artists for completeness (we're not blitting)
    artists = left_lines + [line_super_real, fill_super, time_text]
    return artists

# -----------------------
# Save helper: ensures even pixel dims for ffmpeg
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

# create animation (blit=False for simplicity)
anim = FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=False)

# Save & show
outname = "ho_vertical_final.mp4"
print("Rendering animation (this may take a moment)...")
save_animation(anim, outname)
print("Saved animation to", outname)

plt.show()

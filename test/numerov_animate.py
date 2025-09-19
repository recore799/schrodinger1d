import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Your Numerov solver (slightly adapted to return psi and x) ---

def outward_integration(psi, f, f_10, icl):
    ncross = 0
    for i in range(1, icl):
        psi[i+1] = (f_10[i] * psi[i] - f[i-1] * psi[i-1]) / f[i+1]
        ncross += (psi[i] * psi[i+1] < 0.0)  # Boolean to int
    return psi[icl], ncross, f_10

def inward_integration(psi, f, icl, mesh, f_10):
    for i in range(mesh-1, icl, -1):
        psi[i-1] = (f_10[i] * psi[i] - f[i+1] * psi[i+1]) / f[i-1]
        if abs(psi[i-1]) > 1e10:
            psi[i-1:-2] /= psi[i-1]


def f_and_icl_ho(vpot, e, dx):
    # Init icl
    icl = -1

    # Setup simple f function to find icl
    f = 2*(vpot - e) * (dx ** 2 / 12)

    # Avoid division by zero
    f = np.where(f == 0.0, 1e-20, f)

    # Classical turning point is the last sign change in f + 1
    sign_changes = np.where(np.diff(np.sign(f)))[0]
    icl = sign_changes[-1] + 1

    # f as required by numerov
    f = 1 - f
    return f, 12.0 - 10 * f, icl

def scale_normalize_ho(psi, psi_icl, icl, x):
    # Match wavefunction at icl and normalize
    scaling_factor = psi_icl / psi[icl]
    psi[icl:-2] *= scaling_factor

    # print(f"psi_icl is {psi_icl:.6f}")
    # print(f"Rescaling factor: {scaling_factor:.6f}")
    # print(f"Wavefunction after rescaling: psi[icl]={psi[icl]:.6f}, psi[mesh]={psi[-1]:.6f}")

    norm = np.sqrt(np.trapezoid(psi**2, x))  # Symmetric normalization
    psi /= norm


def harmonic_oscillator(nodes=0, xmax=10.0, mesh=500, max_iter=1000, tol=1e-10):
    x, dx = np.linspace(0, xmax, mesh+1, retstep=True)
    vpot = 0.5 * x**2  

    e_lower, e_upper = np.min(vpot), np.max(vpot)

    for iter in range(max_iter):
        e = 0.5 * (e_lower + e_upper)
        f, f_10, icl = f_and_icl_ho(vpot, e, dx)

        psi = np.zeros(mesh+1)
        if nodes % 2:
            psi[0] = 0.0
            psi[1] = dx
        else:
            psi[0] = 1.0
            psi[1] = 0.5 * (12.0 - 10.0 * f[0]) * psi[0] / f[1]

        psi_icl, ncross, f_10 = outward_integration(psi, f, f_10, icl)
        if nodes % 2 == 0:
            ncross *= 2
        else:
            ncross = 2 * ncross + 1

        if ncross != nodes:
            if ncross > nodes:
                e_upper = e
            else:
                e_lower = e
            continue  

        psi[-1] = dx
        psi[-2] = f_10[-1] * psi[-1] / f[-2]
        inward_integration(psi, f, icl, mesh, f_10)

        scale_normalize_ho(psi, psi_icl, icl, x)

        djump = (psi[icl+1] + psi[icl-1] - (14.0 - 12.0 * f[icl]) * psi[icl]) / dx

        if (e_upper - e_lower) < tol:
            break

        if djump * psi[icl] > 0.0:
            e_upper = e
        else:
            e_lower = e

    return e, psi, x

# Keep your helper functions here: f_and_icl_ho, scale_normalize_ho, outward_integration, inward_integration

# --- Plot the first few states ---
states = []
energies = []
for n in range(4):  # first four states
    e, psi, x = harmonic_oscillator(nodes=n)
    states.append(psi)
    energies.append(e)

plt.figure()
for n, (psi, e) in enumerate(zip(states, energies)):
    plt.plot(x, psi + e, label=f"n={n}")  # shifted up by energy
plt.legend()
plt.title("Harmonic oscillator wavefunctions")
plt.xlabel("x")
plt.ylabel("ψ(x) + E")
plt.show()

# --- Animate a superposition ---
# Choose coefficients (normalized)
c = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])  

def psi_super(x, t):
    total = np.zeros_like(x, dtype=complex)
    for n, (psi, e) in enumerate(zip(states, energies)):
        total += c[n] * psi * np.exp(-1j * e * t)
    return total

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-5, 5)
ax.set_ylim(0, 1.2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    t = i * 0.05
    psi_t = psi_super(x, t)
    prob_density = np.abs(psi_t)**2
    line.set_data(x, prob_density)
    return line,

ani = FuncAnimation(fig, animate, init_func=init,
                    frames=300, interval=50, blit=True)

# Save as MP4 (requires ffmpeg)
ani.save("sho_superposition.mp4", writer="ffmpeg", fps=30)

# Alternatively, save as GIF (requires imagemagick)
# ani.save("sho_superposition.gif", writer="imagemagick", fps=20)

plt.show()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import solve_atom, solve_atom_bisection, init_mesh



import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm


import plotly.graph_objects as go


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import plotly.graph_objects as go

# --- Scatter style orbital probability cloud ---
def plot_orbital_scatter(n, l, m, rmax=20.0, mesh=2000, nsamples=1_000_000):
    # Solve radial part
    e, iterations, psi_r = solve_atom(n=n, l=l, rmax=rmax, mesh=mesh)
    xmesh, rgrid, dx = init_mesh(rmax, mesh, Z=1)

    # Avoid r=0 to prevent division by zero
    R = psi_r[1:] / rgrid[1:]
    rgrid = rgrid[1:]

    # Normalize radial function
    R /= np.sqrt(np.trapz(np.abs(R)**2 * rgrid**2, rgrid))

    # Interpolator
    R_interp = lambda r: np.interp(r, rgrid, R)

    # Sample spherical coordinates (uniform in volume)
    U = np.random.rand(nsamples)
    r = rmax * U**(1/3)
    theta = np.arccos(2*np.random.rand(nsamples)-1)
    phi = 2*np.pi*np.random.rand(nsamples)

    # Wavefunction
    Ylm = sph_harm(m, l, phi, theta)
    psi = R_interp(r) * Ylm
    prob = np.abs(psi)**2

    # Weighted resampling
    prob /= prob.sum()
    idx = np.random.choice(np.arange(nsamples), size=nsamples//5, p=prob)

    r, theta, phi, psi, prob = r[idx], theta[idx], phi[idx], psi[idx], prob[idx]

    # Cartesian coords
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Color by phase/sign
    phase = np.angle(psi)
    colors = np.where(phase >= 0, "royalblue", "crimson")
    alpha_vals = (prob / prob.max())**0.5

    # Plot
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=0.5, c=colors, alpha=alpha_vals)
    ax.set_title(f"Hydrogen orbital (n={n}, l={l}, m={m})")
    ax.set_axis_off()
    plt.show()


# --- Isosurface style orbital plot (smooth lobes) ---
def plot_orbital_isosurface(n, l, m, rmax=15.0, grid=100, mesh=2000, iso=0.01):
    # Cartesian grid
    x = np.linspace(-rmax, rmax, grid)
    y = np.linspace(-rmax, rmax, grid)
    z = np.linspace(-rmax, rmax, grid)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Convert to spherical
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arccos(np.divide(Z, r, out=np.zeros_like(Z), where=r>1e-8))
    phi = np.arctan2(Y, X)

    # Radial function
    e, it, psi_r = solve_atom(n=n, l=l, rmax=rmax, mesh=mesh)
    xmesh, rgrid, dx = init_mesh(rmax, mesh, Z=1)
    R = psi_r[1:] / rgrid[1:]
    rgrid = rgrid[1:]
    R /= np.sqrt(np.trapz(np.abs(R)**2 * rgrid**2, rgrid))

    R_interp = np.interp(r.ravel(), rgrid, R).reshape(r.shape)

    # Wavefunction
    Ylm = sph_harm(m, l, phi, theta)
    psi = R_interp * Ylm
    density = np.abs(psi)**2

    # Interactive isosurface
    fig = go.Figure(data=go.Isosurface(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        value=density.ravel(),
        isomin=iso, isomax=density.max(),
        surface_count=3,
        colorscale="RdBu",
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)))
    fig.show()

# Scatter cloud (dense points, blue/red lobes)
plot_orbital_scatter(2, 1, 0, nsamples=500_000)

# Isosurface (smooth textbook-like 3D lobes)
# plot_orbital_isosurface(2, 1, 0, grid=100)

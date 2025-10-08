import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.special import sph_harm

# --- Import Numerov hydrogen solver ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from numerov.numerov import solve_atom, init_mesh


def hydrogen_density_plot(n, l, m, rmax=25.0, mesh=1421, npts=400, save_path=None):
    """
    Plot the probability density |ψ_nlm(r,θ,φ)|² for the hydrogen atom in the xz-plane (φ=0),
    using Numerov-computed radial wavefunctions and analytical spherical harmonics.

    Args:
        n, l, m : Quantum numbers
        rmax (float): maximum radius (Bohr)
        mesh (int): radial mesh points for Numerov solver
        npts (int): resolution of 2D plot grid
        save_path (str): optional path to save image
    """
    # Solve radial equation with Numerov
    e, iterations, u_r = solve_atom(n=n, l=l, rmax=rmax, mesh=mesh)
    x, r, dx = init_mesh(rmax, mesh, Z=1)

    # Convert Numerov u(r) -> R(r) = u(r)/r
    R_r = np.zeros_like(r)
    R_r[1:] = u_r[1:] / r[1:]
    R_r[0] = R_r[1]  # avoid singularity

    # Normalize R(r)
    norm = np.trapz(np.abs(R_r)**2 * r**2, r)
    R_r /= np.sqrt(norm)

    # Grid in xz-plane (φ = 0)
    x_vals = np.linspace(-rmax/2, rmax/2, npts)
    z_vals = np.linspace(-rmax/2, rmax/2, npts)
    X, Z = np.meshgrid(x_vals, z_vals)
    R = np.sqrt(X**2 + Z**2)
    Theta = np.arccos(np.divide(Z, R, out=np.zeros_like(Z), where=R!=0))
    Phi = np.zeros_like(R)

    # Interpolate radial function on R grid
    R_interp = np.interp(R.flatten(), r, R_r, left=0, right=0).reshape(R.shape)

    # Angular part
    Ylm = sph_harm(m, l, Phi, Theta)
    psi = R_interp * Ylm
    density = np.abs(psi)**2

    # --- Plot styling ---
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    ax.set_facecolor("black")

    im = ax.imshow(
        density,
        extent=[-rmax/2, rmax/2, -rmax/2, rmax/2],
        origin='lower',
        cmap="magma",
        norm=PowerNorm(0.5),  # enhance contrast in darker regions
    )

    # Titles and labels
    ax.set_title(
        f"Átomo de Hidrógeno – |ψₙₗₘ(r,θ,φ)|²\n(n,l,m)=({n},{l},{m})",
        color="white", fontsize=14, pad=15
    )
    ax.set_xlabel("x [Bohr]", color="white")
    ax.set_ylabel("z [Bohr]", color="white")
    ax.tick_params(colors="white")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Densidad de probabilidad electrónica", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="black")
        print(f"Saved density plot to {save_path}")

    plt.show()


# Example usage:
hydrogen_density_plot(n=3, l=1, m=0, rmax=25, mesh=1421, npts=600, save_path="hydrogen_310.png")

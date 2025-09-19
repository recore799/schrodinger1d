# RUNS, BUT HASN’T BEEN VALIDATED
# --- hydrogen_cloud.py -------------------------------------------------------
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import matplotlib.pyplot as plt

from numerov import solve_atom, init_mesh               # your library
from scipy.special import sph_harm
from skimage.measure import marching_cubes              # pip install scikit-image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------- helpers ----------------------------------------
def radial_from_numerov(n, l, rmax=30.0, mesh=1421):
    """
    Returns (E, r_grid, u_grid, R_interp) where
    u(r)=r*R(r) on 'r_grid', and R_interp(r) is a linear interpolant for R(r).
    Assumes your solve_atom returns u(r) (you plotted it as 'r R_{nl}(r)').
    """
    E, iters, u_r = solve_atom(n=n, l=l, rmax=rmax, mesh=mesh)
    x, r_grid, dx = init_mesh(rmax, mesh, Z=1)

    # Avoid division by zero at r=0: define R(0) by smooth limit using first sample
    R_grid = np.zeros_like(u_r)
    R_grid[1:] = u_r[1:] / r_grid[1:]
    R_grid[0]  = R_grid[1]  # simple smooth extension

    # Ensure correct normalization: ∫ |R|^2 r^2 dr = ∫ |u|^2 dr = 1
    # Many Numerov solvers already normalize u; if not, normalize here.
    norm_u = np.trapz(np.abs(u_r)**2, r_grid)
    if not np.isclose(norm_u, 1.0, rtol=1e-3):
        u_r = u_r / np.sqrt(norm_u)
        R_grid[1:] = u_r[1:] / r_grid[1:]
        R_grid[0]  = R_grid[1]

    def R_interp(r_query):
        # simple linear interpolation onto arbitrary r (clip outside)
        return np.interp(r_query, r_grid, R_grid, left=0.0, right=0.0)

    return E, r_grid, u_r, R_interp

def cartesian_grid(N=96, extent=20.0):
    """
    Build a cubic grid in Bohr units. Returns (X,Y,Z), radius r, polar θ, azimuth φ.
    extent: half-size; domain is [-extent, extent] in each axis.
    """
    lin = np.linspace(-extent, extent, N)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    # θ=arccos(z/r) in [0,π], φ=atan2(y,x) in [−π,π]
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arccos(np.where(r>0, Z/r, 1.0))
        phi   = np.arctan2(Y, X)
    return X, Y, Z, r, theta, phi

def real_sph_harm(l, m, theta, phi):
    """
    Real form combinations are prettier for plotting.
    m>0:  √2 * Re[Y_l^m],  m<0:  √2 * Im[Y_l^{|m|}],  m=0: Y_l^0
    """
    if m == 0:
        return sph_harm(0, l, phi, theta).real
    elif m > 0:
        return np.sqrt(2) * sph_harm(m, l, phi, theta).real
    else:
        return np.sqrt(2) * sph_harm(-m, l, phi, theta).imag

# ---------------------------- main builder -----------------------------------
def probability_cloud(n, l, m, rmax=30.0, grid_N=96, box_extent=None,
                      iso_level=0.08, opacity=0.7, cmap_face="#4F46E5"):
    """
    Render an isosurface of |ψ_{nlm}|^2 for a stationary state.
    - box_extent: half-size of the cubic box; default ties to rmax (nice framing)
    - iso_level:  isosurface level as a fraction of max density (0<level<1)
    - cmap_face:  face color (hex or mpl color); change per taste
    """
    if box_extent is None:
        box_extent = 0.85 * rmax

    # 1) Radial part from Numerov
    E, r_grid, u_r, R_of_r = radial_from_numerov(n, l, rmax=rmax, mesh=max(grid_N*15, 1201))

    # 2) Build 3-D grid
    X, Y, Z, r, theta, phi = cartesian_grid(N=grid_N, extent=box_extent)

    # 3) Evaluate wavefunction
    R = R_of_r(r)
    Ylm = real_sph_harm(l, m, theta, phi)        # real combination
    psi = R * Ylm                                # global phase drops from |psi|^2
    rho = np.abs(psi)**2

    # 4) Normalize density to its max for a clean iso choice
    rho /= rho.max() + 1e-15

    # 5) Marching cubes for isosurface
    verts, faces, _, _ = marching_cubes(rho.astype(np.float32), level=iso_level,
                                        spacing=(2*box_extent/(grid_N-1),)*3)

    # 6) Plot
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], linewidths=0.2, alpha=opacity)
    mesh.set_facecolor(cmap_face)
    mesh.set_edgecolor("k")
    ax.add_collection3d(mesh)

    # Axes aesthetics
    lim = box_extent
    ax.set_xlim(0, 2*lim); ax.set_ylim(0, 2*lim); ax.set_zlim(0, 2*lim)
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=20, azim=30)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("x (a₀)"); ax.set_ylabel("y (a₀)"); ax.set_zlabel("z (a₀)")

    plt.title(fr"Hydrogen |$\psi_{{{n}{l}{m}}}|^2$   (isosurface at {iso_level:.2f}·max)")
    plt.tight_layout()
    plt.show()

    return dict(E=E, rho_max=float(rho.max()), iso=iso_level)

# ---------------------------- examples ---------------------------------------
if __name__ == "__main__":
    # Classic “dumbbell”: 2p_z  (n=2, l=1, m=0)
    probability_cloud(n=3, l=2, m=0, rmax=30, grid_N=112, iso_level=0.08, cmap_face="#10B981")

    # Four-lobed 3d (n=3, l=2, m=2_real)
    # probability_cloud(n=3, l=2, m=2, rmax=35, grid_N=120, iso_level=0.06, cmap_face="#6366F1")

    # “Clover” 3d (n=3, l=2, m=0)
    # probability_cloud(n=3, l=2, m=0, rmax=35, grid_N=120, iso_level=0.05, cmap_face="#F59E0B")
# ----------------------------------------------------------------------------- 

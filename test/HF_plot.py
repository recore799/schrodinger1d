import numpy as np
import matplotlib.pyplot as plt

from src.hf.rhf_s_old import scf_rhf

def build_sto3g_basis(zeta: float) -> list[tuple[float, float]]:
    """Build STO-3G basis for a given Slater exponent (zeta)"""
    # Fundamental 1s STO-3G parameters for Zeta=1.0
    d = [0.444635, 0.535328, 0.154329]
    alpha = [0.109818, 0.405771, 2.227660]    

    basis = []
    for d, alpha in zip(d, alpha):
        alpha_scaled = alpha * (zeta ** 2)
        norm_factor = (2.0 * alpha_scaled / np.pi) ** 0.75
        d_scaled = d * norm_factor
        basis.append((alpha_scaled, d_scaled))
    
    return basis


def evaluate_sto3g_1s(primitive_list, center, coords):
    """
    Evaluate a normalized STO-3G 1s contracted GTO at grid coordinates (N,3).
    `primitive_list` is [(alpha, coeff), ...] already normalized as in your build_sto3g_basis().
    """
    diff = coords - center
    r2 = np.sum(diff**2, axis=1)
    values = np.zeros_like(r2)
    for alpha, coeff in primitive_list:
        values += coeff * np.exp(-alpha * r2)
    return values

def plot_molecular_orbital(results, orbital_index=0, extent=3.0, npts=200, plane="xz", ax=None):
    """
    Plot a single molecular orbital ψ_i(r) in a 2D plane slice.
    If ax is provided, plot on it; otherwise, create a new figure.
    """
    C = results["orbital_coefficients"]
    eps = results["orbital_energies"]
    ao_list = results["ao_primitives"]
    centers = results["ao_centers"]

    grid = np.linspace(-extent, extent, npts)
    X, Y = np.meshgrid(grid, grid)

    if plane == "xz":
        coords = np.stack((X, np.zeros_like(X), Y), axis=-1).reshape(-1, 3)
        xlabel, ylabel = "x [Bohr]", "z [Bohr]"
    elif plane == "xy":
        coords = np.stack((X, Y, np.zeros_like(X)), axis=-1).reshape(-1, 3)
        xlabel, ylabel = "x [Bohr]", "y [Bohr]"
    elif plane == "yz":
        coords = np.stack((np.zeros_like(X), X, Y), axis=-1).reshape(-1, 3)
        xlabel, ylabel = "y [Bohr]", "z [Bohr]"
    else:
        raise ValueError("plane must be one of 'xz', 'xy', 'yz'")

    ao_values = np.zeros((coords.shape[0], len(ao_list)))
    for i, (prim, center) in enumerate(zip(ao_list, centers)):
        ao_values[:, i] = evaluate_sto3g_1s(prim, center, coords)

    psi = ao_values @ C[:, orbital_index]
    psi_grid = psi.reshape(X.shape)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = None

    c = ax.contourf(X, Y, psi_grid, levels=80, cmap="RdBu_r")
    plt.colorbar(c, ax=ax, label=f"ψ_{orbital_index}(r)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{results['molecule']} MO {orbital_index}\nε = {eps[orbital_index]:.6f} Ha")
    ax.set_aspect("equal")

    return fig, ax



def plot_density(results, extent=3.0, npts=200, plane="xz"):
    """
    Plot the total electron density ρ(r) in a plane slice.
    """
    P = results["density_matrix"]
    ao_list = results["ao_primitives"]
    centers = results["ao_centers"]

    # Grid
    grid = np.linspace(-extent, extent, npts)
    X, Y = np.meshgrid(grid, grid)

    if plane == "xz":
        coords = np.stack((X, np.zeros_like(X), Y), axis=-1).reshape(-1, 3)
        xlabel, ylabel = "x [Bohr]", "z [Bohr]"
    elif plane == "xy":
        coords = np.stack((X, Y, np.zeros_like(X)), axis=-1).reshape(-1, 3)
        xlabel, ylabel = "x [Bohr]", "y [Bohr]"
    elif plane == "yz":
        coords = np.stack((np.zeros_like(X), X, Y), axis=-1).reshape(-1, 3)
        xlabel, ylabel = "y [Bohr]", "z [Bohr]"
    else:
        raise ValueError("plane must be one of 'xz', 'xy', 'yz'")

    # Evaluate AOs
    ao_values = np.zeros((coords.shape[0], len(ao_list)))
    for i, (prim, center) in enumerate(zip(ao_list, centers)):
        ao_values[:, i] = evaluate_sto3g_1s(prim, center, coords)

    # Compute total electron density
    rho = np.einsum("ij,pi,pj->p", P, ao_values, ao_values)
    rho_grid = rho.reshape(X.shape)

    # Plot
    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, rho_grid, levels=80, cmap="viridis")
    plt.colorbar(label="ρ(r)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Electron Density")
    plt.axis("equal")
    plt.show()


sto3g_h = build_sto3g_basis(zeta=1.24)  # Helium basis
sto3g_he = build_sto3g_basis(zeta=2.095)  # Helium basis
primitives_heh = [sto3g_he, sto3g_h]  # [He, H]

pos_heh = np.array([[0,0,0],[1.4632,0,0]])
Z_heh = (2.0,1.0)

results = scf_rhf(primitives_heh, pos=pos_heh, R=1.4632, Z=Z_heh, n_elec=2, R_nuc=pos_heh, Z_nuc=Z_heh, molecule= "HeH⁺", verbose=1)

# Plot HOMO (since HeH⁺ has 1 occupied MO)
# plot_molecular_orbital(results, orbital_index=0, extent=3.0, plane="xz")

# Plot electron density
# plot_density(results, extent=3.0, plane="xz")

primitives_h = [sto3g_h, sto3g_h]
pos_h = np.array([[0,0,0],[1.4,0,0]])
Z_heh = (1.0,1.0)

results1 = scf_rhf(primitives_h, pos=pos_heh, R=1.4, Z=Z_heh, n_elec=2, R_nuc=pos_heh, Z_nuc=Z_heh, molecule= "H²", verbose=1)

# Plot HOMO (since HeH⁺ has 1 occupied MO)
# plot_molecular_orbital(results1, orbital_index=0, extent=3.0, plane="xz")

# Plot electron density
# plot_density(results1, extent=3.0, plane="xz")


# Create one figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_molecular_orbital(results, orbital_index=0, extent=3.0, plane="xz", ax=axes[0])
plot_molecular_orbital(results1, orbital_index=0, extent=3.0, plane="xz", ax=axes[1])

axes[0].set_title("HeH⁺ HOMO")
axes[1].set_title("H₂ HOMO")

plt.tight_layout()
plt.savefig("molecular_orbitals.png", dpi=300)
plt.show()

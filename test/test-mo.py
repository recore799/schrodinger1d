# heh_density_sto3g.py
import numpy as np
import matplotlib.pyplot as plt

# MO coefficients
C = np.array([[-0.80140853,  0.78233794],
              [-0.33768711, -1.06783692]])

# density matrix
P = np.array([[1.28451268, 0.54124998],
                   [0.54124998, 0.22806435]])

# If you prefer to recompute P from C (RHF closed-shell: 2 * sum occ C[:,i]C[:,i].T)
# uncomment the next line to compute P from C instead:
# P = 2.0 * np.outer(C[:,0], C[:,0])   # HeH+ has 2 electrons: only 1 occupied MO (i=0)

# Geometry (Angstrom)
centers = np.array([[0.0, 0.0, 0.0],    # He at origin
                        [1.4632, 0.0, 0.0]]) # H placed at x=1.4632 Å (change if needed)

# gaussian exponents
zeta_H  = 1.24
zeta_He = 2.095

# ----------------- STO-3G 1s reference primitives (zeta = 1) ---------------
# Standard STO-3G 1s primitive exponents and contraction coeffs (zeta=1).
# We'll scale exponents by zeta^2 for each atom. (coefficients remain the same).
alpha_ref = np.array([2.22766, 0.405771, 0.109818])   # reference exponents
c_ref     = np.array([0.154329, 0.535328, 0.444635])  # contraction coefficients
# source: standard STO-3G table for 1s. (These are the common STO-3G numbers.)
# See literature / tables for STO-3G. (Used here as reference.) :contentReference[oaicite:3]{index=3}

# Build basis list: one contracted 1s per atom (order matches centers_ang)
zetas = [zeta_He, zeta_H]
nbasis = len(zetas)

# convert Angstrom -> Bohr for internal computation if needed
# ang_to_bohr = 1.0 / 0.529177210903
# centers = centers * ang_to_bohr

# ---------- helper functions ----------
def norm_prim_s(alpha):
    """Normalization factor for an s-type primitive Gaussian (atomic units)."""
    return (2.0 * alpha / np.pi) ** (3/4)

def overlap_prim_normed(alpha_i, alpha_j):
    """Overlap integral between two normalized s primitives:
       S_ij = <N_i e^{-alpha_i r^2} | N_j e^{-alpha_j r^2}>
            = (2*sqrt(alpha_i*alpha_j)/(alpha_i+alpha_j))^(3/2)
    """
    return (2.0 * np.sqrt(alpha_i * alpha_j) / (alpha_i + alpha_j)) ** (1.5)

def build_contracted( zeta ):
    """
    Return arrays (alphas_scaled, c_scaled, N_contr)
    where alphas_scaled = alpha_ref * zeta^2,
    c_scaled = c_ref (we will renormalize the contracted function),
    and N_contr is the normalization factor so that
    phi_contr(r) = N_contr * sum_j c_j * N_j * exp(-alpha_j r^2)
    is normalized to 1.
    """
    alphas = alpha_ref * (zeta**2)
    # normalized primitive factors
    Nprims = np.array([norm_prim_s(a) for a in alphas])
    # compute overlap matrix between normalized primitives
    S = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            S[i,j] = overlap_prim_normed(alphas[i], alphas[j])
    # contraction coefficient vector (apply to normalized primitives)
    c = c_ref.copy()
    # contracted normalization: N_contr = 1/sqrt( c^T S c )
    denom = c @ (S @ c)
    Ncontr = 1.0 / np.sqrt(denom)
    return alphas, c, Nprims, Ncontr

def eval_contracted_s(alphas, c, Nprims, Ncontr, R, rpoints):
    """
    Evaluate contracted normalized s-function centered at R on positions rpoints.
    rpoints shape (N,3) in Bohr.
    returns array shape (N,)
    """
    # rpoints (N,3)
    val = np.zeros(rpoints.shape[0])
    for aj, cj, Nj in zip(alphas, c, Nprims):
        diff = rpoints - R
        rsq = np.sum(diff**2, axis=1)
        val += cj * Nj * np.exp(-aj * rsq)
    return Ncontr * val

# Build basis data for each center
basis_data = []
for z in zetas:
    alphas, c, Nprims, Ncontr = build_contracted(z)
    basis_data.append( (alphas, c, Nprims, Ncontr) )

# ---------- grid (2D slice) ----------
# We'll make a slice in the plane z = 0 by default (same as previous example).
xmin, xmax = -2.5, 2.5   # Angstrom
ymin, ymax = -2.5, 2.5
nx, ny = 400, 400        # resolution

xs = np.linspace(xmin, xmax, nx) #* ang_to_bohr
ys = np.linspace(ymin, ymax, ny) #* ang_to_bohr
X, Y = np.meshgrid(xs, ys, indexing='xy')
Z = np.zeros_like(X)
rgrid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1) # (N,3)

# ---------- evaluate basis on grid ----------
Phi = np.zeros((nbasis, rgrid.shape[0]))  # (nbasis, Npoints)
for mu in range(nbasis):
    alphas, c, Nprims, Ncontr = basis_data[mu]
    R = centers[mu]
    Phi[mu,:] = eval_contracted_s(alphas, c, Nprims, Ncontr, R, rgrid)

# ---------- density on grid ----------
rho_flat = np.einsum('mn,mk,nk->k', P, Phi, Phi)
rho = rho_flat.reshape((ny, nx))

# ---------- MO (occupied) on grid ----------
psi_occ = np.dot(C[:,0], Phi)   # occupied MO (first column)
psi2_map = (psi_occ**2).reshape((ny, nx))

# ---------- plotting ----------
fig, ax = plt.subplots(1,2, figsize=(13,5))

# density
levels = np.linspace(0.0, rho.max(), 40)
im = ax[0].contourf(X, Y, rho, levels=levels, cmap='viridis') # divide by ang_to_bohr if needed
ax[0].set_title('Electron density (STO-3G, slice z=0)')
ax[0].set_xlabel('x (Å)'); ax[0].set_ylabel('y (Å)')
for i, R in enumerate(centers):
    ax[0].scatter(R[0], R[1], marker='o', s=140, edgecolor='k', zorder=5,
                  label='He' if i==0 else 'H')
ax[0].legend()
fig.colorbar(im, ax=ax[0], label=r'$\rho$ (e / $a_0^3$)')

# occupied MO^2
im2 = ax[1].contourf(X, Y, psi2_map, levels=40, cmap='plasma') # divide by ang_to_bohr if needed
ax[1].set_title('|psi_occ|^2 (occupied MO)')
ax[1].set_xlabel('x (Å)'); ax[1].set_ylabel('y (Å)')
for i, R in enumerate(centers):
    ax[1].scatter(R[0], R[1], marker='o', s=140, edgecolor='k', zorder=5)
fig.colorbar(im2, ax=ax[1], label='|psi|^2 (a0^-3)')

plt.tight_layout()
plt.savefig('density.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------- optional: quick cube writer (requires building full 3D rho grid) ----------
def write_cube_simple(fname, origin_bohr, nx, ny, nz, axvecs, rho3d, atom_positions_bohr, atom_Z):
    with open(fname, 'w') as f:
        f.write("Cube generated by heh_density_sto3g.py\n")
        f.write("Density grid\n")
        f.write(f"{len(atom_positions_bohr):5d} {origin_bohr[0]:12.6f} {origin_bohr[1]:12.6f} {origin_bohr[2]:12.6f}\n")
        f.write(f"{nx:5d} {axvecs[0][0]:12.6f} {axvecs[0][1]:12.6f} {axvecs[0][2]:12.6f}\n")
        f.write(f"{ny:5d} {axvecs[1][0]:12.6f} {axvecs[1][1]:12.6f} {axvecs[1][2]:12.6f}\n")
        f.write(f"{nz:5d} {axvecs[2][0]:12.6f} {axvecs[2][1]:12.6f} {axvecs[2][2]:12.6f}\n")
        for znum, pos in zip(atom_Z, atom_positions_bohr):
            f.write(f"{znum:5d} {0.000000:12.6f} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        flat = rho3d.ravel(order='C')
        for i,val in enumerate(flat):
            f.write(f"{val:13.6e}")
            if (i+1)%6==0:
                f.write("\n")

# If you want a cube, build a 3D grid and call write_cube_simple(...) with proper args.

print("Saved density.png. (Units: Bohr internally. Axis labels in Å.)")

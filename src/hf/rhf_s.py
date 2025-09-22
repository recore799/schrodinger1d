"""
RHF (s-type GTO) – poster-friendly version
------------------------------------------
- Clean SCF loop
- Always returns a `results` dict (was `return None`)
- Exposes extra fields useful for plotting on grids:
    * ao_primitives: list over AO μ -> [(alpha, coeff), ...]
    * ao_centers:    list over AO μ -> (xμ,yμ,zμ)
    * n_occ:         number of occupied spatial MOs (RHF)
    * atom_of_mu:    index of the atom each AO belongs to (for Mulliken)
- Verbose printing delegated to `print_final_results(results, verbose)`
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import eigh, fractional_matrix_power

# Keep your existing integral builder
from src.hf.s_integrals import build_integral_arrays

def scf_rhf(
    primitives: list[list[tuple[float, float]]],
    pos: np.ndarray,                          # nuclear positions shape (n_atoms,3) in bohr
    R: float,                                 # bond distance (unused here, kept for logging)
    Z: tuple,                                 # tuple of nuclear charges per atom (e.g., (1,1))
    n_elec: int,                              # number of electrons
    R_nuc: np.ndarray, Z_nuc: list,           # for nuclear repulsion energy
    molecule,                                 # label string
    max_iter: int = 50,
    conv_tol: float = 1e-6,
    verbose: int = 1,
) -> dict:
    """
    Restricted Hartree–Fock (RHF) with s-type contracted GTOs (minimal basis).
    Returns a `results` dict ready for plotting.
    """
    # Integrals in AO basis
    # S (overlap), T (kinetic), V (nuclear attraction), H_core = T+V, eri_dict (sparse 2e integrals)
    S, T, V, H_core, eri_dict = build_integral_arrays(primitives, pos, Z)

    nbf = S.shape[0]
    n_occ = n_elec // 2

    if verbose >= 1:
        print(f"[RHF] nbf={nbf}, n_elec={n_elec}, n_occ={n_occ}, R={R:.4f} bohr")

    # Symmetric orthogonalization X = S^{-1/2}
    try:
        X = fractional_matrix_power(S, -0.5)
    except Exception:
        eigvals, eigvecs = eigh(S)
        X = eigvecs @ np.diag(1.0 / np.sqrt(np.clip(eigvals, 1e-14, None))) @ eigvecs.T

    # Initial guess: zero density (core Hamiltonian guess)
    P = np.zeros_like(S)

    # Precompute nuclear repulsion
    E_nuc = compute_nuclear_repulsion_energy(Z_nuc, R_nuc)

    E_elec = None
    E_total = None
    converged = False
    rms_change = np.inf

    for it in range(1, max_iter + 1):
        # Build Fock matrix (your sparse builder)
        F = build_fock_matrix_sparse(H_core, P, eri_dict)

        # Electronic energy (stable expression)
        E_elec = float(np.sum(P * H_core) + 0.5 * np.sum(P * (F - H_core)))
        E_total = E_elec + E_nuc

        # Transform & diagonalize
        Fp = X.T @ F @ X
        eps, Cprime = eigh(Fp)         # ascending eigenvalues
        C = X @ Cprime                 # MO coeffs in AO basis

        # New density
        C_occ = C[:, :n_occ]
        P_new = 2.0 * C_occ @ C_occ.T

        # Convergence check
        rms_change = float(np.sqrt(np.mean((P_new - P) ** 2)))
        if verbose >= 3:
            print(f" iter {it:2d}  E_tot={E_total: .10f}   ΔP_rms={rms_change:.3e}")

        if it > 1 and rms_change < conv_tol:
            converged = True
            break

        P = P_new

    if verbose >= 1 and not converged:
        print(f"[RHF] WARNING: did not converge in {max_iter} iterations (ΔP_rms={rms_change:.3e})")

    # Package results
    results = {
        "energy_electronic": E_elec,
        "energy_nuclear": E_nuc,
        "energy_total": E_total,
        "orbital_energies": eps,
        "orbital_coefficients": C,
        "density_matrix": P_new if converged else P,  # last density
        "fock_matrix": F,
        "overlap_matrix": S,
        "kinetic_matrix": T,
        "nuclear_matrix": V,
        "core_hamiltonian": H_core,
        "iterations": it,
        "converged": converged,
        "eri_tensor": eri_dict,
        "molecule": molecule,
        # --- extras for plotting / analysis ---
        "n_occ": n_occ,
        "ao_primitives": primitives_to_ao_list(primitives),
        "ao_centers": ao_centers_from_pos(primitives, pos),
        "atom_of_mu": atom_of_mu_from_primitives(primitives),
        "Z_nuc": Z_nuc,
        "R_nuc": R_nuc,
    }

    if verbose >= 1:
        print_final_results(results, verbose)

    return results


# ---------- helpers ----------

def compute_nuclear_repulsion_energy(Z_nuc, R_nuc):
    E_nuc = 0.0
    n = len(Z_nuc)
    for i in range(n):
        for j in range(i + 1, n):
            Rij = np.linalg.norm(np.asarray(R_nuc[i]) - np.asarray(R_nuc[j]))
            if Rij > 1e-12:
                E_nuc += Z_nuc[i] * Z_nuc[j] / Rij
    return float(E_nuc)


def build_fock_matrix_sparse(H_core: np.ndarray, P: np.ndarray, eri_dict: dict) -> np.ndarray:
    """
    F_μν = H_core_μν + sum_{λσ} P_{λσ} [ (μν|λσ) - 0.5 (μλ|νσ) ]
    `eri_dict` is a sparse dict with keys (mu,nu,lam,sig) or canonical pairs.
    Replace this with your fast implementation if already present elsewhere.
    """
    nbf = H_core.shape[0]
    G = np.zeros_like(H_core)

    # naive build (ok for tiny bases)
    for mu in range(nbf):
        for nu in range(nbf):
            acc = 0.0
            for lam in range(nbf):
                for sig in range(nbf):
                    P_ls = P[lam, sig]
                    if P_ls == 0.0:
                        continue
                    # Coulomb (μν|λσ)
                    J = eri_dict.get((mu, nu, lam, sig), 0.0)
                    # Exchange (μλ|νσ)
                    K = eri_dict.get((mu, lam, nu, sig), 0.0)
                    acc += P_ls * (J - 0.5 * K)
            G[mu, nu] = acc

    return H_core + G


def primitives_to_ao_list(primitives):
    """
    For a minimal s-basis per atom:
      primitives is already a list per AO: [[(alpha, c), ...], [(alpha, c), ...], ...]
    """
    # If `primitives` is provided per atom (1 s AO per atom), this is identity
    return list(primitives)


def ao_centers_from_pos(primitives, pos):
    """
    Returns a list of centers per AO (μ): (xμ,yμ,zμ). For minimal s basis, 1 AO per atom.
    """
    centers = [tuple(p) for p in pos]
    return centers


def atom_of_mu_from_primitives(primitives):
    """
    For minimal s basis: AO μ belongs to atom μ (0..n_atoms-1).
    """
    return list(range(len(primitives)))


# ---------- printing ----------

def print_final_results(results: dict, verbose: int = 1):
    if verbose <= 0:
        return
    print("\n" + "=" * 60)
    print(f"FINAL SCF RESULTS – {results.get('molecule','')}")
    print("=" * 60)
    print(f"{'Electronic energy:':<24} {results['energy_electronic']:>14.8f} Ha")
    print(f"{'Nuclear repulsion:':<24} {results['energy_nuclear']:>14.8f} Ha")
    print(f"{'Total energy:':<24} {results['energy_total']:>14.8f} Ha")
    print(f"{'SCF iterations:':<24} {results['iterations']:>14d}")
    print(f"{'Converged:':<24} {str(results['converged']):>14}")
    if verbose >= 2:
        eps = results["orbital_energies"]
        print("\nOrbital energies (Ha):")
        for i, e in enumerate(eps):
            print(f"  eps[{i}] = {e: .8f}")
    if verbose >= 3:
        print("\nOverlap S:\n", results["overlap_matrix"])
        print("\nCore H:\n", results["core_hamiltonian"])

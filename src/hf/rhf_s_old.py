import numpy as np
from scipy.linalg import eigh, fractional_matrix_power

from src.hf.s_integrals import build_integral_arrays, canonical_eri_key


def scf_rhf(
        primitives: list[list[tuple[float, float]]],
        pos: np.ndarray, R: float, Z: tuple, n_elec: int,
        R_nuc: np.ndarray, Z_nuc: list, molecule,
        max_iter=50, conv_tol=1e-6, verbose=1
        ) -> dict:
    """
    Restricted Hartree-Fock SCF calculation for diatomic molecules.
    Args:
        primitives: list of two lists containing (alpha, coeff) pairs for each atom
        pos: np.ndarray of 3D nuclear positions
        R: bond distance in atomic units
        Z: nuclear charges of each atom
        max_iter: maximum number of SCF iterations
        conv_tol: convergence tolerance for density matrix
        verbose: whether to print iteration details
    
    Returns:
        dict containing final energy, orbitals, density matrix, etc.
    """
    
    # Build integral arrays, eri_dict is a sparse representation the eri tensor
    S, T, V, H_core, eri_dict = build_integral_arrays(primitives, pos, Z)

   
    # Number of basis functions and electrons
    nbf = S.shape[0]
    n_occ = n_elec // 2  # Number of occupied orbitals (doubly occupied)
    
    if verbose >= 1:
        print(f"[RHF] nbf={nbf}, n_elec={n_elec}, n_occ={n_occ}, R={R:.4f} bohr")
    
    # Symmetric orthogonalization: X = S^(-1/2)
    # This transforms the basis to an orthonormal one
    try:
        X = fractional_matrix_power(S, -0.5)
    except np.linalg.LinAlgError:
        # Fallback: canonical orthogonalization
        eigvals, eigvecs = eigh(S)
        X = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


    # Initial guess: core Hamiltonian
    P = np.zeros_like(S)
    E_nuc = compute_nuclear_repulsion_energy(Z_nuc, R_nuc)

    # SCF iterations
    for iteration in range(max_iter):
        
        # Build Fock matrix
        F = build_fock_matrix_sparse(H_core, P, eri_dict)
        
        # Calculate electronic energy
        E_elec = np.sum(P * H_core) + 0.5 * np.sum(P * (F - H_core))     

        E_total = E_elec + E_nuc
        
        # Transform Fock matrix to orthogonal basis: F' = X^T * F * X
        F_prime = X.T @ F @ X
        
        # Diagonalize F' to get orbital energies and coefficients
        orbital_energies, C_prime = eigh(F_prime)
        
        # Transform back to original (non-orthogonal) basis: C = X * C'
        C = X @ C_prime
        
        # Build new density matrix
        # P_μν = 2 * Σ_i^{occ} C_μi * C_νi
        C_occ = C[:, :n_occ]
        P_new = 2.0 * C_occ @ C_occ.T

        # Check convergence (RMS change in density matrix)
        delta_P = P_new - P
        rms_change = np.sqrt(np.mean(delta_P**2))
        
        if verbose >= 3:
            print(f" iter {iteration:2d}  E_tot={E_total: .10f}   ΔP_rms={rms_change:.3e}")
        
        # Check for convergence
        if iteration > 1 and rms_change < conv_tol:
            converged = True
            break

        P = P_new

    # AFTER SCF loop
    if verbose >= 1 and not converged:
        print(f"[RHF] WARNING: did not converge in {max_iter} iterations (ΔP_rms={rms_change:.3e})")
    
    # Package results
    results = {
        "energy_electronic": E_elec,
        "energy_nuclear": E_nuc,
        "energy_total": E_total,
        "orbital_energies": orbital_energies,
        "orbital_coefficients": C,
        "density_matrix": P_new if converged else P,  # last density
        "fock_matrix": F,
        "overlap_matrix": S,
        "kinetic_matrix": T,
        "nuclear_matrix": V,
        "core_hamiltonian": H_core,
        "iterations": iteration,
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

    print_final_results(results, verbose)
    return results

    # results = {
    #     # Default results
    #     'energy_electronic': E_elec,
    #     'energy_nuclear': E_nuc,
    #     'energy_total': E_total,
    #     'orbital_energies': orbital_energies,
    #     'orbital_coefficients': C,
    #     'density_matrix': P,
    #     'fock_matrix': F,
    #     'overlap_matrix': S,
    #     'kinetic_matrix': T,
    #     'nuclear_matrix': V,
    #     'core_hamiltonian': H_core,
    #     'iterations': iteration + 1,
    #     'converged': rms_change < conv_tol,
    #     'eri_tensor': eri_dict,
    #     'molecule': molecule
        
    # }
    
    # print_final_results(results, verbose)
    
    # return results

def compute_nuclear_repulsion_energy(Z_nuc, R_nuc):
    # Nuclear repulsion energy (generalized for polyatomics)
    E_nuc = 0.0
    E_nuc = sum(Z_nuc[i]*Z_nuc[j] / np.linalg.norm(R_nuc[i]-R_nuc[j])
                for i in range(len(Z_nuc))
                for j in range(i+1, len(Z_nuc)))
    return E_nuc

def build_fock_matrix_sparse(H_core: np.ndarray, P: np.ndarray, eri_dict: dict) -> np.ndarray:
    """
    Build the Fock matrix: F_μν = H_μν^core + Σ_λσ P_λσ [(μν|λσ) - 0.5*(μλ|νσ)]
    """
    nbf = H_core.shape[0]
    G = np.zeros_like(H_core)

    for mu in range(nbf):
        for nu in range(nbf):
            for lam in range(nbf):
                for sig in range(nbf):
                    key1 = canonical_eri_key(mu, nu, lam, sig)  # (μν|λσ)
                    key2 = canonical_eri_key(mu, lam, nu, sig)  # (μλ|νσ)
                    eri1 = eri_dict[key1]
                    eri2 = eri_dict[key2]
                    G[mu, nu] += P[lam, sig] * (eri1 - 0.5 * eri2)

    return H_core + G

def primitives_to_ao_list(primitives):
    """For a minimal s-basis per atom."""
    return list(primitives)

def ao_centers_from_pos(primitives, pos):
    """For minimal s basis: 1 AO per atom → use atom positions."""
    return [tuple(p) for p in pos]

def atom_of_mu_from_primitives(primitives):
    """AO μ belongs to atom μ (0..n_atoms-1)."""
    return list(range(len(primitives)))


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
        # print("\nOverlap S:\n", results["overlap_matrix"])
        # print("\nCore H:\n", results["core_hamiltonian"])
        print("\nCoefficients:\n", results["orbital_coefficients"])
        print("\nDensity matrix:\n", results["density_matrix"])

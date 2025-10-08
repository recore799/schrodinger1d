from src.numerov.numerov import solve_atom, solve_atom_bisection, init_mesh


import numpy as np
import matplotlib.pyplot as plt

def plot_hydrogen_radial_wavefunctions(states, rmax=30.0, mesh=1421, save_path=None):
    """
    Plot hydrogen atom radial wavefunctions Rₙₗ(r) for given (n,l) states.
    
    Args:
        states (list of tuples): [(n1,l1), (n2,l2), ...] quantum numbers
        rmax (float): Maximum radial distance (default: 30 Bohr radii)
        mesh (int): Number of grid points (default: 1421)
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Coulomb potential (-1/r) for reference
    r_plot = np.linspace(0.01, rmax, 500)  # Avoid r=0 divergence
    # plt.plot(r_plot, -1/r_plot, 'k--', alpha=0.3, label="V(r) = -1/r")
    
    # Plot each state's radial wavefunction
    for n, l in states:
        # Solve for the wavefunction
        e, iterations, psi_r = solve_atom(n=n, l=l, rmax=rmax, mesh=mesh)
        
        # Get radial grid and convert ψ(r) to R(r) = ψ(r)/r
        x, r, dx = init_mesh(rmax, mesh, Z=1)
        
        # Plot with distinctive line styles
        line_style = '-' if l == 0 else ('--' if l == 1 else ':')
        plt.plot(r, psi_r, 
                linewidth=1.8, 
                linestyle=line_style,
                label=f"n={n}, l={l} (E={e:.4f})")
    
    # Styling
    plt.title("Hydrogen Atom Radial Wavefunctions $rR_{n\ell}(r)$", fontsize=14)
    plt.xlabel("Radial distance $r$ (a₀)", fontsize=12)
    plt.ylabel("$rR_{n\ell}(r)$", fontsize=12)
    plt.xlim(0, rmax)
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=10, framealpha=1)
    plt.tight_layout()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

# Example usage: Plot low-lying states
states_to_plot = [(1,0), (2,0), (2,1), (3,0), (3,1), (3,2)]
plot_hydrogen_radial_wavefunctions(states_to_plot, save_path="hydrogen.png")

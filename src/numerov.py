import numpy as np
from scipy.optimize import bisect
import sys

###################### Storage for old implementations #########################

############################ Hydrogen Atom #####################################

def hydrogen_atom(n=0, l=0, Z=1, xmin=-8.0, xmax=np.log(100.0), mesh=1260, max_iter=200, tol=1e-10):
    # Logarithmic mesh in x
    x, dx = np.linspace(xmin, xmax, mesh+1, retstep=True)

    # Physical mesh in r
    r = np.exp(x) / Z

    # Precompute common terms
    r2 = r**2
    ddx12 = (dx**2) / 12.0
    lnhfsq = (l+0.5)**2

    v0 = - 2*Z / r
    veff = v0 + lnhfsq / r2
    e_lower = np.min(veff)
    e_upper = v0[-1]

    # print(f"Initial bounds are e_lower: {e_lower:.6f}, e_upper: {e_upper:.6f}")

    e = 0.5 * (e_lower + e_upper)
    de = 1e10
    converged = False

    # print(f"Initial energy bounds: e_lower={e_lower:.6f}, e_upper={e_upper:.6f}, e={e:.6f}")

    iterations = 1
    # Bisection loop
    for iteration in range(max_iter):

        iterations += 1

        if abs(de) <= tol:
            converged = True
            break

        # Find icl
        f, f_10, icl = compute_f_and_icl_atom(mesh, ddx12, r2, lnhfsq, v0, e)


        if icl < 0 or icl >= mesh - 2:
            sys.stderr.write(f"ERROR: hydrogen_atom: icl={icl} out of range (mesh={mesh})\n")
            sys.exit(1)

        # Initialize wavefunction
        psi = np.zeros(mesh+1)

        # Initialize wavefunction
        psi[0] = (r[0] ** (l + 1)) * (1 - (2 * Z * r[0]) / (2*l + 2)) / np.sqrt(r[0])
        psi[1] = (r[1] ** (l + 1)) * (1 - (2 * Z * r[1]) / (2*l + 2)) / np.sqrt(r[1])
        # print(f"Initial psi[0,1]: {psi[0]:.6f}, {psi[1]:.6f}")

        # Outward integration
        nodes = n-l-1
        psi_icl, ncross, f_10 = outward_integration(psi, f, f_10, icl)

        # print(f"After outward: psi[icl]={psi_icl:.6f}, nodes={ncross} (expected {nodes})")

        # Adjust energy bounds based on node count
        if ncross != nodes:
            if ncross > nodes:
                e_upper = e
            else:
                e_lower = e
            e = (e_upper + e_lower) * 0.5
            continue  # Skip inward integration if node count is wrong

        psi[-1] = dx
        psi[-2] = f_10[-1] * psi[-1] / f[-2]

        # print(f_10[-1], f[-2])
        # print(f"Init psi at boundary: psi[-1]= {psi[-1]:.6f}, psi[-2]= {psi[-2]:.6f}")

        # Inward integration (from x=xmax to icl)
        inward_integration(psi, f, icl, mesh, f_10)
        # print(f"Init at boundary (after integration): psi[-1]= {psi[-1]:.6f}, psi[-2]= {psi[-2]:.6f}")


        scale_normalize_atom1(psi, psi_icl, icl, x, r2)

        # Update energy
        e, e_lower, e_upper, de = update_energy(icl, f, psi, dx, ddx12, e, e_lower, e_upper)

        # print(f"psi[icl-1]:{psi[icl-1]:.6f}, psi[icl]:{psi[icl]:.6f}, psi[icl+1]:{psi[icl+1]:.6f}")


    if not converged:
        error_msg = (f"ERROR: hydrogen_atom not converged after {iterations} iterations.\n"
                     f"Final de={de:.2e}, e={e:.6f}, nodes expected (n={n}, l={l})={nodes}, found={ncross}")
        sys.stderr.write(error_msg + "\n")
        sys.exit(1)
    # else:
    #     print(f"Convergence achieved at iterations {iterations}, de = {de:.2e}")

    return e, iterations

def update_energy1(icl, f, psi, dx, ddx12, e, e_lower, e_upper):
    """New energy update with debug prints."""
    print("\n=== NEW ENERGY UPDATE ===")
    print(f"Input - icl: {icl}, e: {e:.6f}, bounds: [{e_lower:.6f},{e_upper:.6f}]")
    print(f"f values at icl-1:icl+1: {f[icl-1]:.6f}, {f[icl]:.6f}, {f[icl+1]:.6f}")
    print(f"psi values at icl-1:icl+1: {psi[icl-1]:.6f}, {psi[icl]:.6f}, {psi[icl+1]:.6f}")

    psi_cusp = (psi[icl-1] * f[icl-1] + psi[icl+1] * f[icl+1] + 10 * f[icl] * psi[icl]) / 12.0
    dfcusp = f[icl] * (psi[icl] / psi_cusp - 1.0)
    de = 0.5 * dfcusp / ddx12 * (psi_cusp ** 2) * dx

    print(f"Cusp calc - psi_cusp: {psi_cusp:.6f}, dfcusp: {dfcusp:.6f}")
    print(f"Energy delta - de: {de:.6f} ({'positive' if de > 0 else 'negative'})")

    if de > 0:
        e_lower = e
        print("Updated lower bound")
    elif de < 0:
        e_upper = e
        print("Updated upper bound")

    e += de
    e = max(min(e, e_upper), e_lower)

    print(f"Output - new e: {e:.6f}, new bounds: [{e_lower:.6f},{e_upper:.6f}]")
    return e, e_lower, e_upper, de

def update_energy(icl, f, psi, dx, ddx12, e, e_lower, e_upper):
    i = icl
    psi_cusp = (psi[i-1] * f[i-1] + psi[i+1] * f[i+1] + 10 * f[i] * psi[i]) / 12.0
    dfcusp = f[i] * (psi[i] / psi_cusp - 1.0)
    de = dfcusp / ddx12 * (psi_cusp ** 2) * dx

    if de > 0:
        e_lower = e
    elif de < 0:
        e_upper = e
    e += de
    e = max(min(e, e_upper), e_lower)
    return e, e_lower, e_upper, de



def compute_f_and_icl_atom(mesh, ddx12, r2, lnhfsq, vpot, e):
    # Compute f in one shot (vectorized)
    f = ddx12 * (lnhfsq + r2 * (vpot - e))

    # Handle zeros to avoid sign issues
    f = np.where(f == 0.0, 1e-20, f)

    # Find classical turning point (last sign change)
    sign_changes = np.where(np.diff(np.sign(f)))[0]
    icl = sign_changes[-1] + 1 if len(sign_changes) > 0 else mesh - 1

    # for i in range(1, 5):
    #     print(f[-i])

    # print("\n")

    # f as required by Numerov
    f = 1.0 - f

    return f, 12.0 - 10.0 * f, icl

################################################################################
################################ INTEGRATION ###################################
def outward_integration(psi, f, f_10, icl):
    ncross = 0
    for i in range(1, icl):
        psi[i+1] = (f_10[i] * psi[i] - f[i-1] * psi[i-1]) / f[i+1]
        ncross += (psi[i] * psi[i+1] < 0.0)  # Boolean to int
    return psi[icl], ncross, f_10

def inward_integration(psi, f, icl, mesh, f_10):
    # Inward integration in [xmax, icl]
    for i in range(mesh-1, icl, -1):
        psi[i-1] = (f_10[i] * psi[i] - f[i+1] * psi[i+1]) / f[i-1]
        if abs(psi[i-1]) > 1e10:
            psi[i-1:-2] /= psi[i-1]


def scale_normalize_atom1(psi, psi_icl, icl, x, r2):
    # Match wavefunction at icl and normalize
    scaling_factor = psi_icl / psi[icl]
    psi[icl:] *= scaling_factor

    # print(f"psi_icl is {psi_icl:.6f}")
    # print(f"Rescaling factor: {scaling_factor:.6f}")
    # print(f"Wavefunction after rescaling: psi[icl]={psi[icl]:.6f}, psi[mesh]={psi[-1]:.6f}")

    norm = np.sqrt(np.trapezoid(psi**2 * r2, x))  # Symmetric normalization
    psi /= norm


    # print(f"Normalization factor: {norm:.6f}")
    # print(f"Wavefunction after normalization (inside func): psi[icl]={psi[icl]:.6f}")



################################################################################

def init_mesh(rmax, mesh, Z=1, xmin=-8.0):
    """
    Initialize the logarithmic and physical mesh arrays.

    Parameters:
        rmax (float): Maximum radial coordinate in physical space (in a.u.).
        mesh (int): Number of mesh intervals.
        Z (float): Nuclear charge (default is 1).
        xmin (float): Lower bound on the logarithmic mesh coordinate x.
                      Default is -8.0, which corresponds to a small r.

    Returns:
        x (ndarray): Logarithmic mesh coordinates.
        dx (float): Uniform spacing in x.
        r (ndarray): Physical mesh computed as r = exp(x)/Z.
    """
    # Compute xmax such that rmax = exp(xmax)/Z
    xmax = np.log(rmax * Z)
    # Compute how many mesh points for the given dx
    # mesh = int((np.log(Z*rmax) - xmin) / dx)

    # Create a uniform mesh in x (logarithmic scale)
    x, dx = np.linspace(xmin, xmax, mesh+1, retstep=True)

    # Convert back to physical r-space
    r = np.exp(x) / Z
    return x, r, dx

def solve_atom(n=1, l=0, Z=1, rmax=500.0, mesh=1421, max_iter=100, tol=1e-10):
    """
    Solve for the hydrogen atom energy eigenvalue using the Numerov method with
    a perturbative correction on the energy.

    Parameters:
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.
        Z (float): Nuclear charge (default 1 for hydrogen).
        mesh (int): Number of points in the mesh.
        rmax (float): Maximum physical radius (in a.u.).
                      Note: internally converted to xmax = log(rmax) for the mesh.
        max_iter (int): Maximum number of iterations in the bisection loop.
        tol (float): Tolerance for convergence on energy.

    Returns:
        e (float): Converged energy eigenvalue.
        iterations (int): Number of iterations required for convergence.
    """
    # Initialize the mesh arrays
    x, r, dx = init_mesh(rmax, mesh, Z)

    # Precompute common terms required by the Numerov algorithm.
    r2 = r**2
    ddx12 = dx**2 / 12.0
    lnhfsq = (l + 0.5)**2

    # Coulomb potential
    v0 = -2 * Z / r
    # Effective potential
    veff = v0 + lnhfsq / r2
    # Initial energy bounds
    e_lower = np.min(veff)
    e_upper = v0[-1]

    e = 0.5 * (e_lower + e_upper)
    de = 1e10  # Initial large energy correction
    iterations = 1
    converged = False

    # Expected number of nodes
    nodes_expected = n - l - 1

    # Bisection loop for refining the energy eigenvalue.
    for iteration in range(max_iter):
        iterations += 1

        if abs(de) <= tol:
            converged = True
            break

        # Compute the Numerov f-array and the classical turning point
        f, f_10, icl = compute_f_and_icl_atom(mesh, ddx12, r2, lnhfsq, v0, e)

        if icl < 0 or icl >= mesh - 2:
            raise ValueError(f"solve_atom: icl = {icl} out of range (mesh = {mesh}).")

        # Set initial conditions for the wavefunction on the mesh.
        psi = np.zeros(mesh+1)
        psi[0] = (r[0] ** (l + 1)) * (1 - (2 * Z * r[0])/(2 * l + 2)) / np.sqrt(r[0])
        psi[1] = (r[1] ** (l + 1)) * (1 - (2 * Z * r[1])/(2 * l + 2)) / np.sqrt(r[1])

        # Outward integration from the origin up to index icl.
        psi_icl, ncross, f_10 = outward_integration(psi, f, f_10, icl)

        # Check that the right number of nodes are within the energy bounds
        if ncross != nodes_expected:
            if ncross > nodes_expected:
                e_upper = e
            else:
                e_lower = e
            e = 0.5 * (e_lower + e_upper)
            continue  # Skip inward integration and update

        # Start the inward integration: initialize the tail of psi.
        psi[-1] = dx  # Apropiate small value
        psi[-2] = f_10[-1] * psi[-1] / f[-2]
        inward_integration(psi, f, icl, mesh, f_10)
        scale_normalize_atom1(psi, psi_icl, icl, x, r2)

        # Update the energy using a perturbative correction.
        e, e_lower, e_upper, de = update_energy(icl, f, psi, dx, ddx12, e, e_lower, e_upper)

    if not converged:
        error_msg = (f"solve_atom did not converge after {iterations} iterations. "
                     f"Final de={de:.2e}, e={e:.6f}, nodes expected (n={n}, l={l})={nodes_expected}, got {ncross}")
        raise RuntimeError(error_msg)

    return e, iterations

# Example usage:
if __name__ == "__main__":
    energy, iters = solve_atom(n=1, l=0, rmax=500.0)
    print(f"Converged energy: {energy:.6f} after {iters} iterations")

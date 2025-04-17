import numpy as np
from scipy.optimize import bisect
import sys

###################### Functions for Numerov algorithm #########################

################### Hydrogen Atom with perturbation updates ####################
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
        psi_icl, ncross = outward_integration(psi, f, f_10, icl)

        # Check that the right number of nodes are within the energy bounds
        if ncross != nodes_expected:
            if ncross > nodes_expected:
                e_upper = e
            else:
                e_lower = e
            e = 0.5 * (e_lower + e_upper)
            continue  # Skip inward integration if node count is wrong

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

    return e, iterations, psi

################################################################################
################################ Helpers1 #######################################

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

    # f as required by Numerov
    f = 1.0 - f

    return f, 12.0 - 10.0 * f, icl


def outward_integration(psi, f, f_10, icl):
    ncross = 0
    for i in range(1, icl):
        psi[i+1] = (f_10[i] * psi[i] - f[i-1] * psi[i-1]) / f[i+1]
        ncross += (psi[i] * psi[i+1] < 0.0)  # Boolean to int
    return psi[icl], ncross

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

    norm = np.sqrt(np.trapezoid(psi**2 * r2, x))  # Symmetric normalization
    psi /= norm


def init_mesh(rmax, mesh, Z=1, xmin=-8.0):
    # Compute xmax such that rmax = exp(xmax)/Z
    xmax = np.log(rmax * Z)
    # Compute how many mesh points for the given dx
    # mesh = int((np.log(Z*rmax) - xmin) / dx)

    # Create a uniform mesh in x (logarithmic scale)
    x, dx = np.linspace(xmin, xmax, mesh+1, retstep=True)

    # Convert back to physical r-space
    r = np.exp(x) / Z
    return x, r, dx


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


################################################################################
###################### Hydrogen Atom with pure bisection #######################
def solve_atom_bisection(n=1, l=0, Z=1, rmax=500.0, mesh=1421, max_iter=100, tol=1e-10):
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

    de = 0

    # Bisection loop for refining the energy eigenvalue.
    for iteration in range(max_iter):
        iterations += 1

        e = 0.5 * (e_lower + e_upper)

        if abs(e_upper - e_lower) <= tol:
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
        psi_icl, ncross = outward_integration(psi, f, f_10, icl)

        # Check that the right number of nodes are within the energy bounds
        if ncross != nodes_expected:
            if ncross > nodes_expected:
                e_upper = e
            else:
                e_lower = e
            e = 0.5 * (e_lower + e_upper)
            continue  # Skip inward integration if node count is wrong

        # Start the inward integration: initialize the tail of psi.
        psi[-1] = dx  # Apropiate small value
        psi[-2] = f_10[-1] * psi[-1] / f[-2]
        inward_integration(psi, f, icl, mesh, f_10)
        scale_normalize_atom1(psi, psi_icl, icl, x, r2)

        # Compute derivative discontinuity
        ddelta = (psi[icl+1] + psi[icl-1] - (14.0 - 12.0 * f[icl]) * psi[icl]) / dx

        # Check convergence
        if (e_upper - e_lower) < tol:
            # print(f"Reached convergence after {iter} iterations")
            break

        # Adjust energy based on ddelta sign
        if ddelta * psi[icl] > 0.0:
            e_upper = e
        else:
            e_lower = e

    if not converged:
        error_msg = (f"solve_atom did not converge after {iterations} iterations. "
                     f"Final de={de:.2e}, e={e:.6f}, nodes expected (n={n}, l={l})={nodes_expected}, got {ncross}")
        raise RuntimeError(error_msg)

    return e, iterations

################################################################################
################# Harmonic Oscillator with pure bisection ######################

def harmonic_oscillator(nodes=0, xmax=10.0, mesh=500, max_iter=1000, tol=1e-10):
    """
    Solve for the energy eigenvalue of the harmonic oscillator using a Numerov-based
    method with an explicit bisection update.

    Parameters:
        nodes (int): Number of nodes in the eigenstate.
        xmax (float): Maximum x value in the mesh.
        mesh (int): Number of points in the mesh (mesh intervals).
        max_iter (int): Maximum iterations in the bisection loop.
        tol (float): Tolerance for the energy convergence.

    Returns:
        e (float): The converged energy eigenvalue.
    """
    # Setup the mesh and potential
    x, dx = np.linspace(0, xmax, mesh+1, retstep=True)
    vpot = 0.5 * x**2  # Harmonic oscillator potential

    # Energy bounds (initially span the range of the potential)
    e_lower = np.min(vpot)
    e_upper = np.max(vpot)

    iterations = 1
    for iter in range(max_iter):
        iterations += 1

        e = 0.5 * (e_lower + e_upper)
        # Compute the Numerov coefficients and get the index of potential discontinuity.
        f, f_10, icl = f_and_icl_ho(vpot, e, dx)

        # Initialize wavefunction using the parity-dependent initializer.
        psi = init_ho_wavefunction(nodes, dx, f)

        # Outward integration from x=0 until icl.
        psi_icl, ncross = outward_integration(psi, f, f_10, icl)

        # Adjust the node count according to the symmetry properties.
        if nodes % 2 == 0:
            # Even
            ncross *= 2
        else:
            # Odd
            ncross = 2 * ncross + 1

        # Update energy bounds if the node count is off.
        if ncross != nodes:
            if ncross > nodes:
                e_upper = e
            else:
                e_lower = e
            continue  # Skip further steps if nodes do not match

        # Inward integration on the tail: initialize boundary conditions.
        psi[-1] = dx
        psi[-2] = f_10[-1] * psi[-1] / f[-2]
        inward_integration(psi, f, icl, mesh, f_10)

        scale_normalize_ho(psi, psi_icl, icl, x)

        # Compute the derivative discontinuity at the matching point
        ddelta = (psi[icl+1] + psi[icl-1] - (14.0 - 12.0 * f[icl]) * psi[icl]) / dx

        # Check convergence: update energy bounds based on the sign of the discontinuity.
        if (e_upper - e_lower) < tol:
            break

        if ddelta * psi[icl] > 0.0:
            e_upper = e
        else:
            e_lower = e

    return e, iterations

################################################################################
################################ Helpers2 #######################################

def init_ho_wavefunction(nodes, dx, f):
    psi = np.zeros_like(f)
    if nodes % 2:
        # Odd parity
        psi[0] = 0.0
        psi[1] = dx
    else:
        # Even parity
        psi[0] = 1.0
        psi[1] = 0.5 * (12.0 - 10.0 * f[0]) * psi[0] / f[1]
    return psi

def f_and_icl_ho(vpot, e, dx):
    # Init icl
    icl = -1

    # Setup simple f function to find icl
    f = 2*(vpot - e) * (dx ** 2 / 12)

    # Avoid division by zero
    f = np.where(f == 1.0, 1+1e-20, f)

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

    norm = np.sqrt(np.trapezoid(2*psi**2, x))  # Symmetric normalization
    psi /= norm

    # print(f"Normalization factor: {norm:.6f}")
    # print(f"Wavefunction after normalization (inside func): psi[icl]={psi[icl]:.6f}")

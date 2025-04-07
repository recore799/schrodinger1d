import numpy as np
from scipy.optimize import bisect
import sys

###################### Storage for old implementations #########################




############################ Hydrogen Atom #####################################

# def hydrogen_atom(n=0, l=0, Z=1, xmin=-8.0, xmax=np.log(100.0), mesh=1260, max_iter=200, tol=1e-10):
#     # Logarithmic mesh in x
#     x, dx = np.linspace(xmin, xmax, mesh+1, retstep=True)

#     # Physical mesh in r
#     r = np.exp(x) / Z

#     # Precompute common terms
#     r2 = r**2
#     ddx12 = (dx**2) / 12.0
#     lnhfsq = (l+0.5)**2

#     v0 = - Z / r
#     veff = v0 + lnhfsq / r2
#     e_lower = np.min(veff)
#     e_upper = v0[-1]

#     # print(f"Initial bounds are e_lower: {e_lower:.6f}, e_upper: {e_upper:.6f}")

#     e = 0.5 * (e_lower + e_upper)
#     de = 1e10
#     converged = False

#     # print(f"Initial energy bounds: e_lower={e_lower:.6f}, e_upper={e_upper:.6f}, e={e:.6f}")

#     iterations = 1
#     # Bisection loop
#     for iteration in range(max_iter):

#         iterations += 1

#         if abs(de) <= tol:
#             converged = True
#             break

#         # Find icl
#         f, f_10, icl = compute_f_and_icl_atom(mesh, ddx12, r2, lnhfsq, v0, e)


#         if icl < 0 or icl >= mesh - 2:
#             sys.stderr.write(f"ERROR: hydrogen_atom: icl={icl} out of range (mesh={mesh})\n")
#             sys.exit(1)

#         # Initialize wavefunction
#         psi = np.zeros(mesh+1)

#         # Initialize wavefunction
#         psi[0] = (r[0] ** (l + 1)) * (1 - (2 * Z * r[0]) / (2*l + 2)) / np.sqrt(r[0])
#         psi[1] = (r[1] ** (l + 1)) * (1 - (2 * Z * r[1]) / (2*l + 2)) / np.sqrt(r[1])
#         # print(f"Initial psi[0,1]: {psi[0]:.6f}, {psi[1]:.6f}")

#         # Outward integration
#         nodes = n-l-1
#         psi_icl, ncross, f_10 = outward_integration(psi, f, f_10, icl)

#         # print(f"After outward: psi[icl]={psi_icl:.6f}, nodes={ncross} (expected {nodes})")

#         # Adjust energy bounds based on node count
#         if ncross != nodes:
#             if ncross > nodes:
#                 e_upper = e
#             else:
#                 e_lower = e
#             e = (e_upper + e_lower) * 0.5
#             continue  # Skip inward integration if node count is wrong

#         psi[-1] = dx
#         psi[-2] = f_10[-1] * psi[-1] / f[-2]

#         # print(f_10[-1], f[-2])
#         # print(f"Init psi at boundary: psi[-1]= {psi[-1]:.6f}, psi[-2]= {psi[-2]:.6f}")

#         # Inward integration (from x=xmax to icl)
#         inward_integration(psi, f, icl, mesh, f_10)
#         # print(f"Init at boundary (after integration): psi[-1]= {psi[-1]:.6f}, psi[-2]= {psi[-2]:.6f}")


#         scale_normalize_atom1(psi, psi_icl, icl, x, r2)

#         # Update energy
#         e, e_lower, e_upper, de = update_energy(icl, f, psi, dx, ddx12, e, e_lower, e_upper)

#         # print(f"psi[icl-1]:{psi[icl-1]:.6f}, psi[icl]:{psi[icl]:.6f}, psi[icl+1]:{psi[icl+1]:.6f}")


#     if not converged:
#         error_msg = (f"ERROR: hydrogen_atom not converged after {iterations} iterations.\n"
#                      f"Final de={de:.2e}, e={e:.6f}, nodes expected (n={n}, l={l})={nodes}, found={ncross}")
#         sys.stderr.write(error_msg + "\n")
#         sys.exit(1)
#     # else:
#     #     print(f"Convergence achieved at iterations {iterations}, de = {de:.2e}")

#     return e, iterations

# def compute_f_and_icl(mesh, ddx12, sqlhf, r2, vpot, e):
#     """Compute the f function and find the classical turning point (icl)."""
#     f = np.zeros(mesh + 1)
#     icl = -1
#     f[0] = ddx12 * (sqlhf + 2 * r2[0] * (vpot[0] - e))
#     for i in range(1, mesh + 1):
#         f[i] = ddx12 * (sqlhf + 2 * r2[i] * (vpot[i] - e))
#         if f[i] == 0.0:
#             f[i] = 1e-20
#         if np.sign(f[i]) != np.sign(f[i-1]):
#             icl = i
#     f = 1.0 - f
#     f_10 = 12.0 - 10.0 * f
#     return f, f_10, icl

# def update_energy1(icl, f, psi, dx, ddx12, e, e_lower, e_upper):
#     """New energy update with debug prints."""
#     print("\n=== NEW ENERGY UPDATE ===")
#     print(f"Input - icl: {icl}, e: {e:.6f}, bounds: [{e_lower:.6f},{e_upper:.6f}]")
#     print(f"f values at icl-1:icl+1: {f[icl-1]:.6f}, {f[icl]:.6f}, {f[icl+1]:.6f}")
#     print(f"psi values at icl-1:icl+1: {psi[icl-1]:.6f}, {psi[icl]:.6f}, {psi[icl+1]:.6f}")

#     psi_cusp = (psi[icl-1] * f[icl-1] + psi[icl+1] * f[icl+1] + 10 * f[icl] * psi[icl]) / 12.0
#     dfcusp = f[icl] * (psi[icl] / psi_cusp - 1.0)
#     de = 0.5 * dfcusp / ddx12 * (psi_cusp ** 2) * dx

#     print(f"Cusp calc - psi_cusp: {psi_cusp:.6f}, dfcusp: {dfcusp:.6f}")
#     print(f"Energy delta - de: {de:.6f} ({'positive' if de > 0 else 'negative'})")

#     if de > 0:
#         e_lower = e
#         print("Updated lower bound")
#     elif de < 0:
#         e_upper = e
#         print("Updated upper bound")

#     e += de
#     e = max(min(e, e_upper), e_lower)

#     print(f"Output - new e: {e:.6f}, new bounds: [{e_lower:.6f},{e_upper:.6f}]")
#     return e, e_lower, e_upper, de

# def update_energy(icl, f, psi, dx, ddx12, e, e_lower, e_upper):
#     i = icl
#     psi_cusp = (psi[i-1] * f[i-1] + psi[i+1] * f[i+1] + 10 * f[i] * psi[i]) / 12.0
#     dfcusp = f[i] * (psi[i] / psi_cusp - 1.0)
#     de = 0.5 * dfcusp / ddx12 * (psi_cusp ** 2) * dx

#     if de > 0:
#         e_lower = e
#     elif de < 0:
#         e_upper = e
#     e += de
#     e = max(min(e, e_upper), e_lower)
#     return e, e_lower, e_upper, de



# def compute_f_and_icl_atom(mesh, ddx12, r2, lnhfsq, vpot, e):
#     # Compute f in one shot (vectorized)
#     f = ddx12 * (lnhfsq + 2 * r2 * (vpot - e))

#     # Handle zeros to avoid sign issues
#     f = np.where(f == 0.0, 1e-20, f)

#     # Find classical turning point (last sign change)
#     sign_changes = np.where(np.diff(np.sign(f)))[0]
#     icl = sign_changes[-1] + 1 if len(sign_changes) > 0 else mesh - 1

#     # for i in range(1, 5):
#     #     print(f[-i])

#     # print("\n")

#     # f as required by Numerov
#     f = 1.0 - f

#     return f, 12.0 - 10.0 * f, icl

# ################################################################################
# ################################ INTEGRATION ###################################
# def outward_integration(psi, f, f_10, icl):
#     ncross = 0
#     for i in range(1, icl):
#         psi[i+1] = (f_10[i] * psi[i] - f[i-1] * psi[i-1]) / f[i+1]
#         ncross += (psi[i] * psi[i+1] < 0.0)  # Boolean to int
#     return psi[icl], ncross, f_10

# def inward_integration(psi, f, icl, mesh, f_10):
#     # Inward integration in [xmax, icl]
#     for i in range(mesh-1, icl, -1):
#         psi[i-1] = (f_10[i] * psi[i] - f[i+1] * psi[i+1]) / f[i-1]
#         if abs(psi[i-1]) > 1e10:
#             psi[i-1:-2] /= psi[i-1]


# def scale_normalize_atom1(psi, psi_icl, icl, x, r2):
#     # Match wavefunction at icl and normalize
#     scaling_factor = psi_icl / psi[icl]
#     psi[icl:] *= scaling_factor

#     # print(f"psi_icl is {psi_icl:.6f}")
#     # print(f"Rescaling factor: {scaling_factor:.6f}")
#     # print(f"Wavefunction after rescaling: psi[icl]={psi[icl]:.6f}, psi[mesh]={psi[-1]:.6f}")

#     norm = np.sqrt(np.trapezoid(psi**2 * r2, x))  # Symmetric normalization
#     psi /= norm


#     # print(f"Normalization factor: {norm:.6f}")
#     # print(f"Wavefunction after normalization (inside func): psi[icl]={psi[icl]:.6f}")






################################################################################
############################## OLD #############################################
# JUST UNCOMMENT THE CODE IT SHOULD RUN

def solve_sheq(n, l, zeta, mesh, dx, r, sqr, r2, vpot, y):
    """Solve the Schrödinger equation using the Numerov method."""
    eps = 1e-10
    n_iter = 200

    ddx12 = dx**2 / 12.0
    sqlhf = (l + 0.5)**2
    x2l2 = 2 * l + 2

    # Initialize energy bounds
    elw, eup = init_energy_bounds(mesh, sqlhf, r2, vpot)
    if eup - elw < eps:
        sys.stderr.write(f"ERROR: solve_sheq: eup={eup} and elw={elw} are too close.\n")
        sys.exit(1)
    # print(f"Initial energy bounds: e_lower={elw:.6f}, e_upper={eup:.6f}")


    e = (elw + eup) * 0.5
    de = 1e10  # Initial large value
    converged = False

    # DEBUG: Check energy bounds
    # print(f"Initial energy bounds: elw={elw:.6f}, eup={eup:.6f}, e={e:.6f}")

    iterations = 0

    for kkk in range(n_iter):
        iterations += 1
        if abs(de) <= eps:
            converged = True
            break

        # Compute f and find icl
        f, icl = compute_f_and_icl(mesh, ddx12, sqlhf, r2, vpot, e)


        # DEBUG: icl evolution
        # print(f"Iteration {kkk+1}: icl={icl}, e={e:.6f}, de={de:.6f}")

        if icl < 0 or icl >= mesh - 2:
            sys.stderr.write(f"ERROR: solve_sheq: icl={icl} out of range (mesh={mesh})\n")
            sys.exit(1)

        f[:] = 1.0 - f[:]

        # Initialize wavefunction
        y[0] = (r[0] ** (l + 1)) * (1 - (2 * zeta * r[0]) / x2l2) / sqr[0]
        y[1] = (r[1] ** (l + 1)) * (1 - (2 * zeta * r[1]) / x2l2) / sqr[1]

        # DEBUG: Initial wavefunction values
        # print(f"y[0]:{y[0]:.6f}, y[1]:{y[1]:.6f}")

        # Outward integration
        nodes = n - l - 1
        ncross, fac = outward_integration(icl, f, y)
        # DEBUG: Number of nodes
        # print(f"Outward integration: ncross={ncross}, nodes={nodes}")
        # print(f"After outward: psi[icl]={fac:.6f}, nodes={ncross} (expected {nodes}) with icl = {icl}")

        if ncross != nodes:
            if ncross > nodes:
                eup = e
            else:
                elw = e
            e = (eup + elw) * 0.5
            # DEBUG: New energy bounds
            # print(f"Adjusting energy bounds: elw={elw:.6f}, eup={eup:.6f}, e={e:.6f}")
            continue


        # Inward integration
        inward_integration(mesh, icl, f, y, dx)


        # print(f"Init at boundary: psi[mesh]= {y[mesh]:.6f}, psi[mesh-1]= {y[mesh-1]:.6f}")
        # print(f"Init at boundary: psi[-1]= {y[-1]:.6f}, psi[-2]= {y[-2]:.6f}")

        # print(f"BF y[icl-1]:{y[icl-1]:.6f}, y[icl]:{y[icl]:.6f}, y[icl+1]:{y[icl+1]:.6f}")
        # Rescale and normalize
        rescale_and_normalize(mesh, icl, y, fac, r2, dx)
        # print(f"AF y[icl-1]:{y[icl-1]:.6f}, y[icl]:{y[icl]:.6f}, y[icl+1]:{y[icl+1]:.6f}")

        # Update energy
        e, elw, eup, de = update_energy(icl, f, y, ddx12, dx, e, elw, eup)
        # print(f"y[icl-1]:{y[icl-1]:.6f}, y[icl]:{y[icl]:.6f}, y[icl+1]:{y[icl+1]:.6f}")
        # DEBUG: Energy updates
        # print(f"Energy update: e={e:.6f}, de={de:.6f}, elw={elw:.6f}, eup={eup:.6f}")

    if not converged:
        error_msg = (f"ERROR")#(f"ERROR: solve_sheq not converged after {n_iter} iterations.\n"
                     #f"Final de={de:.2e}, e={e:.6f}, nodes expected={nodes}, found={ncross}")
        sys.stderr.write(error_msg + "\n")
        sys.exit(1)
    # else:
        # print(f"Convergence achieved at iteration {kkk+1}, de = {de:.2e}")

    return e, iterations

def do_mesh(mesh, zmesh, xmin, dx, r, sqr, r2):
        for i in range(mesh + 1):
                x = xmin + dx * i
                r[i] = np.exp(x) / zmesh
                sqr[i] = np.sqrt(r[i])
                r2[i] = r[i] * r[i]
        # print( " radial grid information:\n")
        # print( f" dx   = {dx:.6f}")
        # print( f", xmin = {xmin:.6f}")
        # print( f", zmesh ={zmesh:.6f}\n")
        # print( f" mesh = {mesh}")
        # print( f", r(0) = {r[0]:.6f}")
        # print( f", r(mesh) = {r[mesh]:.6f}\n")

def init_energy_bounds(mesh, sqlhf, r2, vpot):
    """Initialize energy bounds (elw and eup)."""
    eup = vpot[mesh]
    sqlhf_over_r2 = sqlhf / r2
    sum_terms = sqlhf_over_r2 + vpot
    elw = np.min(sum_terms)
    return elw, eup

def compute_f_and_icl(mesh, ddx12, sqlhf, r2, vpot, e):
    """Compute the f function and find the classical turning point (icl)."""
    f = np.zeros(mesh + 1)
    icl = -1
    f[0] = ddx12 * (sqlhf + r2[0] * (vpot[0] - e))
    for i in range(1, mesh + 1):
        f[i] = ddx12 * (sqlhf + r2[i] * (vpot[i] - e))
        if f[i] == 0.0:
            f[i] = 1e-20
        if np.sign(f[i]) != np.sign(f[i-1]):
            icl = i

    # for i in range(1,5):
    #         print(f[-i])

    # print("\n")
    return f, icl




def outward_integration(icl, f, y):
    """Perform outward integration and count the number of nodes."""
    ncross = 0
    for i in range(1, icl):
        y[i+1] = ((12.0 - f[i] * 10.0) * y[i] - f[i-1] * y[i-1]) / f[i+1]
        if y[i] * y[i+1] < 0:
            ncross += 1

    # DEBUG: Outward integration steps
    # print(f"Outward y[icl]:{y[icl]:.6f}")
    # for i in range(1, icl, 100):  # Sample every 100 steps
    #     print(f"Step {i}: y={y[i]:.6f}, f={f[i]:.6f}")

    return ncross, y[icl]

def inward_integration(mesh, icl, f, y, dx):
    """Perform inward integration."""
    y[mesh] = dx
    y[mesh-1] = (12.0 - f[mesh] * 10.0) * y[mesh] / f[mesh-1]
    # print((12.0 - f[mesh] * 10.0), f[mesh-1])
    # print(f"Init at boundary: psi[-1]= {y[-1]:.6f}, psi[-2]= {y[-2]:.6f}")
    for i in range(mesh-1, icl, -1):
        y[i-1] = ((12.0 - f[i] * 10.0) * y[i] - f[i+1] * y[i+1]) / f[i-1]
        if y[i-1] > 1e10:
            y[i-1:-2] /= y[i-1]

def rescale_and_normalize(mesh, icl, y, fac, r2, dx):
    """Rescale and normalize the wavefunction."""
    # DEBUG: Rescaling parameters
    # print(f"fac before rescaling: {fac:.6f}, y[icl]: {y[icl]:.6f}")
    scaling_factor = fac / y[icl]
    y[icl:mesh+1] *= scaling_factor

    # DEBUG: Normalization parameters
    # print(f"psi_icl is {fac:.6f}")
    # print(f"Rescaling factor: {scaling_factor:.6f}")
    # print(f"Wavefunction after rescaling: y[icl]={y[icl]:.6f}, y[mesh]={y[mesh]:.6f}")

    norm_sq = np.sum(y[1:mesh+1]**2 * r2[1:mesh+1] * dx)
    norm = np.sqrt(norm_sq)
    y[:mesh+1] /= norm

    # DEBUG: Post-normalization values
    # print(f"Normalization factor: {norm:.6f}")
    # print(f"Wavefunction after normalization: y[icl]={y[icl]:.6f}")

def update_energy(icl, f, y, ddx12, dx, e, elw, eup):
    """Compute the cusp condition and update the energy."""
    i = icl
    ycusp = (y[i-1] * f[i-1] + y[i+1] * f[i+1] + 10 * f[i] * y[i]) / 12.0
    dfcusp = f[i] * (y[i] / ycusp - 1.0)
    de = 0.5*dfcusp / ddx12 * (ycusp ** 2) * dx

    # DEBUG: Energy update parameters
    # print(f"ycusp: {ycusp:.6f}, dfcusp: {dfcusp:.6f}, de: {de:.6f}")
    # print(f"Energy terms: ddx12={ddx12:.6f}, ycusp^2={ycusp**2:.6f}")

    if de > 0:
        elw = e
    elif de < 0:
        eup = e
    e += de
    e = max(min(e, eup), elw)
    return e, elw, eup, de

def update_energy1(icl, f, y, ddx12, dx, e, elw, eup):
    """Original energy update with debug prints."""
    print("\n=== OLD ENERGY UPDATE ===")
    print(f"Input - icl: {icl}, e: {e:.6f}, bounds: [{elw:.6f},{eup:.6f}]")
    print(f"f values at icl-1:icl+1: {f[icl-1]:.6f}, {f[icl]:.6f}, {f[icl+1]:.6f}")
    print(f"y values at icl-1:icl+1: {y[icl-1]:.6f}, {y[icl]:.6f}, {y[icl+1]:.6f}")

    ycusp = (y[icl-1] * f[icl-1] + y[icl+1] * f[icl+1] + 10 * f[icl] * y[icl]) / 12.0
    dfcusp = f[icl] * (y[icl] / ycusp - 1.0)
    de = 0.5*dfcusp / ddx12 * (ycusp ** 2) * dx

    print(f"Cusp calc - ycusp: {ycusp:.6f}, dfcusp: {dfcusp:.6f}")
    print(f"Energy delta - de: {de:.6f} ({'positive' if de > 0 else 'negative'})")

    if de > 0:
        elw = e
        print("Updated lower bound")
    elif de < 0:
        eup = e
        print("Updated upper bound")

    e += de
    e = max(min(e, eup), elw)

    print(f"Output - new e: {e:.6f}, new bounds: [{elw:.6f},{eup:.6f}]")
    return e, elw, eup, de



def harmonic_oscillator_test(nodes=0, xmax=10.0, mesh=500, max_iter=1000, tol=1e-10):
    """
    Solves the quantum harmonic oscillator using the Numerov method and bisection.

    Args:
        nodes (int): Desired number of nodes in the wavefunction.
        xmax (float): Maximum x value (symmetric about 0).
        mesh (int): Number of grid points.
        max_iter (int): Maximum number of bisection iterations.
        tol (float): Tolerance for energy convergence.

    Returns:
        e (float): Converged energy eigenvalue.
        x (array): Grid points.
        psi (array): Wavefunction (normalized).
    """
    # Initialize grid and potential
    x, dx = np.linspace(0, xmax, mesh, retstep=True)
    vpot = x**2  # Harmonic oscillator potential

    # Initial energy bounds
    e_lower = np.min(vpot)
    e_upper = np.max(vpot)

    # Bisection loop
    for iter in range(max_iter):
        e = 0.5 * (e_lower + e_upper)

        # Setup simple f funcion to find icl
        icl = -1
        f = (vpot - 2*e) * (dx ** 2 / 12)

        # Avoid division by zero in sign check
        f = np.where(f == 0.0, 1e-20, f)

        # Find classical turning point (last sign change in f)
        sign_changes = np.where(np.diff(np.sign(f)))[0]
        icl = sign_changes[-1] + 1  # Last crossing index

        # f as required by numerov
        f = 1 - f
        # Initialize wavefunction based on parity
        psi = np.zeros(mesh)
        if nodes % 2:
            # Odd
            psi[0] = 0.0
            psi[1] = dx
        else:
            # Even
            psi[0] = 1.0
            psi[1] = 0.5 * (12.0 - 10.0 * f[0]) * psi[0] / f[1]

        # Outward integration (from x=0 to icl)
        ncross = 0
        for i in range(1, icl):
            psi[i+1] = ((12.0 - 10.0 * f[i]) * psi[i] - f[i-1] * psi[i-1]) / f[i+1]
            if psi[i] * psi[i+1] < 0.0:
                ncross += 1

        psi_icl = psi[icl]

        # Account for symmetry in node count
        if nodes % 2 == 0:
            ncross *= 2
        else:
            ncross = 2 * ncross + 1

        # Adjust energy bounds based on node count
        if ncross != nodes:
            if ncross > nodes:
                e_upper = e
            else:
                e_lower = e
            continue  # Skip inward integration if node count is wrong

        # Inward integration (from x=xmax to icl)
        psi[-1] = dx  # Seed value at xmax
        psi[-2] = (12.0 - 10.0 * f[-1]) * psi[-1] / f[-2]

        for i in range(mesh-2, icl, -1):
            psi[i-1] = ((12.0 - 10.0 * f[i]) * psi[i] - f[i+1] * psi[i+1]) / f[i-1]

        # Match wavefunction at icl and normalize
        scaling_factor = psi_icl / psi[icl]
        psi[icl:] *= scaling_factor
        norm = np.sqrt(np.trapezoid(psi**2, x) * 2)  # Symmetric normalization
        psi /= norm

        # Compute derivative discontinuity
        djump = (psi[icl+1] + psi[icl-1] - (14.0 - 12.0 * f[icl]) * psi[icl]) / dx

        # Check convergence
        if (e_upper - e_lower) < tol:
            break

        # Adjust energy based on djump sign
        if djump * psi[icl] > 0.0:
            e_upper = e
        else:
            e_lower = e

    return e


#################################################################################
############################ BASIC NUMEROV WITH CLASSES #########################
class numerov0:
    def __init__(self, V, xL=-5, xR=5, n=501, tol=1e-8):
        """
        Initializes the integrator for any differential equation in the form:
            ψ''(x) = -G(x) * ψ(x).

        Parameters:
            V (callable): Potential for the Schrödinger equation where -G(x) = 2*E - V(x)
            xL (float): Left boundary
            xR (float): Right boundary
            n (int): Number of grid points
            tol (float): Default tolerance for convergence and node detection
        """
        self.V = V
        self.xL = xL
        self.xR = xR
        self.n = n
        self.tol = tol
        self.x, self.h = np.linspace(xL, xR, n, retstep=True, dtype=float)
        self.last_psi = None  # Cache the last computed psi

    def numerov(self, E):
        """
        Propagates a function using Numerov's method for a given effective potential G(x)

        Parameters:
            E: float, energy used to propagate wavefunction
        Returns:
            psi (np.ndarray): The computed wavefunction
        """
        psi = np.zeros(self.n, dtype=float) # Initiate array
        psi[0] = 0.0 # Initial boundary condition
        psi[1] = 1e-6  # Kickstart the propagation

        G = 2*E - self.V(self.x)
        # Denominator
        F = 1 + (self.h**2 / 12) * G

        for i in range(1, self.n - 1):
            psi[i + 1] = ((12 - 10 * F[i]) * psi[i] - F[i - 1] * psi[i - 1]) / F[i + 1]

        return psi

    def count_nodes(self, psi, tol=None):
        """
        Counts the number of sign changes in wavefunction.

        Parameters:
            psi (np.ndarray): The wavefunction array.
            tol (float): Tolerance for considering a value as zero.
        Returns:
            int: Number of nodes in the wavefunction
        """
        if tol is None:
            tol = self.tol

        nodes = 0
        # Initialize with the first nonzero sign
        # (starting from the first propagated value of psi, psi[2])
        s_prev = np.sign(psi[2]) if abs(psi[2]) > tol else 0

        for val in psi[3:]:
            s = np.sign(val) if abs(val) > tol else 0

            # If we haven't got a previous sign (due to being near zero), update it
            if s_prev == 0 and s != 0:
                s_prev = s
                continue

            # Continue count if we have a valid previous sign and it differs from the current
            if s != 0 and s != s_prev:
                nodes += 1
                s_prev = s

        return nodes

    def _bracket_eigenvalue(self, target_nodes, E_min, E_max, num=500, tol=None):
        """
        Scans energies between E_min and E_max to find an interavl where
        the number of nodes changes from target_nodes to something else.
        """
        if tol is None:
            tol = self.tol

        energies = np.linspace(E_min, E_max, num)
        prev_nodes = None
        bracket = None
        for E in energies:
            psi = self.numerov(E)
            nodes = self.count_nodes(psi, tol)
            if prev_nodes is not None and prev_nodes == target_nodes and nodes != target_nodes:
                bracket = (E_prev, E)
                break
            E_prev, prev_nodes = E, nodes
        else:
            raise ValueError(f"Could not bracket eigenvalue for {target_nodes} nodes")
        # print(f"Eigenenergy interval found ({bracket[0]:.6f},{bracket[1]:.6f})")
        return bracket

    def _boundary(self, E):
        """
        Compute boundary condition at xR for energy E and cache the wavefunction.
        """
        psi = self.numerov(E)
        self.last_psi = psi  # Cache the computed wavefunction
        return psi[-1]

    def solve_state(self, energy_level, E_min, E_max, tol=None):
        """
        Solve for the specified eigenstate using the shooting method.

        Returns:
            tuple: (energy, normalized_wavefunction)
        """
        if tol is None:
            tol = self.tol

        # Bracketing
        bracket = self._bracket_eigenvalue(energy_level, E_min, E_max, tol=tol)

        # Find eigenvalue
        # Too inconsistent to just call bisect without bracketing
        #E = bisect(self._boundary, E_min, E_max, xtol=tol)
        E = bisect(self._boundary, bracket[0], bracket[1], xtol=tol)

        # Retrieve cached wavefunction and normalize
        psi = self.last_psi
        norm = np.sqrt(np.trapezoid(psi**2, self.x))
        return E, psi


#################################################################################

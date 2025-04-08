import numpy as np

def setup_hydrogen_problem(n=1, l=0, Z=1, xmin=-8.0, xmax=np.log(100.0), mesh=1260):
    """
    Initialize arrays and parameters for hydrogen atom Schrodinger equation solver.
    Returns a dictionary containing all precomputed quantities.
    """
    # Logarithmic mesh
    x, dx = np.linspace(xmin, xmax, mesh+1, retstep=True)
    r = np.exp(x) / Z

    # Precompute quantities
    params = {
        'n': n,
        'l': l,
        'Z': Z,
        'mesh': mesh,
        'dx': dx,
        'r': r,
        'r2': r**2,
        'ddx12': (dx**2)/12.0,
        'lnhfsq': (l+0.5)**2,
        'x': x,
        'v0': -2*Z/r, # Coulomb potential
        'nodes': n - l - 1  # Expected number of nodes
    }

    # Energy bounds
    veff = params['v0'] + params['lnhfsq']/params['r2'] # Effective potential
    params.update({
        'e_lower': np.min(veff),
        'e_upper': params['v0'][-1],
        'e_guess': 0.5*(np.min(veff) + params['v0'][-1])
    })

    return params

def hydrogen_atom(params, max_iter=200, tol=1e-10):
    """
    Solves the radial Schrodinger equation for hydrogen-like atoms.
    Args:
        params: Dictionary from setup_hydrogen_problem()
        max_iter: Maximum iterations
        tol: Energy convergence tolerance
    Returns:
        (energy, iterations)
    """
    # Unpack parameters
    n, l, Z = params['n'], params['l'], params['Z']
    mesh, dx = params['mesh'], params['dx']
    r, r2, x = params['r'], params['r2'], params['x']
    ddx12, lnhfsq = params['ddx12'], params['lnhfsq']
    nodes = params['nodes']
    v0 = params['v0']

    # Initialize energy bounds
    e = params['e_guess']
    e_lower, e_upper = params['e_lower'], params['e_upper']
    # Initial large number for energy correction
    de = 1e10
    converged = False

    for iteration in range(1, max_iter+1):
        if abs(de) <= tol:
            converged = True
            break

        # Find classical turning point
        f, f_10, icl = compute_f_and_icl_atom(
            mesh, ddx12, r2, lnhfsq, v0, e
        )

        # Safety check
        if icl < 1 or icl >= mesh - 1:
            raise ValueError(f"Invalid classical turning point icl={icl}")

        # Initialize wavefunction
        psi = initialize_wavefunction(r, l, Z, mesh)

        # Outward integration
        psi_icl, ncross, f_10 = outward_integration(psi, f, f_10, icl)

        # Make sure energy bounds has correct number of nodes before inward integration
        if ncross != nodes:
            e_lower, e_upper = adjust_energy_bounds(ncross, nodes, e, e_lower, e_upper)
            e = 0.5*(e_lower + e_upper)
            continue

        # Set boundary conditions for inward integration
        psi[-1] = dx
        psi[-2] = f_10[-1] * psi[-1] / f[-2]

        # Full integration
        inward_integration(psi, f, icl, mesh, f_10)
        scale_normalize_atom1(psi, psi_icl, icl, x, r2)

        # Energy update
        e, e_lower, e_upper, de = update_energy(
            icl, f, psi, dx, ddx12, e, e_lower, e_upper
        )

    if not converged:
        raise RuntimeError(
            f"Not converged after {max_iter} iterations. "
            f"Final de={de:.2e}, nodes expected (n={n}, l={l})={nodes}, found={ncross}"
        )

    return e, iteration

# Helper functions
def initialize_wavefunction(r, l, Z, mesh):
    """Initialize radial wavefunction at first two points."""
    psi = np.zeros(mesh+1)
    l2 = 2*l + 2  # denominator
    for i in [0, 1]:
        psi[i] = (r[i]**(l+1)) * (1 - (2*Z*r[i])/l2) / np.sqrt(r[i])
    return psi

def adjust_energy_bounds(ncross, nodes, e, e_lower, e_upper):
    """Adjust energy bounds based on node count."""
    if ncross > nodes:
        return e_lower, e  # New upper bound
    else:
        return e, e_upper  # New lower bound



################################################################################
################################ INTEGRATION ###################################

def outward_integration(psi, f, f_10, icl):
    ncross = 0
    for i in range(1, icl):
        psi[i+1] = (f_10[i] * psi[i] - f[i-1] * psi[i-1]) / f[i+1]
        ncross += (psi[i] * psi[i+1] < 0.0)  # Boolean to int
    return psi[icl], ncross, f_10

def inward_integration(psi, f, icl, mesh, f_10):
    for i in range(mesh-1, icl, -1):
        psi[i-1] = (f_10[i] * psi[i] - f[i+1] * psi[i+1]) / f[i-1]
        if abs(psi[i-1]) > 1e10:
            psi[i-1:-2] /= psi[i-1]


def scale_normalize_atom1(psi, psi_icl, icl, x, r2):
    # Match wavefunction at icl and normalize
    scaling_factor = psi_icl / psi[icl]
    psi[icl:] *= scaling_factor

    norm = np.sqrt(np.trapezoid(psi**2 * r2, x))
    psi /= norm


def update_energy(icl, f, psi, dx, ddx12, e, e_lower, e_upper):
    i = icl
    psi_cusp = (psi[i-1] * f[i-1] + psi[i+1] * f[i+1] + 10 * f[i] * psi[i]) / 12.0
    dfcusp = f[i] * (psi[i] / psi_cusp - 1.0)
    de = 0.5 * dfcusp / ddx12 * (psi_cusp ** 2) * dx

    if de > 0:
        e_lower = e
    elif de < 0:
        e_upper = e
    e += de
    e = max(min(e, e_upper), e_lower)
    return e, e_lower, e_upper, de



def compute_f_and_icl_atom(mesh, ddx12, r2, lnhfsq, vpot, e):
    f = ddx12 * (lnhfsq + r2 * (vpot - e))

    # Handle zeros to avoid sign issues
    f = np.where(f == 0.0, 1e-20, f)

    # Find classical turning point (last sign change)
    sign_changes = np.where(np.diff(np.sign(f)))[0]
    icl = sign_changes[-1] + 1 if len(sign_changes) > 0 else mesh - 1

    # f as required by Numerov
    f = 1.0 - f

    return f, 12.0 - 10.0 * f, icl

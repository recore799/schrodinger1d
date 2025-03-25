import numpy as np
import sys

def do_mesh(mesh, zmesh, xmin, dx, rmax, r, sqr, r2):
        for i in range(mesh + 1):
                x = xmin + dx * i
                r[i] = np.exp(x) / zmesh
                sqr[i] = np.sqrt(r[i])
                r2[i] = r[i] * r[i]

def init_energy_bounds(mesh, sqlhf, r2, vpot):
    """Initialize energy bounds (elw and eup)."""
    eup = vpot[mesh]
    sqlhf_over_r2 = sqlhf / r2[:mesh+1]
    sum_terms = sqlhf_over_r2 + vpot[:mesh+1]
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
    for i in range(mesh-1, icl, -1):
        y[i-1] = ((12.0 - f[i] * 10.0) * y[i] - f[i+1] * y[i+1]) / f[i-1]
        if y[i-1] > 1e10:
            y[i-1:mesh+1] /= y[i-1]

    # DEBUG: Inward integration steps
    # print(f"Inward y[icl]:{y[icl]:.6f}")
    # if kkk == 0:  # First iteration debug
    #     print(f"First inward step: y[mesh]={y[mesh]:.6f}, y[mesh-1]={y[mesh-1]:.6f}")

def rescale_and_normalize(mesh, icl, y, fac, r2, dx):
    """Rescale and normalize the wavefunction."""
    # DEBUG: Rescaling parameters
    # print(f"fac before rescaling: {fac:.6f}, y[icl]: {y[icl]:.6f}")

    scaling_factor = fac / y[icl]
    y[icl:mesh+1] *= scaling_factor

    # DEBUG: Normalization parameters
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
    de = dfcusp / ddx12 * (ycusp ** 2) * dx

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

def solve_sheq(n, l, zeta, mesh, dx, r, sqr, r2, vpot, y):
    """Solve the Schrödinger equation using the Numerov method."""
    eps = 1e-10
    n_iter = 100

    ddx12 = dx**2 / 12.0
    sqlhf = (l + 0.5)**2
    x2l2 = 2 * l + 2

    # Initialize energy bounds
    elw, eup = init_energy_bounds(mesh, sqlhf, r2, vpot)
    if eup - elw < eps:
        sys.stderr.write(f"ERROR: solve_sheq: eup={eup} and elw={elw} are too close.\n")
        sys.exit(1)

    e = (elw + eup) * 0.5
    de = 1e10  # Initial large value
    converged = False

    # DEBUG: Check energy bounds
    # print(f"Initial energy bounds: elw={elw:.6f}, eup={eup:.6f}, e={e:.6f}")

    for kkk in range(n_iter):
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

        # Rescale and normalize
        rescale_and_normalize(mesh, icl, y, fac, r2, dx)

        # Update energy
        e, elw, eup, de = update_energy(icl, f, y, ddx12, dx, e, elw, eup)
        # DEBUG: Energy updates
        # print(f"Energy update: e={e:.6f}, de={de:.6f}, elw={elw:.6f}, eup={eup:.6f}")

    if not converged:
        error_msg = (f"ERROR: solve_sheq not converged after {n_iter} iterations.\n"
                     f"Final de={de:.2e}, e={e:.6f}, nodes expected={nodes}, found={ncross}")
        sys.stderr.write(error_msg + "\n")
        sys.exit(1)
    else:
        print(f"Convergence achieved at iteration {kkk+1}, de = {de:.2e}")

    return e

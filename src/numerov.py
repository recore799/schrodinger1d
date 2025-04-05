import numpy as np
import matplotlib.pyplot as plt


################################################################################
###################### Current Numerov Harmonic Oscillator #####################
def harmonic_oscillator(nodes=0, xmax=10.0, mesh=500, max_iter=1000, tol=1e-20):
    # Initialize grid and potential
    x, dx = np.linspace(0, xmax, mesh, retstep=True)
    vpot = 0.5 * x**2  # Harmonic oscillator potential

    # Initial energy bounds
    e_lower = np.min(vpot)
    e_upper = np.max(vpot)

    # Bisection loop
    for iter in range(max_iter):
        e = 0.5 * (e_lower + e_upper)

        # Find icl
        f, icl = f_and_icl_ho(vpot, e, dx)

        # Initialize wavefunction
        psi = np.zeros(mesh)
        # Initialize wavefunction based on parity
        if nodes % 2:
            # Odd
            psi[0] = 0.0
            psi[1] = dx
        else:
            # Even
            psi[0] = 1.0
            psi[1] = 0.5 * (12.0 - 10.0 * f[0]) * psi[0] / f[1]

        # Outward integration
        psi_icl, ncross, f_10 = outward_integration(psi, f, icl)

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

        psi[-1] = dx
        psi[-2] = f_10[-1] * psi[-1] / f[-2]

        # Inward integration (from x=xmax to icl)
        inward_integration(psi, f, icl, mesh, f_10)

        scale_normalize(psi, psi_icl, icl, x)

        # Compute derivative discontinuity
        djump = (psi[icl+1] + psi[icl-1] - (14.0 - 12.0 * f[icl]) * psi[icl]) / dx

        # Check convergence
        if (e_upper - e_lower) < tol:
            # print(f"Reached convergence after {iter} iterations")
            break

        # Adjust energy based on djump sign
        if djump * psi[icl] > 0.0:
            e_upper = e
        else:
            e_lower = e

    return e


def f_and_icl_ho(vpot, e, dx):
    # Init icl
    icl = -1

    # Setup simple f function to find icl
    f = 2*(vpot - e) * (dx ** 2 / 12)

    # Avoid division by zero
    f = np.where(f == 0.0, 1e-20, f)

    # Classical turning point is the last sign change in f + 1
    sign_changes = np.where(np.diff(np.sign(f)))[0]
    icl = sign_changes[-1] + 1

    # f as required by numerov
    f = 1 - f
    return f, icl

#################################################################################


def hydrogen_atom(n=0, l=0, Z=1, xmin=-8.0, xmax=np.log(100.0), mesh=1260, max_iter=1000, tol=1e-10):
    # Logarithmic mesh in x
    x, dx = np.linspace(xmin, xmax, mesh, retstep=True)

    # Physical mesh in r
    r = np.exp(x) / Z

    # Step sizes in r (for integrations or derivatives)
    # dr = r * dx

    v0 = - Z / r
    veff = v0 + ((l + 0.5)/r)**2

    e_lower = np.min(veff)
    e_upper = v0[-1]

    e = 0.5 * (e_lower + e_upper)

    # print(f"Initial energy bounds: e_lower={e_lower:.6f}, e_upper={e_upper:.6f}")

    iter = 0
    # Bisection loop
    for iter in range(max_iter):
        e = 0.5 * (e_lower + e_upper)

        # Find icl
        f, icl = compute_f_and_icl_atom(mesh, dx, r, v0, l, e)

        # Initialize wavefunction
        psi = np.zeros(mesh)

        # Initialize wavefunction
        psi[0] = (r[0] ** (l + 1)) * (1 - (2 * Z * r[0]) / (2*l + 2)) / np.sqrt(r[0])
        psi[1] = (r[1] ** (l + 1)) * (1 - (2 * Z * r[1]) / (2*l + 2)) / np.sqrt(r[1])

        # Outward integration
        psi_icl, ncross, f_10 = outward_integration(psi, f, icl)

        # Adjust energy bounds based on node count
        nodes = n-l-1
        if ncross != nodes:
            if ncross > nodes:
                e_upper = e
            else:
                e_lower = e
            continue  # Skip inward integration if node count is wrong

        psi[-1] = dx
        psi[-2] = f_10[-1] * psi[-1] / f[-2]

        # Inward integration (from x=xmax to icl)
        inward_integration(psi, f, icl, mesh, f_10)

        scale_normalize(psi, psi_icl, icl, r)

        # Compute derivative discontinuity
        djump = (psi[icl+1] + psi[icl-1] - (14.0 - 12.0 * f[icl]) * psi[icl]) / dx

        # Check convergence
        if (e_upper - e_lower) < tol:
            # print(f"Reached convergence after {iter} iterations")
            break

        # Adjust energy based on djump sign
        if djump * psi[icl] > 0.0:
            e_upper = e
        else:
            e_lower = e

    return e


def compute_f_and_icl_atom(mesh, dx, r, vpot, l, e):
    # Precompute common terms
    r2 = r ** 2
    sqlhf = (l + 0.5) ** 2  # (l + 0.5)^2
    ddx12 = dx ** 2 / 12.0   # Δx² / 12

    # Compute f in one shot (vectorized)
    f = ddx12 * (sqlhf + 2 * r2 * (vpot - e))

    # Handle zeros to avoid sign issues
    f = np.where(f == 0.0, 1e-20, f)

    # Find classical turning point (last sign change)
    sign_changes = np.where(np.diff(np.sign(f)))[0]
    icl = sign_changes[-1] + 1 if len(sign_changes) > 0 else mesh - 1

    # f as required by Numerov
    # f = 1.0 - f

    return 1.0 - f, icl



################################################################################
################################ INTEGRATION ###################################
def outward_integration(psi, f, icl):
    ncross = 0
    f_10 = 12.0 - 10.0 * f  # Precompute
    for i in range(1, icl):
        psi[i+1] = (f_10[i] * psi[i] - f[i-1] * psi[i-1]) / f[i+1]
        ncross += (psi[i] * psi[i+1] < 0.0)  # Boolean to int
    return psi[icl], ncross, f_10

def inward_integration(psi, f, icl, mesh, f_10):
    # f_10 = 12.0 - 10.0 * f # Precompute
    # Inward integration in [xmax, icl]
    for i in range(mesh-2, icl, -1):
        psi[i-1] = (f_10[i] * psi[i] - f[i+1] * psi[i+1]) / f[i-1]

def scale_normalize(psi, psi_icl, icl, r):
    # Match wavefunction at icl and normalize
    scaling_factor = psi_icl / psi[icl]
    psi[icl:] *= scaling_factor
    norm = np.sqrt(np.trapezoid(psi**2, r))  # Symmetric normalization
    psi /= norm


###############################################################################

import timeit

def test_hydrogen_levels():
    print("\nHydrogen Atom Energy Levels (n=1 to 10, l=0):")
    print("--------------------------------------------")
    print(" n | Computed E (a.u.) | Expected E (a.u.) | Error")
    print("---|-------------------|-------------------|-------")

    Z = 1
    l = 0
    xmin = -8.0
    xmax = np.log(100.0)
    mesh = 1260

    for n in range(1, 7):
        start_time = timeit.default_timer()
        e_computed = hydrogen_atom(n, l, Z, xmin, xmax, mesh)
        e_expected = -Z**2 / (2*n**2)
        error = abs(e_computed - e_expected)
        elapsed = timeit.default_timer() - start_time

        print(f"{n:2} | {e_computed:.8f}        | {e_expected:.8f}        | {error:.2e} | {elapsed:.4f} sec")

    print("\n------ Performance Summary ------")
    print(f"Mesh size: {mesh}, r_max: {np.exp(xmax):.1f} a.u.")


if __name__ == "__main__":
    test_hydrogen_levels()

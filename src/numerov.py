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








def V_ho(x):
    return x**2

def main0():

    integrator = numerov0(V=V_ho, xL=-5, xR=5)

    # Find the ground state (0 nodes)
    E0, psi0 = integrator.solve_state(energy_level=0, E_min=0, E_max=10)
    nodes0 = integrator.count_nodes(psi0)

    print(f"Ground state energy: {E0:.6f}")
    print(f"Ground state node count: {nodes0}")

    # Plot the ground state wavefunction
    plt.figure(figsize=(8, 4))
    plt.plot(integrator.x, psi0, label='Ground State (0 nodes)')
    plt.title('Ground State Wavefunction')
    plt.xlabel('x')
    plt.ylabel(r'$\psi(x)$')
    plt.legend()
    plt.grid(True)
    plt.show()


def vpot(zeta,r):
    return -2 * zeta / r
import numpy as np

def main1():
    # Test for both zeta values
    for zeta in [1, 2]:
        print(f"\n===== Testing for zeta = {zeta} =====")

        # Parameters
        mesh = 1000
        xmin = -8.0          # r_min = e^{-8} ≈ 3.3e-5 a.u.
        xmax = np.log(100.0)  # r_max = 100 a.u.
        Z = 1.0               # Hydrogen

        # Logarithmic mesh in x
        x, dx = np.linspace(xmin, xmax, mesh, retstep=True)

        # Physical mesh in r
        r = np.exp(x) / Z

        # Step sizes in r (for integrations or derivatives)
        dr = r * dx

        # Compute the potential
        vpot_arr = vpot(zeta, r)

        # Test different quantum numbers
        max_n = 3  # Maximum principal quantum numb er to test
        for n in range(1, max_n + 1):
            # For each n, l can range from 0 to n-1
            for l in range(0, n):
                # Reinitialize wavefunction array for each calculation
                y.fill(0)

                e = solve_sheq(n, l, zeta, mesh, dx, r, sqr, r2, vpot_arr, y)

                # Theoretical energy for comparison (in atomic units)
                theoretical_e = -zeta**2 / (2 * n**2)
                error = abs(e - theoretical_e)

                print(f"n={n}, l={l}: Computed E = {e:.6f}, "
                      f"Theoretical E = {theoretical_e:.6f}, "
                      f"Error = {error:.2e}")


if __name__ == "__main__":
    main1()

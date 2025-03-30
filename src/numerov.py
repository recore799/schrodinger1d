import numpy as np
from scipy.optimize import bisect

import matplotlib.pyplot as plt

import sys

### BASIC NUMEROV ###
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
        print(f"Eigenenergy interval found ({bracket[0]:.6f},{bracket[1]:.6f})")
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
        return E, psi / norm

########################################################################



### HYDROGEN ATOM ###

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

    return ncross, y[icl]

def inward_integration(mesh, icl, f, y, dx):
    """Perform inward integration."""
    y[mesh] = dx
    y[mesh-1] = (12.0 - f[mesh] * 10.0) * y[mesh] / f[mesh-1]
    for i in range(mesh-1, icl, -1):
        y[i-1] = ((12.0 - f[i] * 10.0) * y[i] - f[i+1] * y[i+1]) / f[i-1]
        if y[i-1] > 1e10:
            y[i-1:mesh+1] /= y[i-1]

def rescale_and_normalize(mesh, icl, y, fac, r2, dx):
    """Rescale and normalize the wavefunction."""
    scaling_factor = fac / y[icl]
    y[icl:mesh+1] *= scaling_factor

    norm_sq = np.sum(y[1:mesh+1]**2 * r2[1:mesh+1] * dx)
    norm = np.sqrt(norm_sq)
    y[:mesh+1] /= norm

def update_energy(icl, f, y, ddx12, dx, e, elw, eup):
    """Compute the cusp condition and update the energy."""
    i = icl
    ycusp = (y[i-1] * f[i-1] + y[i+1] * f[i+1] + 10 * f[i] * y[i]) / 12.0
    dfcusp = f[i] * (y[i] / ycusp - 1.0)
    de = dfcusp / ddx12 * (ycusp ** 2) * dx

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

    for kkk in range(n_iter):
        if abs(de) <= eps:
            converged = True
            break

        # Compute f and find icl
        f, icl = compute_f_and_icl(mesh, ddx12, sqlhf, r2, vpot, e)

        if icl < 0 or icl >= mesh - 2:
            sys.stderr.write(f"ERROR: solve_sheq: icl={icl} out of range (mesh={mesh})\n")
            sys.exit(1)

        f[:] = 1.0 - f[:]

        # Initialize wavefunction
        y[0] = (r[0] ** (l + 1)) * (1 - (2 * zeta * r[0]) / x2l2) / sqr[0]
        y[1] = (r[1] ** (l + 1)) * (1 - (2 * zeta * r[1]) / x2l2) / sqr[1]

        # Outward integration
        nodes = n - l - 1
        ncross, fac = outward_integration(icl, f, y)
        if ncross != nodes:
            if ncross > nodes:
                eup = e
            else:
                elw = e
            e = (eup + elw) * 0.5
            continue

        # Inward integration
        inward_integration(mesh, icl, f, y, dx)

        # Rescale and normalize
        rescale_and_normalize(mesh, icl, y, fac, r2, dx)

        # Update energy
        e, elw, eup, de = update_energy(icl, f, y, ddx12, dx, e, elw, eup)

    if not converged:
        error_msg = (f"ERROR: solve_sheq not converged after {n_iter} iterations.\n"
                     f"Final de={de:.2e}, e={e:.6f}, nodes expected={nodes}, found={ncross}")
        sys.stderr.write(error_msg + "\n")
        sys.exit(1)
    else:
        print(f"Convergence achieved at iteration {kkk+1}, de = {de:.2e}")

    return e

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

        zmesh = 1
        rmax = 100
        xmin = -8.0
        dx = 0.01

        # Calculate mesh as integer to avoid float indices
        mesh = int((np.log(zmesh * rmax) - xmin) / dx)
        # Ensure mesh is at least 0 to prevent negative array sizes
        mesh = max(mesh, 0)

        # Initialize arrays with mesh+1 elements
        r = np.zeros(mesh + 1, dtype=float)
        sqr = np.zeros(mesh + 1, dtype=float)
        r2 = np.zeros(mesh + 1, dtype=float)
        y = np.zeros(mesh + 1, dtype=float)  # Wavefunction array

        # Generate the logarithmic mesh
        do_mesh(mesh, zmesh, xmin, dx, rmax, r, sqr, r2)

        # Compute the potential
        vpot_arr = vpot(zeta, r)

        # Test different quantum numbers
        max_n = 3  # Maximum principal quantum number to test
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

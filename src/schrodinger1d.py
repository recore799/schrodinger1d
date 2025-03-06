import numpy as np
from scipy.optimize import bisect

class SchrodingerSolver:
    def __init__(self, V, xL=-5, xR=5, n=201, tol=1e-6):
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

    def _bracket_eigenvalue(self, target_nodes, E_min, E_max, num=50, tol=None):
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

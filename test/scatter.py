import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import solve_atom, solve_atom_bisection, init_mesh



import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

def plot_orbital(n, l, m, rmax=20.0, mesh=800, nsamples=50000):
    # Solve radial part
    e, iterations, psi_r = solve_atom(n=n, l=l, rmax=rmax, mesh=mesh)
    x, rgrid, dx = init_mesh(rmax, mesh, Z=1)
    
    # Avoid r=0 to prevent division by zero
    R = psi_r[1:] / rgrid[1:]
    rgrid = rgrid[1:]
    
    # Normalize radial function
    R /= np.sqrt(np.trapz(np.abs(R)**2 * rgrid**2, rgrid))
    
    # Make interpolation function
    R_interp = lambda r: np.interp(r, rgrid, R)
    
    # Sample spherical coordinates
    U = np.random.rand(nsamples)
    r = rmax * U**(1/3)                     # uniform in volume
    theta = np.arccos(2*np.random.rand(nsamples)-1)
    phi = 2*np.pi*np.random.rand(nsamples)
    
    # Evaluate wavefunction
    Ylm = sph_harm(m, l, phi, theta)
    psi = R_interp(r) * Ylm
    prob = np.abs(psi)**2
    
    # Weighted resampling (keep more high-probability points)
    prob /= prob.sum()
    idx = np.random.choice(np.arange(nsamples), size=nsamples//4, p=prob)
    
    r, theta, phi = r[idx], theta[idx], phi[idx]
    
    # Convert to Cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Plot scatter
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=1, alpha=0.3)
    ax.set_title(f"Hydrogen orbital n={n}, l={l}, m={m}")
    ax.set_axis_off()
    plt.show()

plot_orbital(3, 2, 0)

import sys
import os
import timeit
import numpy as np

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov_debug import (
    # hydrogen_atom,  # Current implementation
    solve_sheq
)

Z = 1
l = 0
xmin = -8.0
xmax = np.log(100.0)
# mesh = 1260
mesh = 1260
# Logarithmic mesh in x
x, dx = np.linspace(xmin, xmax, mesh+1, retstep=True)

# Physical mesh in r
r = np.exp(x) / Z

# Precompute common terms
r2 = r**2
sqr = np.sqrt(r)

vpot = - Z/r

y = np.zeros(mesh+1)

n = 1

e, iterations = solve_sheq(n, l, Z, mesh, dx, r, sqr, r2, vpot, y)

print(e)

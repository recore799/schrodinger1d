import sys
import os
import timeit
import numpy as np

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import (
    hydrogen_atom,  # Current implementation
    # solve_sheq
)


n=1
l=0
Z=1
xmin=-8.0
xmax=np.log(100.0)
mesh=1260

e_numerov, iterations = hydrogen_atom(n, l, Z, xmin, xmax, mesh)

print(e_numerov)

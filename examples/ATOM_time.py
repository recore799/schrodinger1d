import sys
import os
import timeit
import numpy as np

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import (
    hydrogen_atom  # Current implementation
)


print("\n Hydrogen Atom:")
start_time = timeit.default_timer()
n = 3
l = 0
Z = 1

# Parameters
xmin = -8.0          # r_min = e^{-8} ≈ 3.3e-5 a.u.
xmax = np.log(100.0)  # r_max = 100 a.u.
mesh = 1260


e = hydrogen_atom(n, l, Z, xmin, xmax, mesh)
print(f"State n={n}, l={l} : e: {e:.8f}")
current_time1 = timeit.default_timer() - start_time

# ---- Summary ----
print("\n------ Performance Summary ------")
print(f"Current: {current_time1:.4f} sec")


# print(f"10 States: {current_time:.4f} sec")

# n_states = 10
# energies = np.zeros(n_states)

# # ---- (2) Current Fast Implementation ----
# print("\nOnly one state:")
# start_time = timeit.default_timer()
# for i in range(n_states):
#     energies[i] = hydrogen_atom(n=i)
#     print(f"State {i}: {energies[i]:.8f}")
# current_time = timeit.default_timer() - start_time



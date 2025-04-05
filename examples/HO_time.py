import sys
import os
import timeit
import numpy as np

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import (
    harmonic_oscillator,  # Current fast implementation
)

from numerov_debug import harmonic_oscillator_test

# Test parameters
n_states = 10  # Number of states to compute
# n_runs = 5     # Number of runs for averaging

# Initialize energy arrays
# energies_class = np.zeros(n_states)  # Unused (commented out)
energies_current = np.zeros(n_states)
energies_modular = np.zeros(n_states)

# ---- (1) Current Fast Implementation ----
print("\nCurrent implementation (fast, non-modular):")
start_time = timeit.default_timer()
for i in range(n_states):
    energies_current[i] = harmonic_oscillator_test(nodes=i)
    print(f"State {i}: {energies_current[i]:.8f}")
current_time = timeit.default_timer() - start_time

# ---- Summary ----
print("\n------ Performance Summary ------")
# print(f"Class-based: {class_time:.4f} sec")  # Uncomment if needed
print(f"Current (non-modular): {current_time:.4f} sec")

import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import numerov0, harmonic_oscillator

# Number of runs for averaging
n_runs = 10

# Initialize energy arrays
energies1 = np.zeros(10)
energies2 = np.zeros(10)

# --- First Implementation ---
print("-------------------- First Implementation ----------")
start_time1 = time.time()

for i in range(10):
    energies1[i] = harmonic_oscillator(nodes=i)
    print(f"Energy for state {i}: {energies1[i]:.8f}")

end_time1 = time.time()
print(f"Total time (Implementation 1): {end_time1 - start_time1:.6f} seconds")

# --- Second Implementation ---
print("\n-------------------- Second Implementation ----------")
integrator = numerov0(V=lambda x: x**2)  # Define potential

start_time2 = time.time()

for i in range(10):
    energies2[i] = integrator.solve_state(energy_level=i, E_min=0, E_max=10)
    print(f"Energy for state {i}: {energies2[i]:.8f}")

end_time2 = time.time()
print(f"Total time (Implementation 2): {end_time2 - start_time2:.6f} seconds")

# --- Summary ---
print("\n----------------- Performance Summary -----------------")
print(f"Implementation 1 (harmonic_oscillator): {end_time1 - start_time1:.6f} s")
print(f"Implementation 2 (numerov0): {end_time2 - start_time2:.6f} s")
print(f"Speed ratio (Impl2/Impl1): {(end_time2 - start_time2)/(end_time1 - start_time1):.2f}x")

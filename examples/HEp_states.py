import timeit
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import solve_atom, solve_atom_bisection

states = 2
l = 0

print(f"Testing ionized Helium atom for {states} states with l={l}")
print("-" * 80)
print(f"{'n':>3} {'Energy (Ry)':>15} {'Theoretical':>15} {'Error':>15} {'Iterations':>12} {'Time (ms)':>12}")
print("-" * 80)

for n in range(0,states):
    state = n + 1
    theoretical_energy = - 4 / state**2

    # Time the function execution
    # timer = timeit.Timer(lambda: solve_atom_bisection(n=state, l=l))
    timer = timeit.Timer(lambda: solve_atom(n=state, l=l, Z=2))
    runs = 3  # Number of runs to average
    time_taken = timer.timeit(number=runs) / runs * 1000  # Convert to milliseconds

    # Get the actual result
    # e, iterations = solve_atom_bisection(n=state, l=l)
    e, iterations, psi = solve_atom(n=state, l=l, Z=2)

    error = abs(e - theoretical_energy)

    print(f"{state:3d} {e:15.8f} {theoretical_energy:15.6f} {error:15.4e} {iterations:12d} {time_taken:12.3f}")

print("-" * 80)

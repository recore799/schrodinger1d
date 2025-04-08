import timeit
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import solve_atom, solve_atom_bisection

states = 5
l = 0

print(f"Comparing hydrogen atom solvers for {states} states with l={l}")
print("-" * 105)
print(f"{'n':>3} {'Method':>15} {'Energy (Ry)':>15} {'Theoretical':>15} {'Error':>12} {'Iterations':>12} {'Time (ms)':>12}")
print("-" * 105)

for n in range(states):
    state = n + 1
    theoretical_energy = -1 / state**2

    # Test solve_atom (Perturbation updates)
    timer_pert = timeit.Timer(lambda: solve_atom(n=state, l=l))
    time_pert = timer_pert.timeit(number=3) / 3 * 1000  # avg in ms
    e_pert, iter_pert = solve_atom(n=state, l=l)
    error_pert = abs(e_pert - theoretical_energy)

    # Test solve_atom_bisection
    timer_bisect = timeit.Timer(lambda: solve_atom_bisection(n=state, l=l))
    time_bisect = timer_bisect.timeit(number=3) / 3 * 1000  # avg in ms
    e_bisect, iter_bisect = solve_atom_bisection(n=state, l=l)
    error_bisect = abs(e_bisect - theoretical_energy)

    # Print results for both methods
    print(f"{state:3d} {'Perturbation':>15} {e_pert:15.10f} {theoretical_energy:15.6f} "
          f"{error_pert:.4e} {iter_pert:12d} {time_pert:12.3f}")

    print(f"{' ':3} {'Bisection':>15} {e_bisect:15.10f} {theoretical_energy:15.6f} "
          f"{error_bisect:.4e} {iter_bisect:12d} {time_bisect:12.3f}")

    print("-" * 105) if state < states else print("=" * 105)

# Print summary statistics
pert_avg_time = sum(time_pert for _ in range(states)) / states
bisect_avg_time = sum(time_bisect for _ in range(states)) / states
print(f"\nSummary: Perturbation average time = {pert_avg_time:.3f} ms | "
      f"Bisection average time = {bisect_avg_time:.3f} ms")

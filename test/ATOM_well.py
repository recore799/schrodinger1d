import timeit
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import solve_atom, solve_atom_bisection

state = 2
l = 0

print(f"Testing hydrogen_atom for {state} states with l={l}")
print("-" * 80)
print(f"{'n':>3} {'Energy (Ry)':>15} {'Iterations':>12}")
print("-" * 80)


e, iterations, psi = solve_atom(n=state, l=l)

print(f"{state:3d} {e:15.8f} {iterations:12d}")


print("-" * 80)

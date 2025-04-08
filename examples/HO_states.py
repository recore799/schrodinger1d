import timeit
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import harmonic_oscillator

def harmonic_oscillator_table(states=5):
    """Generate formatted table of harmonic oscillator energies"""
    print("\nTesting harmonic_oscillator for {} states".format(states))
    print("-" * 78)
    print("{:>3} {:>15} {:>15} {:>15} {:>12} {:>12}".format(
        "n", "Energy (ħω)", "Theoretical", "Error", "Iterations", "Time (ms)"))
    print("-" * 78)

    for i in range(states):
        # Theoretical energy E_n = (n + 0.5)
        theoretical = i + 0.5

        # Time the calculation
        timer = timeit.Timer(lambda: harmonic_oscillator(nodes=i))
        time_ms = timer.timeit(number=10) * 100  # Average time in ms

        # Get the actual result
        e, iterations = harmonic_oscillator(nodes=i)
        error = abs(e - theoretical)

        print("{:3d} {:15.8f} {:15.8f} {:15.2e} {:12d} {:12.3f}".format(
            i, e, theoretical, error, iterations, time_ms))

    print("-" * 78)

# Example usage:
if __name__ == "__main__":
    harmonic_oscillator_table(states=6)

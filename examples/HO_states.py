import timeit
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import harmonic_oscillator

for i in range(5):
    e = harmonic_oscillator(nodes = i)
    print(e)

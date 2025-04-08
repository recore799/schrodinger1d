from numerov_test import harmonic_oscillator


for i in range(5):
    e = harmonic_oscillator(nodes = i)
    print(e)

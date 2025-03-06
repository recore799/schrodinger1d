# 1D Schrödinger Equation Solver

A Python package for solving the 1D Schrödinger equation numerically using the **Numerov method**. This package provides tools to compute wavefunctions and energy eigenvalues for arbitrary potentials, with built-in support for the harmonic oscillator and hydrogen atom potentials.

## Features

- **Numerov Integration**: Solve the 1D Schrödinger equation for any potential.
- **Harmonic Oscillator**: Predefined potential for the quantum harmonic oscillator.
- **Hydrogen Atom**: Predefined potential for the radial Schrödinger equation of the hydrogen atom.
- **Wavefunction Visualization**: Plot wavefunctions and probability densities.
- **Extensible**: Easily add new potentials and solvers.

### Example Results

#### Harmonic Oscillator Eigenfunctions (First 5 States)
![Harmonic Oscillator Eigenfunctions](examples/harmonic_oscillator_eigenfunctions.png)

**Note:** Wavefunctions are vertically offset by their energy eigenvalues ($E_n = 2n+1$).


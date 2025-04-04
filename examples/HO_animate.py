#!/usr/bin/env python3

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import numerov0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the harmonic oscillator potential.
def harmonic_oscillator(x):
    return x**2

def main():
    # Create an instance of your solver (adjust the class name if needed)
    integrator = numerov0(V=harmonic_oscillator, xL=-5, xR=5, n=501, tol=1e-7)

    num_states = 10
    E = np.zeros(num_states)
    psi = np.zeros((num_states, len(integrator.x)))

    # Compute eigenstates using your solver.
    for n in range(num_states):
        E[n], psi[n] = integrator.solve_state(energy_level=n, E_min=0, E_max=10)

    # Choose coefficients for the linear combination (here, equal weighting).
    coeffs = np.ones(num_states) / np.sqrt(num_states)

    # Setup the figure for animation.
    fig, ax = plt.subplots()
    # Start with the initial linear combination at t=0.
    psi_initial = np.sum(coeffs[:, None] * psi, axis=0)
    line, = ax.plot(integrator.x, psi_initial, 'b-', lw=2)
    ax.set_xlim(integrator.x[0], integrator.x[-1])
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("Re(ψ(x,t))")
    title = ax.set_title("Time t = 0.00")

    # Define the animation update function.
    def update(t):
        # Calculate the time-dependent wavefunction:
        # ψ(x,t) = ∑ coeffs[n] * ψ_n(x) * exp(-i*E[n]*t)
        psi_t = np.sum(coeffs[:, None] * psi * np.exp(-1j * E[:, None] * t), axis=0)
        # Update the plot with the real part.
        line.set_ydata(np.real(psi_t))
        title.set_text(f"Time t = {t:.2f}")
        return line, title

    # Create an animation over the desired time interval.
    # Here, we go from t = 0 to t = 20 in 400 frames.
    frames = np.linspace(0, 20, 400)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    # Save the animation to an MP4 file with 30 frames per second.
    #ani.save("traveling_wave.mp4", writer="ffmpeg", fps=60)
    plt.show()


if __name__ == '__main__':
    main()

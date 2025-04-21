# Numerov Algorithm

A Python implementation of the Numerov algorithm for solving radial Schrödinger equations for quantum systems.


## DOCUMENTATION

[Metodo de Numerov](docs/numerov.pdf)

## Project Structure

```
.
├── src/                   # Working code
│   ├── numerov.py
│
├── examples/              # Fully functional examples
│   ├── ATOM_states.py     # Hydrogen atom energy levels
│   └── HO_states.py       # Harmonic oscillator levels
│
├── test/                  # Under development - may be broken
│   ├── numerov_test.py
│   └── test_hydrogen.py   # Dev space
│   └── storage.py         # Old implementations
│
└── docs/                  # Documentation

```

## Example Outputs

### Hydrogen Atom Energy Levels (`ATOM_states.py`)

```text
Testing hydrogen_atom for 6 states with l=0
--------------------------------------------------------------------------------
  n     Energy (Ry)     Theoretical           Error   Iterations    Time (ms)
--------------------------------------------------------------------------------
  1     -1.00000000       -1.000000      2.3248e-10           11        6.081
  2     -0.25000000       -0.250000      1.2138e-10           22       13.237
  3     -0.11111111       -0.111111      1.9830e-10           12        6.143
  4     -0.06250000       -0.062500      3.0494e-10           12        6.026
  5     -0.04000000       -0.040000      4.4468e-10           15        7.944
  6     -0.02777778       -0.027778      6.1631e-10           15        8.104
--------------------------------------------------------------------------------
```


### ionized Helium Atom ground state and first excited state (`HEp_states.py`)

```text
Testing ionized Helium atom for 2 states with l=0
--------------------------------------------------------------------------------
  n     Energy (Ry)     Theoretical           Error   Iterations    Time (ms)
--------------------------------------------------------------------------------
  1     -4.00000000       -4.000000      1.0008e-09            9        5.686
  2     -1.00000000       -1.000000      6.2772e-10           12        7.611
--------------------------------------------------------------------------------
```

### Harmonic Oscillator Levels (`HO_states.py`)

```text
Testing harmonic_oscillator for 6 states
------------------------------------------------------------------------------
  n     Energy (ħω)     Theoretical           Error   Iterations    Time (ms)
------------------------------------------------------------------------------
  0      0.50000029      0.50000000        2.86e-07           41        8.569
  1      1.50000067      1.50000000        6.69e-07           41        8.722
  2      2.50000092      2.50000000        9.18e-07           41        8.882
  3      3.50000109      3.50000000        1.09e-06           41        8.753
  4      4.50000122      4.50000000        1.22e-06           41        9.168
  5      5.50000133      5.50000000        1.33e-06           41        8.909
------------------------------------------------------------------------------
```


## Use library

1. Clone the repository:
   ```bash
   git clone https://github.com/recore799/schrodinger1d.git
   cd schrodinger1d
   ```

2. Run the examples:
   ```bash
   python examples/ATOM_states.py
   python examples/HO_states.py
   ```

## Dependencies

- Python 3.6+
- NumPy
- Matplotlib (for visualization examples)

## License

[GPL License](LICENSE)

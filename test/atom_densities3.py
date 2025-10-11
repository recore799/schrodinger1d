import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from src.numerov.numerov import solve_atom, init_mesh

n, l, m = 4, 3, 0
_, _, psi = solve_atom(n, l, mesh=3001)
_, r, _ = init_mesh(300.0, 3001, 1)

psi /= np.sqrt(np.trapz(np.abs(psi)**2 * r**2, r))

x = np.linspace(-30, 30, 1000)
z = np.linspace(-30, 30, 1000)
X, Z = np.meshgrid(x, z)
Y0 = 0
R = np.sqrt(X**2 + Y0**2 + Z**2)
theta = np.arccos(np.divide(Z, R, where=R>0, out=np.zeros_like(R)))
phi = np.arctan2(Y0, X)
Ylm = np.real(sph_harm(m, l, phi, theta))
psi_r = np.interp(R, r, psi)
psi_total = psi_r * Ylm
densidad = np.abs(psi_total)**2
densidad /= densidad.max()

plt.figure(figsize=(6,5))
plt.imshow(densidad, extent=[-30,30,-30,30], cmap="cividis", origin="lower")
plt.title("Átomo de Hidrógeno - |ψ₄₃₀|²\n(n=4, l=3, m=0)")
plt.xlabel("x [Bohr]")
plt.ylabel("z [Bohr]")
plt.colorbar(label="Densidad de Probabilidad Electrónica |ψ|²")
plt.tight_layout()
plt.savefig("hydrogen_430.png", dpi=300)
plt.show()

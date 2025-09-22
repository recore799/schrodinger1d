# src/hf/grid.py
import numpy as np

# ====== Gauss s-primitive ======
def s_norm(alpha: float) -> float:
    """Normalización de un primitivo s-GTO en 3D: (2a/pi)^(3/4)."""
    return (2.0*alpha/np.pi)**0.75

def eval_sGTO_primitive(alpha: float, center, X, Y, Z=0.0, use_norm=True):
    """
    Valor de un primitivo s-GTO en la malla (X,Y,Zconst).
    center = (xA, yA, zA)
    """
    xA, yA, zA = center
    r2 = (X - xA)**2 + (Y - yA)**2 + (Z - zA)**2
    g = np.exp(-alpha * r2)
    return (s_norm(alpha) * g) if use_norm else g

def eval_contracted_sGTO(prims, center, X, Y, Z=0.0, use_norm=True):
    """
    AO s contraído: χμ = sum_k d_k * g(alpha_k)
    prims: lista [(alpha, coeff), ...]
    """
    val = np.zeros_like(X, dtype=float)
    for alpha, d in prims:
        val += d * eval_sGTO_primitive(alpha, center, X, Y, Z, use_norm=use_norm)
    return val

# ====== MO y densidad ======
def mo_on_grid_s(primitives_per_AO, centers, C_vec, X, Y, Z=0.0, use_norm=True):
    """
    primitives_per_AO: lista de AO (μ) -> lista de (alpha, coeff)
    centers: lista de AO (μ) -> (xμ,yμ,zμ)
    C_vec: coeficientes MO en base AO (μ)
    """
    psi = np.zeros_like(X, dtype=float)
    for mu, (prims, ctr, c) in enumerate(zip(primitives_per_AO, centers, C_vec)):
        if abs(c) < 1e-14:
            continue
        psi += c * eval_contracted_sGTO(prims, ctr, X, Y, Z, use_norm=use_norm)
    return psi

def density_on_grid_s(primitives_per_AO, centers, C_occ, X, Y, Z=0.0, use_norm=True):
    """
    Densidad RHF: ρ = 2 * sum_{i ocupados} |ψ_i|^2
    C_occ: matriz (nbf, n_occ) de coeficientes de MOs ocupados
    """
    rho = np.zeros_like(X, dtype=float)
    for i in range(C_occ.shape[1]):
        psi_i = mo_on_grid_s(primitives_per_AO, centers, C_occ[:, i], X, Y, Z, use_norm=use_norm)
        rho += 2.0 * psi_i**2
    return rho

# ====== Utilidades de malla ======
def make_grid(xmin, xmax, nx, ymin, ymax, ny):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    return X, Y

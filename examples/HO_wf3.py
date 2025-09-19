# plot_oscillator.py
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# --- Localiza y añade ./src o ../src al sys.path si existen ---
here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
cand_paths = [os.path.join(here, "src"), os.path.join(here, "..", "src")]
for p in cand_paths:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# --- Importa tu solver ---
try:
    from numerov import harmonic_oscillator
except Exception as e:
    raise RuntimeError(
        "No pude importar 'harmonic_oscillator' desde numerov.py.\n"
        "Coloca numerov.py dentro de ./src o ../src.\n"
        f"Detalle: {e}"
    )

def harmonic_oscillator_wf(state: int, mesh: int, xmax: float):
    """
    Devuelve x_full, psi_full (ya normalizada por tu función) y e_num.
    Se asume que 'harmonic_oscillator' retorna (e, iterations, psi_pos) en [0,xmax].
    """
    e_num, iterations, psi_pos = harmonic_oscillator(nodes=state, mesh=mesh, xmax=xmax)

    # Eje simétrico
    x_full = np.linspace(-xmax, xmax, 2*mesh + 1)
    psi_full = np.zeros_like(x_full)

    # psi_pos suele tener longitud mesh+1 (incluye x=0 y x=xmax)
    # Colocamos la mitad positiva (índices mesh .. 2*mesh)
    if len(psi_pos) != mesh + 1:
        # Ajuste defensivo por si tu implementación cambia discretización
        # Reinterpola a mesh+1 puntos entre [0,xmax]
        x_pos_target = np.linspace(0.0, xmax, mesh + 1)
        x_pos_given = np.linspace(0.0, xmax, len(psi_pos))
        psi_pos = np.interp(x_pos_target, x_pos_given, psi_pos)

    psi_full[mesh:] = psi_pos

    # Simetría para x<0 (sin duplicar el punto x=0)
    if state % 2 == 0:
        # par: psi(-x)=psi(x)
        psi_full[:mesh] = np.flip(psi_pos[1:])
    else:
        # impar: psi(-x)=-psi(x)
        psi_full[:mesh] = -np.flip(psi_pos[1:])

    return x_full, psi_full, e_num


if __name__ == "__main__":
    # ---------- Parámetros ----------
    states_to_plot = [0, 1, 2, 3, 4, 5]
    mesh = 600
    xmax = 8.0

    # Escala vertical de las psi alrededor de En para que SE VEAN
    psi_scale = 0.55   # si se ven pequeñas, súbelo (0.6–0.7)

    # ---------- Figura ----------
    fig = plt.figure(figsize=(11, 8))

    # Potencial V(x)=x^2/2
    x = np.linspace(-xmax, xmax, 2000)
    vpot = 0.5 * x**2
    plt.plot(x, vpot, color="black", linewidth=1.6, alpha=0.65, label="V(x) = 1/2 x^2")

    # Curvas de cada estado (psi normalizada por TU función), desplazadas por En
    for n in states_to_plot:
        x_full, psi, e_num = harmonic_oscillator_wf(state=n, mesh=mesh, xmax=xmax)
        e_teo = n + 0.5

        # Desplaza por En (puedes cambiar a e_teo si lo prefieres)
        psi_offset = e_num + psi_scale * psi
        plt.plot(x_full, psi_offset, linewidth=2.0, label=f"n={n}, E={e_num:.2f}")

    # Líneas horizontales E_teo
    for n in states_to_plot:
        e_teo = n + 0.5
        plt.axhline(e_teo, color="gray", linestyle=":", alpha=0.5)

    # Estética (solo ASCII para evitar problemas de fuentes)
    plt.title("Oscilador armonico: psi_n(x) normalizadas, desplazadas por E_n", fontsize=16)
    plt.xlabel("x", fontsize=13)
    plt.ylabel("E_n + escala * psi_n(x)", fontsize=13)
    plt.grid(alpha=0.2)

    # Límites verticales razonables
    ymax = max(n + 0.5 for n in states_to_plot) + (psi_scale + 0.35)
    ymin = min(0.0, vpot.min()) - 0.2
    plt.ylim(ymin, ymax)

    # Leyenda compacta
    plt.legend(loc="upper right", frameon=False, fontsize=10)

    plt.tight_layout()

    # Guardado PNG + PDF
    out_base = "oscillator_states"
    plt.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_base}.pdf", bbox_inches="tight")

    plt.show()

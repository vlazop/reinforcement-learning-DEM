"""
Gráficos del proyecto (todos se guardan como PNG en results/):

  00_grid_dem.png           qué es el DEM: una malla de celdas, un número por celda
  01_entorno.png            el problema: terreno real, costo y obstáculos
  02_politica_qlearning.png la política aprendida (flechas) + ruta A→B
  03_curva_aprendizaje.png  recompensa por episodio (¿el agente mejora?)
  04_comparacion_dijkstra.png ruta aprendida vs óptimo exacto (Dijkstra)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from entorno import ACCIONES

COLOR_RUTA = "#D55E00"       # naranja (paleta Okabe-Ito, segura para daltonismo)
COLOR_OBSTACULO = "#7f1d1d"  # granate: se distingue de las sombras del relieve
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": False})


def _fondo_terreno(ax, env):
    """
    Dibuja el fondo común de los mapas: relieve sombreado (hillshade)
    del DEM real + obstáculos + inicio y meta.
    """
    ls = LightSource(azdeg=315, altdeg=45)
    sombra = ls.hillshade(env.elevacion, vert_exag=2)
    ax.imshow(sombra, cmap="gray", alpha=0.9, interpolation="bilinear")

    # Obstáculos (pendiente intransitable) encima del relieve.
    mascara = np.ma.masked_where(~env.obstaculo, np.ones_like(env.elevacion))
    ax.imshow(mascara, cmap=plt.matplotlib.colors.ListedColormap([COLOR_OBSTACULO]),
              alpha=0.85, interpolation="nearest")

    ax.plot(env.inicio[1], env.inicio[0], "o", color="white", ms=9,
            mec="black", mew=1.5, zorder=5)
    ax.plot(env.meta[1], env.meta[0], "*", color="gold", ms=16,
            mec="black", mew=1, zorder=5)
    ax.annotate("A (inicio)", (env.inicio[1], env.inicio[0]),
                xytext=(6, -6), textcoords="offset points", fontsize=8,
                color="black", bbox=dict(fc="white", alpha=0.8, ec="none", pad=1))
    ax.annotate("B (meta)", (env.meta[1], env.meta[0]),
                xytext=(-6, 10), textcoords="offset points", fontsize=8,
                ha="right", color="black",
                bbox=dict(fc="white", alpha=0.8, ec="none", pad=1))
    ax.set_xticks([])
    ax.set_yticks([])


def grid_dem(env, ruta="results/00_grid_dem.png", zoom=6):
    """
    Figura 0 (didáctica): ¿qué es exactamente el DEM?

    Panel izquierdo: todo el mapa como lo que realmente es, una malla de
    celdas. Cada casilla representa un cuadrado de terreno de
    tam_celda x tam_celda metros (60 m tras la reducción).

    Panel derecho: zoom a una ventana de `zoom` x `zoom` celdas, con el
    número que guarda cada una escrito encima: su elevación en metros
    sobre el nivel del mar. Una celda = un cuadrado de terreno = un número.
    """
    tc = env.tam_celda
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # --- Panel A: el DEM completo con su cuadrícula ---
    im = axes[0].imshow(env.elevacion, cmap="terrain")
    # Dibujar la línea de cada celda para que se vea que es una malla.
    axes[0].set_xticks(np.arange(-0.5, env.columnas), minor=True)
    axes[0].set_yticks(np.arange(-0.5, env.filas), minor=True)
    axes[0].grid(which="minor", color="black", linewidth=0.2, alpha=0.35)
    axes[0].tick_params(which="minor", size=0)
    axes[0].set_xticks([]), axes[0].set_yticks([])
    axes[0].set_title(
        f"El DEM es una malla: {env.filas} x {env.columnas} celdas "
        f"de {tc:.0f} m x {tc:.0f} m"
    )
    axes[0].set_xlabel(
        f"{env.columnas} celdas x {tc:.0f} m = "
        f"{env.columnas * tc / 1000:.1f} km de terreno de lado"
    )
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="elevación (m)")

    # Marcar la ventana que ampliamos en el panel derecho.
    r0 = min(env.inicio[0], env.filas - zoom)
    c0 = min(env.inicio[1], env.columnas - zoom)
    axes[0].add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), zoom, zoom,
                                    fill=False, edgecolor=COLOR_RUTA, lw=2))

    # --- Panel B: zoom con el número de cada celda a la vista ---
    ventana = env.elevacion[r0:r0 + zoom, c0:c0 + zoom]
    # `extent` pone los ejes en METROS reales en vez de índices de celda.
    axes[1].imshow(ventana, cmap="terrain",
                   extent=[0, zoom * tc, zoom * tc, 0])
    axes[1].set_xticks(np.arange(0, zoom * tc + 1, tc))
    axes[1].set_yticks(np.arange(0, zoom * tc + 1, tc))
    axes[1].grid(color="black", linewidth=0.6, alpha=0.45)
    axes[1].set_xlabel("metros")
    axes[1].set_ylabel("metros")
    for i in range(zoom):
        for j in range(zoom):
            axes[1].text((j + 0.5) * tc, (i + 0.5) * tc,
                         f"{ventana[i, j]:.0f}",
                         ha="center", va="center", fontsize=8,
                         bbox=dict(fc="white", alpha=0.65, ec="none", pad=1))
    axes[1].set_title(
        f"Zoom (recuadro naranja): una celda = {tc:.0f} m x {tc:.0f} m\n"
        "y guarda UN número: su elevación en metros"
    )

    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def mapa_entorno(env, ruta="results/01_entorno.png"):
    """Figura 1: el problema — terreno real, costo y obstáculos."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

    _fondo_terreno(axes[0], env)
    axes[0].set_title("Terreno real (Copernicus DEM, celdas de 60 m) — Misti, Arequipa")

    im = axes[1].imshow(env.costo, cmap="YlOrBr")
    mascara = np.ma.masked_where(~env.obstaculo, np.ones_like(env.costo))
    axes[1].imshow(mascara,
                   cmap=plt.matplotlib.colors.ListedColormap([COLOR_OBSTACULO]),
                   interpolation="nearest")
    axes[1].set_title("Costo por celda (1 + 8·pendiente); granate = obstáculo")
    axes[1].set_xticks([]), axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="costo del paso")

    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def mapa_politica(env, politica, camino, ruta="results/02_politica_qlearning.png",
                  paso=3):
    """
    Figura 2: la política aprendida como flechas sobre el terreno,
    y la ruta A→B que resulta de seguirla.
    `paso` submuestrea las flechas para que el mapa se pueda leer.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _fondo_terreno(ax, env)

    # Flechas de la política (solo en celdas libres, cada `paso` celdas).
    filas = np.arange(0, env.filas, paso)
    cols = np.arange(0, env.columnas, paso)
    F, C = np.meshgrid(filas, cols, indexing="ij")
    df = np.array([ACCIONES[a][0] for a in politica[F, C].ravel()]).reshape(F.shape)
    dc = np.array([ACCIONES[a][1] for a in politica[F, C].ravel()]).reshape(F.shape)
    libre = ~env.obstaculo[F, C]
    ax.quiver(C[libre], F[libre], dc[libre], -df[libre],
              color="#111111", alpha=0.8, scale=38, width=0.0032)

    # La ruta que sigue el agente desde el inicio.
    ys, xs = zip(*camino)
    ax.plot(xs, ys, color=COLOR_RUTA, lw=2.4, zorder=4, solid_capstyle="round")

    ax.set_title("Política aprendida por Q-learning (flechas) y ruta A→B")
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def comparacion_dijkstra(env, camino_rl, costo_rl, camino_dij, costo_dij,
                         ruta="results/04_comparacion_dijkstra.png"):
    """
    Figura 4: la ruta aprendida por Q-learning (sin conocer el mapa)
    junto al óptimo exacto de Dijkstra (con el mapa completo). La
    cercanía entre ambas es la validación del aprendizaje.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _fondo_terreno(ax, env)

    ys, xs = zip(*camino_dij)
    ax.plot(xs, ys, color="#0072B2", lw=3.2, zorder=3, solid_capstyle="round",
            label=f"Dijkstra (óptimo): costo {costo_dij:.1f}")
    ys, xs = zip(*camino_rl)
    ax.plot(xs, ys, color=COLOR_RUTA, lw=2.0, zorder=4, solid_capstyle="round",
            label=f"Q-learning: costo {costo_rl:.1f}")

    brecha = 100 * (costo_rl - costo_dij) / costo_dij
    ax.set_title("Ruta aprendida vs. óptimo exacto "
                 f"(brecha: {brecha:.1f}%)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def _media_movil(x, ventana=100):
    """
    Promedia cada punto con sus 100 vecinos. La recompensa episodio a
    episodio es muy ruidosa (el azar de ε mete picos); suavizada se ve
    la tendencia real: ¿está mejorando el agente o no?
    """
    return np.convolve(x, np.ones(ventana) / ventana, mode="valid")


def curva_aprendizaje(recompensas, ruta="results/03_curva_aprendizaje.png",
                      ventana=100):
    """
    Figura 3: recompensa total por episodio (media móvil). Si la curva
    sube y se aplana, el agente aprendió.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(_media_movil(recompensas, ventana), color=COLOR_RUTA, lw=1.8)

    # Recortamos el eje y a propósito: los primeros episodios son tan
    # malos (recompensas de ~-7000, puro chocar y vagar) que, si los
    # mostráramos completos, aplastarían visualmente toda la parte
    # interesante de la curva — la mejora — contra el cero.
    ax.set_ylim(-1600, 150)
    ax.set_xlabel("Episodio")
    ax.set_ylabel(f"Recompensa total (media móvil {ventana} ep.)")
    ax.set_title("Curva de aprendizaje de Q-learning (eje y recortado en −1600)")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)

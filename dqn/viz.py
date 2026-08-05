"""
Funciones de gráficos (código común a los 3 notebooks).

Cada función arma una figura de matplotlib y la deja lista para mostrar. Los
notebooks solo llaman a estas funciones, así el código de dibujo vive en un
solo lugar y las 3 libretas se ven consistentes.
"""

import matplotlib.pyplot as plt
import numpy as np

from train import suavizar
from baselines import ruta_greedy, dijkstra, costo_camino


def mapa_terreno(env):
    """Elevación y pendiente lado a lado. Para presentar el terreno real."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    im1 = a1.imshow(env.elevacion, cmap="terrain")
    a1.set_title("Elevación (m)")
    plt.colorbar(im1, ax=a1, shrink=0.8)
    im2 = a2.imshow(env.pendiente, cmap="magma")
    a2.set_title("Pendiente (m/m)")
    plt.colorbar(im2, ax=a2, shrink=0.8)
    plt.tight_layout()
    return fig


def que_ve_el_agente(env):
    """
    Muestra el mapa completo con el agente y su ventana 9x9, y al lado el
    parche que la red realmente recibe. Deja claro qué es la observación local.
    """
    obs = env.reset()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.imshow(env.vista, cmap="terrain")
    a1.plot(env.estado[1], env.estado[0], "o", color="red", ms=8, label="inicio")
    a1.plot(env.meta[1], env.meta[0], "*", color="gold", ms=16,
            markeredgecolor="black", label="meta")
    r = env.radio
    a1.add_patch(plt.Rectangle((env.estado[1]-r-.5, env.estado[0]-r-.5),
                               env.parche, env.parche, fill=False, ec="red", lw=2))
    a1.legend()
    a1.set_title("El mapa (el agente NO lo ve entero)")
    a2.imshow(obs[:-2].reshape(env.parche, env.parche), cmap="terrain",
              vmin=0, vmax=1)
    a2.set_title(f"Lo que la red recibe: parche {env.parche}x{env.parche}\n"
                 f"+ vector a la meta ({obs[-2]:+.2f}, {obs[-1]:+.2f})")
    a2.set_xticks([]); a2.set_yticks([])
    plt.tight_layout()
    return fig


def curvas_aprendizaje(historia, titulo="DQN"):
    """4 paneles: recompensa, éxito de entrenamiento, pasos, test."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(historia["recompensa"], alpha=.25, lw=.5)
    ax[0, 0].plot(*suavizar(historia["recompensa"]))
    ax[0, 0].set_title("Recompensa por episodio (media móvil 100)")
    ax[0, 1].plot(*suavizar(historia["exito"]))
    ax[0, 1].set_ylim(0, 1.05)
    ax[0, 1].set_title("Tasa de éxito en entrenamiento")
    ax[1, 0].plot(*suavizar(historia["pasos"]))
    ax[1, 0].set_title("Pasos por episodio")
    ax[1, 1].plot(historia["eval_ep"], historia["eval_exito"], "o-")
    ax[1, 1].set_ylim(0, 1.05)
    ax[1, 1].set_title("TEST: éxito desde inicios nunca vistos")
    for a in ax.ravel():
        a.set_xlabel("episodio"); a.grid(alpha=.3)
    fig.suptitle(titulo)
    plt.tight_layout()
    return fig


def rutas_vs_dijkstra(env, agente, pares, n=4):
    """Dibuja n rutas del agente (rojo) contra el óptimo de Dijkstra (blanco)."""
    pares = pares[:n]
    meta = tuple(int(v) for v in pares[0][1])   # la meta es fija: va en el suptítulo
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for axi, (ini, m) in zip(axes.ravel(), pares):
        ini = tuple(int(v) for v in ini)
        camino, llego = ruta_greedy(env, agente, ini, m)
        cam_dij, costo_opt = dijkstra(env, ini, m)
        axi.imshow(env.vista, cmap="terrain")
        axi.plot([c[1] for c in cam_dij], [c[0] for c in cam_dij],
                 "--", color="white", lw=1.5, label=f"Dijkstra ({costo_opt:.0f})")
        etiqueta = f"DQN ({costo_camino(env, camino):.0f})" if llego else "DQN (no llegó)"
        axi.plot([c[1] for c in camino], [c[0] for c in camino],
                 color="red", lw=2, label=etiqueta)
        axi.plot(ini[1], ini[0], "o", color="red", ms=7)
        axi.plot(m[1], m[0], "*", color="gold", ms=15, markeredgecolor="black")
        brecha = (costo_camino(env, camino)/costo_opt - 1)*100 if llego else float("nan")
        titulo = f"inicio ({ini[0]},{ini[1]})"
        if llego:
            titulo += f"  ·  brecha {brecha:+.0f}%"
        axi.set_title(titulo, fontsize=10)
        axi.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Rutas: DQN (rojo) vs Dijkstra (blanco) — meta fija ({meta[0]},{meta[1]})",
                 fontsize=12)
    plt.tight_layout()
    return fig


def comparar_curvas(historias, etiquetas, titulo="Comparación"):
    """
    Superpone las curvas de test de varias corridas (para el experimento
    determinista vs estocástico, o para las ablaciones).
    historias: lista de dicts 'historia'; etiquetas: lista de strings.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for h, et in zip(historias, etiquetas):
        a1.plot(*suavizar(h["recompensa"]), label=et)
        a2.plot(h["eval_ep"], h["eval_exito"], "o-", label=et)
    a1.set_title("Recompensa (media móvil 100)"); a1.set_xlabel("episodio")
    a2.set_title("Test: éxito desde inicios nunca vistos")
    a2.set_xlabel("episodio"); a2.set_ylim(0, 1.05)
    for a in (a1, a2):
        a.grid(alpha=.3); a.legend()
    fig.suptitle(titulo)
    plt.tight_layout()
    return fig

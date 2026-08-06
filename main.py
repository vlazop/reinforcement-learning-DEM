"""
Script principal del proyecto:

  1. Construye el entorno (gridworld) desde el DEM real (data/dem.tif).
  2. Entrena el agente con Q-learning (escrito desde cero).
  3. Genera las figuras en results/ (todo el resultado son imágenes).

Uso:
    python main.py
"""

import time

import plots
from dijkstra import ruta_optima
from entorno import TerrenoEnv
from qlearning import entrenar, politica_desde_Q

# Elegimos γ = 1.0 (sin descuento) porque la tarea es episódica: tiene
# un final claro (llegar a B) y lo que queremos minimizar es el costo
# TOTAL del camino, contando cada paso igual. Si usáramos γ < 1, el
# premio de la meta se iría "desinflando" con la distancia: en una ruta
# de ~100 pasos, 100·0.99^100 ≈ 37. Con el premio tan diluido, al agente
# le saldría más a cuenta deambular por celdas baratas que ir a la meta.
# (Mismo planteo que el cliff walking de Sutton & Barto.)
GAMMA = 1.0
ALPHA = 0.5         # usamos un α grande para que cada experiencia corrija
                    # fuerte la tabla. Podemos permitírnoslo porque el entorno
                    # no tiene azar: lo que el agente vive una vez es lo que
                    # vivirá siempre, no hace falta promediar con cautela.
EPISODIOS = 15000
SEMILLA = 42        # semilla fija: mismos números aleatorios en cada corrida,
                    # para que los resultados sean reproducibles


def main():
    inicio_total = time.time()

    # 1. Construir el mundo: cargar el terreno real y convertirlo en
    #    tablero (todo el MDP vive en entorno.py).
    env = TerrenoEnv()
    print(f"Mapa: {env.filas}x{env.columnas} celdas "
          f"({(~env.obstaculo).sum()} libres, {env.obstaculo.sum()} obstáculos)")
    print(f"Inicio A={env.inicio}  Meta B={env.meta}\n")

    # Figura 0: qué es el DEM (malla de celdas, un número de altura por
    # celda). Puramente didáctica, no afecta al entrenamiento.
    plots.grid_dem(env)
    plots.mapa_entorno(env)

    # 2. Entrenar: el agente recorre el mapa miles de veces y va
    #    llenando su tabla Q solo con prueba y error.
    t = time.time()
    Q, recompensas, pasos = entrenar(
        env, episodios=EPISODIOS, alpha=ALPHA, gamma=GAMMA, semilla=SEMILLA,
    )
    # De la tabla Q sale el plan (política) y, siguiéndolo desde A,
    # la ruta final. Este es el "examen": sin exploración, sin azar.
    politica = politica_desde_Q(Q)
    camino, costo, llego = env.seguir_politica(politica)

    estado = "llegó a la meta" if llego else "NO llegó"
    print(f"[Q-learning] {EPISODIOS} episodios ({time.time()-t:.1f} s) | "
          f"{estado}: {len(camino)} pasos, costo {costo:.1f}")

    # 3. Vara de medir: Dijkstra calcula el óptimo exacto usando el mapa
    #    completo (cosa que el agente nunca vio). La brecha entre ambos
    #    costos dice qué tan bien aprendió el agente a ciegas.
    camino_dij, costo_dij = ruta_optima(env)
    brecha = 100 * (costo - costo_dij) / costo_dij
    print(f"[Dijkstra]   óptimo exacto: {len(camino_dij)} pasos, "
          f"costo {costo_dij:.1f} | brecha de Q-learning: {brecha:.1f}%")

    # 4. Figuras.
    plots.mapa_politica(env, politica, camino)
    plots.curva_aprendizaje(recompensas)
    plots.comparacion_dijkstra(env, camino, costo, camino_dij, costo_dij)

    print(f"\nListo. Figuras en results/ ({time.time()-inicio_total:.0f} s en total)")


if __name__ == "__main__":
    main()

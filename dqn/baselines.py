"""
Líneas base para comparar contra el agente (código común a los notebooks).

- Dijkstra: la ruta de costo mínimo EXACTA, calculada con el mapa completo.
  Es la "vara de medir": el agente solo ve parches locales y nunca el mapa
  entero, así que la brecha entre su ruta y la de Dijkstra dice qué tan bien
  aprendió a navegar a ciegas. Ojo: Dijkstra asume mundo determinista.
- ruta_greedy: la ruta que sigue el agente entrenado (política greedy, ε=0).
- costo_camino: costo total (escala del parcial) de una lista de celdas.
"""

import heapq

from entorno_drl import ACCIONES


def ruta_greedy(env, agente, inicio, meta):
    """Ruta que produce el agente siguiendo su política greedy desde 'inicio'."""
    obs = env.reset(inicio=inicio, meta=meta)
    terminado, camino = False, [env.estado]
    while not terminado:
        obs, _, terminado, llego = env.step(agente.elegir_accion(obs, 0.0))
        camino.append(env.estado)
    return camino, llego


def dijkstra(env, inicio, meta):
    """
    Ruta de costo mínimo exacta de 'inicio' a 'meta' usando el mapa completo.
    Devuelve (camino, costo_total). Es el óptimo contra el que comparamos.
    """
    dist = {inicio: 0.0}
    previo = {}
    cola = [(0.0, inicio)]
    while cola:
        d, u = heapq.heappop(cola)
        if u == meta:
            break
        if d > dist.get(u, float("inf")):
            continue
        for df, dc in ACCIONES.values():
            v = (u[0] + df, u[1] + dc)
            if not (0 <= v[0] < env.filas and 0 <= v[1] < env.columnas):
                continue
            if env.obstaculo[v]:
                continue
            nd = d + env.costo[v]
            if nd < dist.get(v, float("inf")):
                dist[v], previo[v] = nd, u
                heapq.heappush(cola, (nd, v))
    camino, c = [meta], meta
    while c != inicio:
        c = previo[c]
        camino.append(c)
    return camino[::-1], dist[meta]


def costo_camino(env, camino):
    """Costo total (escala del parcial) de una lista de celdas."""
    return float(sum(env.costo[c] for c in camino[1:]))


def brecha_vs_dijkstra(env, agente, pares):
    """
    Métrica resumen: sobre 'pares', promedio de cuánto se aleja la ruta del
    agente del óptimo de Dijkstra (%). Solo cuenta los pares donde el agente
    llegó. Devuelve (brecha_media_%, cuántos_llegó, total).
    """
    brechas, llegadas = [], 0
    for inicio, meta in pares:
        camino, llego = ruta_greedy(env, agente, inicio, meta)
        if not llego:
            continue
        llegadas += 1
        _, costo_opt = dijkstra(env, inicio, meta)
        costo_ag = costo_camino(env, camino)
        brechas.append(100 * (costo_ag / costo_opt - 1))
    media = sum(brechas) / len(brechas) if brechas else float("nan")
    return media, llegadas, len(pares)

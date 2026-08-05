"""
Entrenamiento y evaluación reutilizables (código común a los 3 notebooks).

Aquí vive el bucle de entrenamiento del agente DQN y el "test" que mide si
llega a la meta desde inicios que nunca vio. Todo devuelve datos (diccionarios,
números) para que cada notebook arme sus propios gráficos con viz.py.

Ideas clave del entrenamiento (aprendidas depurando, ver HALLAZGOS_INFORME.md):
- ε-greedy que decae con los PASOS: primero explora, luego explota.
- Curriculum: los inicios arrancan cerca de la meta y se alejan con el tiempo,
  para que el agente aprenda primero lo fácil.
- Test periódico (ε=0) sobre pares fijos nunca entrenados = generalización.
"""

import time

import numpy as np


def pares_de_evaluacion(env, n=20, semilla=12345):
    """
    Pares (inicio, meta) de test, sorteados con OTRA semilla que el
    entrenamiento. En modo meta fija la meta es siempre la misma y solo
    varía el inicio: medimos si el agente llega desde inicios nunca vistos.
    """
    rng = np.random.default_rng(semilla)
    pares = []
    while len(pares) < n:
        inicio = tuple(env.libres[rng.integers(len(env.libres))])
        meta = env.meta_fija if env.meta_fija is not None else \
            tuple(env.libres[rng.integers(len(env.libres))])
        if abs(inicio[0] - meta[0]) + abs(inicio[1] - meta[1]) >= env.dist_minima:
            pares.append((inicio, meta))
    return pares


def evaluar(env, agente, pares):
    """
    Test greedy (ε=0) sobre pares fijos. Devuelve (tasa de éxito, pasos
    promedio). Sin exploración: es la política final pura.
    """
    exitos, pasos = 0, 0
    for inicio, meta in pares:
        obs = env.reset(inicio=inicio, meta=meta)
        terminado = False
        while not terminado:
            obs, _, terminado, llego = env.step(agente.elegir_accion(obs, 0.0))
        exitos += int(llego)
        pasos += env.pasos
    return exitos / len(pares), pasos / len(pares)


def entrenar(env, agente, episodios, eps_fin=0.05, eps_pasos=150_000,
             eval_cada=100, curriculum=True, verbose=True):
    """
    Entrena el agente y devuelve un diccionario 'historia' con listas:
      recompensa, pasos, exito         (una entrada por episodio)
      eval_ep, eval_exito, eval_pasos  (una entrada cada 'eval_cada' episodios)

    curriculum=True: el inicio arranca a <=20 celdas de la meta y el tope crece
    hasta cubrir todo el mapa en la primera mitad del entrenamiento.
    """
    pares = pares_de_evaluacion(env)
    historia = {"recompensa": [], "pasos": [], "exito": [],
                "eval_ep": [], "eval_exito": [], "eval_pasos": []}
    paso_global, t0 = 0, time.time()
    curr_ini, curr_fin = 20, env.filas + env.columnas
    curr_hasta = max(1, episodios // 2)

    for ep in range(1, episodios + 1):
        if curriculum:
            dist_max = curr_ini + min(1.0, ep / curr_hasta) * (curr_fin - curr_ini)
            obs = env.reset(dist_maxima=dist_max)
        else:
            obs = env.reset()

        total, terminado, llego = 0.0, False, False
        while not terminado:
            eps = max(eps_fin, 1.0 - paso_global / eps_pasos)
            accion = agente.elegir_accion(obs, eps)
            obs_sig, recompensa, terminado, llego = env.step(accion)
            agente.observar_y_entrenar(obs, accion, recompensa, obs_sig,
                                       terminado and llego)
            obs = obs_sig
            total += recompensa
            paso_global += 1

        historia["recompensa"].append(total)
        historia["pasos"].append(env.pasos)
        historia["exito"].append(int(llego))

        if ep % eval_cada == 0 or ep == episodios:
            ex, pa = evaluar(env, agente, pares)
            historia["eval_ep"].append(ep)
            historia["eval_exito"].append(ex)
            historia["eval_pasos"].append(pa)
            if verbose:
                print(f"ep {ep:5d} | recompensa {total:8.1f} | ε {eps:.2f} | "
                      f"test {ex:5.0%} éxito, {pa:5.0f} pasos | "
                      f"{time.time()-t0:5.0f} s")

    return historia


def suavizar(x, k=100):
    """Media móvil; devuelve (xs, ys) listos para plotear."""
    k = min(k, len(x))
    if k < 1:
        return range(0), np.array([])
    return range(k - 1, len(x)), np.convolve(x, np.ones(k) / k, mode="valid")

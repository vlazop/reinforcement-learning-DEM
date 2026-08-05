"""
Bucle de entrenamiento para métodos de POLÍTICA (REINFORCE).

Se diferencia del de DQN (train.py) en que entrena por EPISODIOS COMPLETOS: junta
todo el episodio (acciones, log-probs, recompensas) y recién al final hace UN
update de gradiente de política. No hay buffer ni updates paso a paso.

Reutiliza el test (evaluar) y los pares fijos de train.py, así la comparación
DQN vs REINFORCE se mide exactamente igual.
"""

import time

from train import pares_de_evaluacion, evaluar   # mismo test que DQN


def entrenar_pg(env, agente, episodios, eval_cada=100, curriculum=True,
                verbose=True):
    """
    Entrena un agente de política y devuelve la misma 'historia' que train.entrenar
    (recompensa, pasos, exito por episodio; eval_* cada 'eval_cada'), para poder
    graficar DQN y REINFORCE con las mismas funciones de viz.py.
    """
    pares = pares_de_evaluacion(env)
    historia = {"recompensa": [], "pasos": [], "exito": [],
                "eval_ep": [], "eval_exito": [], "eval_pasos": []}
    t0 = time.time()
    curr_ini, curr_fin = 20, env.filas + env.columnas
    curr_hasta = max(1, episodios // 2)

    for ep in range(1, episodios + 1):
        if curriculum:
            dist_max = curr_ini + min(1.0, ep / curr_hasta) * (curr_fin - curr_ini)
            obs = env.reset(dist_maxima=dist_max)
        else:
            obs = env.reset()

        log_probs, recompensas = [], []
        total, terminado, llego = 0.0, False, False
        while not terminado:
            accion, log_prob = agente.actuar(obs)          # muestrea de la política
            obs, recompensa, terminado, llego = env.step(accion)
            log_probs.append(log_prob)
            recompensas.append(recompensa)
            total += recompensa

        agente.actualizar(log_probs, recompensas)          # un update por episodio

        historia["recompensa"].append(total)
        historia["pasos"].append(env.pasos)
        historia["exito"].append(int(llego))

        if ep % eval_cada == 0 or ep == episodios:
            ex, pa = evaluar(env, agente, pares)
            historia["eval_ep"].append(ep)
            historia["eval_exito"].append(ex)
            historia["eval_pasos"].append(pa)
            if verbose:
                print(f"ep {ep:5d} | recompensa {total:8.1f} | "
                      f"test {ex:5.0%} éxito, {pa:5.0f} pasos | "
                      f"{time.time()-t0:5.0f} s")

    return historia

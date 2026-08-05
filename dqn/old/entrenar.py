"""
Entrenamiento del agente DQN sobre el terreno del Misti.

Uso típico:
    python entrenar.py                          # DQN completo, 3000 episodios
    python entrenar.py --episodios 300          # corrida corta de prueba
    python entrenar.py --sin-replay             # ablación 1
    python entrenar.py --sin-target             # ablación 2
    python entrenar.py --doble                  # Double DQN
    python entrenar.py --resbalon 0.2           # transición estocástica

Deja en results/:
    <nombre>.pt          pesos entrenados
    <nombre>_metricas.csv una fila por episodio (recompensa, pasos, éxito...)
"""

import argparse
import csv
import os
import time

import numpy as np

from entorno_drl import TerrenoDRLEnv, NUM_ACCIONES
from dqn import AgenteDQN


def evaluar(env, agente, pares):
    """
    Examen sin exploración (ε=0) sobre pares (A, B) fijos que el
    entrenamiento nunca sorteó. Devuelve tasa de éxito y pasos promedio.
    """
    exitos, pasos_tot = 0, 0
    for inicio, meta in pares:
        obs = env.reset(inicio=inicio, meta=meta)
        terminado = False
        while not terminado:
            accion = agente.elegir_accion(obs, epsilon=0.0)
            obs, _, terminado, llego = env.step(accion)
        exitos += int(llego)
        pasos_tot += env.pasos
    return exitos / len(pares), pasos_tot / len(pares)


def pares_de_evaluacion(env, n=20, semilla=12345):
    """
    Pares (inicio, meta) fijos de examen, sorteados con OTRA semilla que el
    entrenamiento. En modo meta fija, la meta es siempre la misma y solo
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


def main():
    p = argparse.ArgumentParser(description="Entrena DQN sobre el DEM del Misti")
    p.add_argument("--episodios", type=int, default=3000)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--resbalon", type=float, default=0.0)
    p.add_argument("--sin-replay", action="store_true")
    p.add_argument("--sin-target", action="store_true")
    p.add_argument("--doble", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--nombre", default=None, help="prefijo de los archivos de salida")
    p.add_argument("--dem", default="../data/dem.tif")
    args = p.parse_args()

    if args.nombre is None:
        etiqueta = ["dqn"]
        if args.sin_replay:
            etiqueta.append("sin-replay")
        if args.sin_target:
            etiqueta.append("sin-target")
        if args.doble:
            etiqueta.append("doble")
        args.nombre = "_".join(etiqueta) + f"_s{args.semilla}"

    env = TerrenoDRLEnv(ruta_dem=args.dem, prob_resbalon=args.resbalon,
                        semilla=args.semilla)
    agente = AgenteDQN(
        dim_obs=env.dim_obs,
        num_acciones=NUM_ACCIONES,
        usar_replay=not args.sin_replay,
        usar_target=not args.sin_target,
        doble=args.doble,
        device=args.device,
        semilla=args.semilla,
    )
    pares_eval = pares_de_evaluacion(env)

    print(f"Mapa {env.filas}x{env.columnas} | celdas libres conectadas: {len(env.libres)}")
    print(f"Config: replay={agente.usar_replay} target={agente.usar_target} "
          f"doble={agente.doble} resbalón={args.resbalon} device={args.device}")

    # ε baja linealmente con los PASOS (no episodios): de 1.0 a 0.05 en
    # los primeros 150k pasos, y ahí se queda (algo de exploración siempre,
    # porque cada episodio es una meta nueva).
    EPS_FIN, EPS_PASOS = 0.05, 150_000
    paso_global = 0

    # CURRICULUM: la meta arranca cerca (dist máx = 20 celdas) y se aleja
    # linealmente hasta cubrir todo el mapa en la primera mitad del
    # entrenamiento. Cura el ping-pong greedy en salidas lejanas: el agente
    # aprende primero los estados fáciles y extiende su competencia hacia
    # afuera, en vez de nunca practicar las salidas lejanas del examen.
    CURR_INI, CURR_FIN = 20, env.filas + env.columnas
    curr_hasta = args.episodios // 2

    os.makedirs("results", exist_ok=True)
    ruta_csv = f"results/{args.nombre}_metricas.csv"
    t0 = time.time()

    with open(ruta_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["episodio", "recompensa", "pasos", "exito", "epsilon",
                    "perdida", "eval_exito", "eval_pasos"])

        for ep in range(1, args.episodios + 1):
            frac = min(1.0, ep / curr_hasta)
            dist_max = CURR_INI + frac * (CURR_FIN - CURR_INI)
            obs = env.reset(dist_maxima=dist_max)
            total, terminado, llego = 0.0, False, False
            perdidas = []

            while not terminado:
                epsilon = max(EPS_FIN, 1.0 - paso_global / EPS_PASOS)
                accion = agente.elegir_accion(obs, epsilon)
                obs_sig, recompensa, terminado, llego = env.step(accion)
                perdida = agente.observar_y_entrenar(
                    obs, accion, recompensa, obs_sig, terminado and llego
                )
                if perdida is not None:
                    perdidas.append(perdida)
                obs = obs_sig
                total += recompensa
                paso_global += 1

            # examen periódico sobre los pares fijos nunca entrenados
            eval_exito = eval_pasos = ""
            if ep % 100 == 0 or ep == args.episodios:
                eval_exito, eval_pasos = evaluar(env, agente, pares_eval)
                print(f"ep {ep:5d} | recompensa {total:8.1f} | ε {epsilon:.2f} | "
                      f"examen: {eval_exito:.0%} de éxito, {eval_pasos:.0f} pasos prom. | "
                      f"{time.time()-t0:.0f} s")

            w.writerow([ep, round(total, 2), env.pasos, int(llego),
                        round(epsilon, 3),
                        round(np.mean(perdidas), 4) if perdidas else "",
                        eval_exito, eval_pasos])

    agente.guardar(f"results/{args.nombre}.pt")
    print(f"\nListo en {time.time()-t0:.0f} s. Pesos: results/{args.nombre}.pt | "
          f"métricas: {ruta_csv}")


if __name__ == "__main__":
    main()

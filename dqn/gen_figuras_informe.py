"""
Entrena los modelos y genera TODAS las figuras del informe en informe-final/figuras/.
Guarda también un resumen numérico en informe-final/figuras/numeros.txt.

Correr desde la carpeta dqn/:  python gen_figuras_informe.py
"""
import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from entorno_drl import TerrenoDRLEnv, NUM_ACCIONES
from dqn import AgenteDQN
from reinforce import AgenteREINFORCE
from train import entrenar, pares_de_evaluacion, evaluar
from train_pg import entrenar_pg
from baselines import brecha_vs_dijkstra
import viz

DEM = "../data/dem.tif"
FIG = "../informe-final/figuras"
SEM = 42
device = "cpu"
np.random.seed(SEM); torch.manual_seed(SEM)

# episodios moderados para que las figuras cuenten la historia en tiempo razonable
EP_MAIN = 1500
EP_AB = 800
EP_STO = 1000
EP_PG = 1200
EP_DIV = 1500

log = open(f"{FIG}/numeros.txt", "w")
def anota(s):
    print(s); log.write(s + "\n"); log.flush()

def guardar(fig, nombre):
    fig.savefig(f"{FIG}/{nombre}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    anota(f"[figura] {nombre}")

# ── 0. Terreno y observación (sin entrenar) ─────────────────────────────
env = TerrenoDRLEnv(DEM, semilla=SEM)
anota(f"Mapa {env.filas}x{env.columnas} | meta fija {env.meta_fija} | obs {env.dim_obs}")
guardar(viz.mapa_terreno(env), "01_terreno.png")
guardar(viz.que_ve_el_agente(env), "02_observacion.png")

# ── 1. DQN meta fija: curva + rutas + brecha ────────────────────────────
anota("\n== DQN meta fija (principal) ==")
env = TerrenoDRLEnv(DEM, semilla=SEM)
ag = AgenteDQN(env.dim_obs, NUM_ACCIONES, device=device, semilla=SEM)
h_dqn = entrenar(env, ag, EP_MAIN, verbose=False)
guardar(viz.curvas_aprendizaje(h_dqn, "DQN - meta fija"), "03_curva_dqn.png")
pares = pares_de_evaluacion(env)
guardar(viz.rutas_vs_dijkstra(env, ag, pares, n=4), "04_rutas_dijkstra.png")
b, ll, tot = brecha_vs_dijkstra(env, ag, pares)
anota(f"DQN final: exito {h_dqn['eval_exito'][-1]:.0%}, pasos {h_dqn['eval_pasos'][-1]:.0f} | "
      f"brecha vs Dijkstra {b:.1f}% ({ll}/{tot} llegaron)")

# ── 2. Ablaciones (4 configs) ───────────────────────────────────────────
anota("\n== Ablaciones ==")
configs = {"DQN completo": dict(), "sin replay": dict(usar_replay=False),
           "sin theta-": dict(usar_target=False), "Double DQN": dict(doble=True)}
hs, ets = [], []
for nombre, extra in configs.items():
    e = TerrenoDRLEnv(DEM, semilla=SEM)
    a = AgenteDQN(e.dim_obs, NUM_ACCIONES, device=device, semilla=SEM, **extra)
    h = entrenar(e, a, EP_AB, eval_cada=100, verbose=False)
    hs.append(h); ets.append(nombre)
    anota(f"  {nombre:14s}: exito final {h['eval_exito'][-1]:.0%}")
guardar(viz.comparar_curvas(hs, ets, "Ablaciones (Liu & Zou)"), "05_ablaciones.png")

# ── 3. Determinista vs estocástico ──────────────────────────────────────
anota("\n== Determinista vs estocastico ==")
hs2 = []
for prob, et in [(0.0, "determinista (0.0)"), (0.25, "estocastico (0.25)")]:
    e = TerrenoDRLEnv(DEM, prob_resbalon=prob, semilla=SEM)
    a = AgenteDQN(e.dim_obs, NUM_ACCIONES, device=device, semilla=SEM)
    h = entrenar(e, a, EP_STO, eval_cada=100, verbose=False)
    hs2.append(h)
    anota(f"  {et:22s}: exito {h['eval_exito'][-1]:.0%}, pasos {h['eval_pasos'][-1]:.0f}")
guardar(viz.comparar_curvas(hs2, ["determinista (0.0)", "estocastico (0.25)"],
        "DQN: determinista vs estocastico"), "06_estocastico.png")

# ── 4. DQN vs REINFORCE ─────────────────────────────────────────────────
anota("\n== DQN (valor) vs REINFORCE (politica) ==")
e = TerrenoDRLEnv(DEM, semilla=SEM)
a_pg = AgenteREINFORCE(e.dim_obs, NUM_ACCIONES, device=device, semilla=SEM)
h_pg = entrenar_pg(e, a_pg, EP_PG, eval_cada=100, verbose=False)
e2 = TerrenoDRLEnv(DEM, semilla=SEM)
a_dq = AgenteDQN(e2.dim_obs, NUM_ACCIONES, device=device, semilla=SEM)
h_dq = entrenar(e2, a_dq, EP_PG, verbose=False)
guardar(viz.comparar_curvas([h_dq, h_pg], ["DQN (valor)", "REINFORCE (politica)"],
        "Valor vs Politica en el mismo terreno"), "07_dqn_vs_reinforce.png")
anota(f"  DQN       final: exito {h_dq['eval_exito'][-1]:.0%}, pasos {h_dq['eval_pasos'][-1]:.0f}")
anota(f"  REINFORCE final: exito {h_pg['eval_exito'][-1]:.0%}, pasos {h_pg['eval_pasos'][-1]:.0f}")

# ── 5. Divergencia con meta aleatoria ───────────────────────────────────
anota("\n== Meta aleatoria: divergencia |Q| ==")
env_r = TerrenoDRLEnv(DEM, meta_fija=False, semilla=SEM)
ag_r = AgenteDQN(env_r.dim_obs, NUM_ACCIONES, device=device, semilla=SEM)
pares_r = pares_de_evaluacion(env_r)
def qmag(ag):
    if len(ag.buffer) < 64: return 0.0
    obs, *_ = ag.buffer.muestrear(64)
    with torch.no_grad():
        return float(ag.red(obs).abs().mean().item())
eps_h, q_h, ex_h = [], [], []
paso = 0
for ep in range(1, EP_DIV + 1):
    obs = env_r.reset(); fin = False
    while not fin:
        e_ = max(0.05, 1.0 - paso/150000)
        ac = ag_r.elegir_accion(obs, e_)
        obs2, r, fin, ll = env_r.step(ac)
        ag_r.observar_y_entrenar(obs, ac, r, obs2, fin and ll)
        obs = obs2; paso += 1
    if ep % 100 == 0:
        ex, _ = evaluar(env_r, ag_r, pares_r)
        eps_h.append(ep); q_h.append(qmag(ag_r)); ex_h.append(ex)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(eps_h, q_h, "o-", color="crimson"); a1.set_title("|Q| medio (divergencia)")
a1.set_xlabel("episodio"); a1.grid(alpha=.3)
a2.plot(eps_h, ex_h, "s-"); a2.set_ylim(0, 1.05)
a2.set_title("exito con meta aleatoria"); a2.set_xlabel("episodio"); a2.grid(alpha=.3)
plt.tight_layout(); guardar(fig, "08_divergencia.png")
anota(f"  |Q| final: {q_h[-1]:.1f} (empezo en {q_h[0]:.1f})  |  exito final: {ex_h[-1]:.0%}")

anota("\n== LISTO: todas las figuras generadas ==")
log.close()

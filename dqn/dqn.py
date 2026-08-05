"""
DQN escrito desde cero (PyTorch): la red, el búfer y el agente.

Es el mismo Q-learning del parcial con la tabla reemplazada por una red,
más los dos mecanismos de la clase 5.1 que lo hacen estable:

- Reproducción de Experiencias (replay buffer): guardamos cada transición
  y entrenamos con mini-batches aleatorios, para romper la correlación
  entre pasos consecutivos.
- Objetivos-Q Fijados (target network): una copia congelada de la red
  calcula los objetivos, y cada C pasos se sincroniza. Sin esto, el
  objetivo se movería con cada update y el entrenamiento oscila o diverge.

Ambos se pueden APAGAR por bandera — así corremos las ablaciones del
informe (mismo experimento que Liu & Zou 2018 hicieron en Atari).
"""

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn


class RedQ(nn.Module):
    """
    La "tabla" del trabajo final: 83 entradas -> 128 -> 128 -> 4 salidas.
    Una pasada devuelve los 4 valores Q del estado (uno por acción).
    ~27 mil pesos, sin importar el tamaño del mapa.
    """

    def __init__(self, dim_obs, num_acciones, oculto=128):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Linear(dim_obs, oculto), nn.ReLU(),
            nn.Linear(oculto, oculto), nn.ReLU(),
            nn.Linear(oculto, num_acciones),
        )

    def forward(self, x):
        return self.capas(x)


class BufferReproduccion:
    """Memoria de reproducción: capacidad fija, lo más viejo se expulsa."""

    def __init__(self, capacidad):
        self.memoria = deque(maxlen=capacidad)

    def guardar(self, obs, accion, recompensa, obs_sig, terminado):
        self.memoria.append((obs, accion, recompensa, obs_sig, terminado))

    def muestrear(self, n):
        lote = random.sample(self.memoria, n)
        obs, acc, rec, sig, fin = zip(*lote)
        return (
            torch.as_tensor(np.array(obs)),
            torch.as_tensor(acc, dtype=torch.long),
            torch.as_tensor(rec, dtype=torch.float32),
            torch.as_tensor(np.array(sig)),
            torch.as_tensor(fin, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.memoria)


class AgenteDQN:
    """
    Junta todo: red principal θ, red objetivo θ⁻, búfer y el update.

    Banderas de ablación:
      usar_replay=False  -> entrena solo con la última transición
      usar_target=False  -> los objetivos se calculan con la propia red θ
      doble=True         -> Double DQN: θ elige la acción, θ⁻ la evalúa
    """

    def __init__(
        self,
        dim_obs,
        num_acciones,
        gamma=0.99,
        lr=1e-3,
        batch=64,
        capacidad_buffer=100_000,
        pasos_calentamiento=2_000,   # experiencias antes de empezar a entrenar
        sincronizar_cada=1_000,      # cada cuántos updates: θ⁻ <- θ
        usar_replay=True,
        usar_target=True,
        doble=False,
        device="cpu",
        semilla=0,
    ):
        torch.manual_seed(semilla)
        random.seed(semilla)
        self.rng = np.random.default_rng(semilla)

        self.device = torch.device(device)
        self.red = RedQ(dim_obs, num_acciones).to(self.device)
        self.red_objetivo = RedQ(dim_obs, num_acciones).to(self.device)
        self.red_objetivo.load_state_dict(self.red.state_dict())
        self.red_objetivo.eval()

        self.optimizador = torch.optim.Adam(self.red.parameters(), lr=lr)
        self.buffer = BufferReproduccion(capacidad_buffer)

        self.num_acciones = num_acciones
        self.gamma = gamma
        self.batch = batch
        self.pasos_calentamiento = pasos_calentamiento
        self.sincronizar_cada = sincronizar_cada
        self.usar_replay = usar_replay
        self.usar_target = usar_target
        self.doble = doble
        self.updates = 0

    # ------------------------------------------------------------------

    def elegir_accion(self, obs, epsilon):
        """ε-greedy, igual que en el parcial: explorar o explotar."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.num_acciones))
        with torch.no_grad():
            x = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            return int(self.red(x).argmax().item())

    def observar_y_entrenar(self, obs, accion, recompensa, obs_sig, terminado):
        """Guarda la transición y, si ya hay material, hace un update."""
        self.buffer.guardar(obs, accion, recompensa, obs_sig, terminado)
        if len(self.buffer) < max(self.batch, self.pasos_calentamiento):
            return None
        if self.usar_replay:
            lote = self.buffer.muestrear(self.batch)
        else:
            # Ablación sin replay: entrenar solo con lo recién vivido,
            # como el Q-learning tabular (batch de 1, correlacionado).
            lote = (
                torch.as_tensor(np.array([obs])),
                torch.as_tensor([accion], dtype=torch.long),
                torch.as_tensor([recompensa], dtype=torch.float32),
                torch.as_tensor(np.array([obs_sig])),
                torch.as_tensor([float(terminado)], dtype=torch.float32),
            )
        return self._update(*(t.to(self.device) for t in lote))

    def _update(self, obs, acc, rec, sig, fin):
        """El update TD del parcial, en versión gradiente sobre un batch."""
        # objetivo = r + γ · max_a' Q(s', a')  — con la red que corresponda
        with torch.no_grad():
            red_eval = self.red_objetivo if self.usar_target else self.red
            if self.doble:
                # Double DQN: la red θ ELIGE la mejor acción del siguiente
                # estado, la red θ⁻ la VALORA. Ruidos distintos -> sin el
                # sesgo optimista del max (clase 5.1 / Van Hasselt 2016).
                mejores = self.red(sig).argmax(dim=1, keepdim=True)
                q_sig = red_eval(sig).gather(1, mejores).squeeze(1)
            else:
                q_sig = red_eval(sig).max(dim=1).values
            objetivo = rec + self.gamma * q_sig * (1.0 - fin)

        prediccion = self.red(obs).gather(1, acc.unsqueeze(1)).squeeze(1)
        perdida = nn.functional.smooth_l1_loss(prediccion, objetivo)

        self.optimizador.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(self.red.parameters(), 10.0)
        self.optimizador.step()

        self.updates += 1
        if self.usar_target and self.updates % self.sincronizar_cada == 0:
            self.red_objetivo.load_state_dict(self.red.state_dict())  # θ⁻ <- θ

        return float(perdida.item())

    # ------------------------------------------------------------------

    def guardar(self, ruta):
        torch.save(self.red.state_dict(), ruta)

    def cargar(self, ruta):
        estado = torch.load(ruta, map_location=self.device)
        self.red.load_state_dict(estado)
        self.red_objetivo.load_state_dict(estado)

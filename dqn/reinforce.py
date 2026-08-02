"""
REINFORCE (policy gradient) — OTRA FAMILIA de RL, para comparar con DQN.

DQN es "basado en valor": aprende Q(s,a) y la acción sale del argmax (política
determinista). REINFORCE es "basado en política": aprende DIRECTAMENTE una
política π(a|s) que da PROBABILIDADES de acción (política estocástica), y ajusta
esas probabilidades para que las acciones que llevaron a buen retorno se vuelvan
más probables.

La idea en una línea (gradiente de política):
    subir  log π(a|s) · G     para cada paso,
donde G es el retorno (recompensa futura descontada) desde ese paso. Si una
acción condujo a mucho retorno, la empujamos hacia arriba; si a poco, hacia abajo.

Diferencias con DQN (útiles para el informe):
- Política ESTOCÁSTICA (da probabilidades) vs la greedy determinista de DQN.
- NO usa buffer de reproducción ni red objetivo. Entrena por EPISODIOS completos.
- Más simple, pero de mayor varianza y menos eficiente en muestras.
"""

import numpy as np
import torch
import torch.nn as nn


class PoliticaRed(nn.Module):
    """
    Red de política: 83 -> 128 -> 128 -> 4. La salida son "logits"; al pasarlos
    por softmax se vuelven las probabilidades de cada acción. Misma forma que la
    RedQ de DQN, pero su salida se interpreta como política, no como valores Q.
    """

    def __init__(self, dim_obs, num_acciones, oculto=128):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Linear(dim_obs, oculto), nn.ReLU(),
            nn.Linear(oculto, oculto), nn.ReLU(),
            nn.Linear(oculto, num_acciones),
        )

    def forward(self, x):
        return self.capas(x)   # logits (antes del softmax)


class AgenteREINFORCE:
    """El agente policy-gradient: muestrea acciones y ajusta la política."""

    def __init__(self, dim_obs, num_acciones, gamma=0.99, lr=1e-3,
                 device="cpu", semilla=0):
        torch.manual_seed(semilla)
        self.rng = np.random.default_rng(semilla)
        self.device = torch.device(device)
        self.politica = PoliticaRed(dim_obs, num_acciones).to(self.device)
        self.opt = torch.optim.Adam(self.politica.parameters(), lr=lr)
        self.gamma = gamma
        self.num_acciones = num_acciones

    def actuar(self, obs):
        """
        Para ENTRENAR: muestrea una acción según las probabilidades de la
        política (exploración natural, sin ε-greedy). Devuelve (acción, log_prob),
        el log-prob se guarda para el update.
        """
        x = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        dist = torch.distributions.Categorical(logits=self.politica(x))
        a = dist.sample()
        return int(a.item()), dist.log_prob(a)

    def elegir_accion(self, obs, epsilon=0.0):
        """
        Para el TEST: la acción más probable (greedy sobre la política). El
        parámetro epsilon existe solo para reusar la misma función evaluar()
        que DQN; con epsilon=0 es puramente greedy.
        """
        with torch.no_grad():
            x = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            if epsilon and self.rng.random() < epsilon:
                return int(self.rng.integers(self.num_acciones))
            return int(self.politica(x).argmax().item())

    def actualizar(self, log_probs, recompensas):
        """
        El update de REINFORCE al terminar un episodio:
        1. Calcula el retorno descontado G_t desde cada paso.
        2. Le resta la media y divide por la desviación (baseline): reduce la
           varianza, que es el gran problema de REINFORCE.
        3. Pérdida = -Σ log π(a_t|s_t) · G_t  (subir la prob. de lo que fue bien).
        """
        G, retornos = 0.0, []
        for r in reversed(recompensas):
            G = r + self.gamma * G
            retornos.insert(0, G)
        retornos = torch.tensor(retornos, dtype=torch.float32, device=self.device)
        if len(retornos) > 1:
            retornos = (retornos - retornos.mean()) / (retornos.std() + 1e-8)

        perdida = -torch.stack([lp * G for lp, G in zip(log_probs, retornos)]).sum()
        self.opt.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(self.politica.parameters(), 10.0)
        self.opt.step()
        return float(perdida.item())

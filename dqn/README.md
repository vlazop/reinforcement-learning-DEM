# Deep RL — navegación sobre terreno real (Misti)

Trabajo final MIA-204. Un agente aprende a navegar terreno real (Copernicus DEM GLO-30,
108×108 celdas de 30 m) hasta una meta fija, partiendo de inicios aleatorios, viendo solo
un parche local 9×9 + un vector a la meta (83 números). La función Q vive en una red
neuronal (DQN).

## Estructura — código común + 3 notebooks

El **entorno y el algoritmo son código común**: un solo lugar, los 3 notebooks lo importan.

| Archivo | Qué es |
|---|---|
| `entorno_drl.py` | El entorno (MDP): terreno, estado local, recompensa con shaping, resbalón |
| `dqn.py` | La red Q, el buffer de reproducción y el agente (DQN y Double DQN) |
| `train.py` | Entrenamiento y examen reutilizables (curriculum, ε-greedy) |
| `baselines.py` | Dijkstra (óptimo exacto) y la brecha del agente vs el óptimo |
| `viz.py` | Funciones de gráficos (terreno, parche, curvas, rutas) |
| `datos.py` | Ubica o descarga el DEM |
| `reinforce.py` | REINFORCE: red de política + `AgenteREINFORCE` (policy gradient) |
| `train_pg.py` | Bucle de entrenamiento por episodios para REINFORCE |

| Notebook | Qué muestra |
|---|---|
| `01_meta_fija.ipynb` | **Solución principal**: DQN determinista, meta fija. Curvas, rutas vs Dijkstra, ablaciones |
| `02_estocastico.ipynb` | Transición estocástica (resbalón): determinista vs estocástico |
| `03_double_dqn.ipynb` | Double DQN + meta aleatoria (divergencia) → trabajo futuro |
| `04_reinforce.ipynb` | REINFORCE (política) vs DQN (valor): otra familia de RL |

## Cómo correr

### En Google Colab (recomendado, con GPU T4)
1. Sube la carpeta `proyecto-rl/` completa a tu Google Drive.
2. Abre el notebook que quieras en Colab.
3. `Runtime → Change runtime type → T4 GPU`.
4. `Runtime → Run all`. La primera celda monta tu Drive e importa el código común.
   Si tu carpeta no está en `MyDrive/proyecto-rl/dqn`, ajusta la variable `RUTA`.

### Localmente
```bash
pip install numpy matplotlib rasterio torch
cd proyecto-rl/dqn
jupyter notebook 01_meta_fija.ipynb   # la celda de setup detecta que no es Colab
```

## Notas de diseño

- **Meta fija + inicio aleatorio**: la meta es siempre la misma celda; el inicio cambia
  cada episodio. El agente aprende a llegar desde cualquier punto (generaliza sobre el
  inicio) y la función de valor queda estable. Es una política **single-goal**: cambiar
  la meta exige reentrenar (ver `../HALLAZGOS_INFORME.md`, sección 5.1).
- **GPU no imprescindible**: la red tiene ~27k parámetros; en CPU también corre bien.
- Todos los descubrimientos y su justificación están en `../HALLAZGOS_INFORME.md`.

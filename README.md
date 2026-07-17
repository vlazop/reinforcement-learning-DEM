# Navegación de costo mínimo sobre terreno real usando Q-learning

Un agente aprende, por prueba y error, a cruzar un terreno real (faldas del
volcán Misti, Arequipa) desde un punto A hasta un punto B gastando el menor
esfuerzo posible, evitando zonas empinadas. El terreno viene de un raster de
elevación satelital real (Copernicus DEM GLO-30, 30 m). Nadie le dice la
ruta al agente: la descubre solo.

## Cómo correr

```bash
pip install numpy matplotlib rasterio
python main.py          # entrena y genera las figuras (~10 s)
```

`data/dem.tif` ya está incluido en el repo, así que no se necesita internet.
Para re-descargar el recorte de elevación desde AWS Open Data:

```bash
python download_dem.py
```

## Archivos

| Archivo | Qué hace |
|---|---|
| `entorno.py` | El entorno: gridworld construido desde el DEM real (MDP: estados, acciones, recompensas) |
| `qlearning.py` | Q-learning escrito desde cero. |
| `plots.py` | Las 3 figuras (matplotlib) |
| `main.py` | Corre todo: construye el entorno, entrena y genera `results/` |
| `download_dem.py` | Descarga el recorte de elevación desde AWS Open Data |

## El MDP

- **Estados:** cada celda de la cuadrícula 54×54 (el DEM de 108×108 celdas de
  30 m se promedia a celdas de 60 m).
- **Acciones:** arriba, abajo, izquierda, derecha.
- **Transición:** determinística; chocar con borde u obstáculo = quedarse.
- **Recompensa:** paso normal `−(1 + 8·pendiente)`; choque `−50`; meta `+100`
  (terminal).
- **γ = 1.0** — la tarea es episódica con meta absorbente: queremos el costo
  TOTAL del camino sin descontar. Con γ < 1 y rutas de ~100 pasos el premio
  de la meta se desvanece y al agente le conviene vagar sin llegar.

## Decisiones de diseño que valen la pena explicar

1. **Exploring starts.** Los episodios de entrenamiento arrancan en celdas
   aleatorias del mapa; la evaluación siempre desde A. Sin esto, propagar el
   valor de la meta hasta A (a ~100 pasos) toma decenas de miles de episodios.
2. **α = 0.5 (alto).** El entorno es determinístico: no hay ruido que
   promediar, así que una tasa de aprendizaje grande solo acelera.
3. **ε decae hasta casi 0.** Primero explora mucho; al final casi solo
   explota lo aprendido.
4. **Celdas inalcanzables se enmascaran** con un BFS desde la meta, para que
   desde toda celda libre exista un camino a la meta.

## Resultados (results/)

Todo el resultado son imágenes:

| Figura | Contenido |
|---|---|
| `00_grid_dem.png` | Qué es el DEM: malla de celdas de 60 m, un valor de elevación por celda |
| `01_entorno.png` | Terreno real sombreado + mapa de costo y obstáculos |
| `02_politica_qlearning.png` | Política aprendida (flechas) + ruta A→B |
| `03_curva_aprendizaje.png` | Recompensa por episodio: ¿el agente mejora? |

## Trabajo futuro (para el trabajo final)

- Comparar contra **SARSA** y contra la solución exacta (**Value Iteration**).
- Obstáculos reales de **OpenStreetMap** (agua, edificios) con `osmnx`.
- Transición **estocástica** (probabilidad de "resbalar").

## Datos

Copernicus DEM GLO-30 (ESA), tile S17/W072, vía AWS Open Data
(`s3://copernicus-dem-30m/`, acceso público sin login). Recorte de ~3×3 km
en las faldas del Misti: bbox (−71.475, −16.375, −71.445, −16.345).

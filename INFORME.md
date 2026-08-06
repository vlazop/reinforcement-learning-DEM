# Navegación de costo mínimo sobre terreno real usando Q-learning

**Curso:** MIA-204 · Aprendizaje por Refuerzo — Trabajo Parcial

---

## Introducción

En este trabajo entrenamos un agente para que aprenda, por prueba y error, a
cruzar un terreno real (las faldas del volcán Misti, en Arequipa) desde un
punto A hasta un punto B gastando el menor esfuerzo posible y evitando las
zonas empinadas. El terreno viene de un raster de elevación satelital real
(Copernicus DEM GLO-30, 30 m de resolución). Nadie le indica la ruta al
agente: la descubre solo, con Q-learning implementado desde cero, sin usar
ninguna librería de RL. Tras 15 000 episodios de entrenamiento (unos 6
segundos), el agente encuentra una ruta de 83 pasos y costo total 154.5 que
llega a la meta bordeando las laderas intransitables.

El informe sigue el orden de la rúbrica: primero cómo modelamos el problema,
después la solución que planteamos, los resultados que obtuvimos, el análisis
de esos resultados y, al final, lo que entendimos del algoritmo.

## 1. Modelado del problema

### 1.1 El problema

Queremos encontrar la ruta de **costo mínimo** entre dos puntos de un terreno
montañoso. Caminar por terreno plano es barato, subir laderas cuesta más, y
las zonas muy empinadas directamente no se pueden pisar. Es un problema con
aplicaciones reales (rutas de senderismo, evacuación, acceso a zonas
rurales), pero lo que nos interesaba era que fuera un caso de estudio
realista para aprendizaje por refuerzo: el agente **no conoce el mapa** (ni
los costos ni los obstáculos) y tiene que aprenderlo únicamente de su
experiencia.

### 1.2 Los datos: un DEM real

En vez de inventar un tablero, usamos terreno de verdad: un recorte de
~3×3 km de las faldas del Misti, tomado del Copernicus DEM GLO-30 (ESA),
tile S17/W072, descargado desde AWS Open Data. Un DEM (modelo digital de
elevación) es literalmente una malla de celdas donde cada celda guarda un
número: la elevación en metros de ese pedazo de terreno.

![Qué es el DEM](results/00_grid_dem.png)

**Figura 0.** El DEM es una malla de 54×54 celdas de 60 m × 60 m
(izquierda). Cada celda guarda un único número: su elevación en metros sobre
el nivel del mar (derecha, zoom del recuadro naranja). El DEM original de
108×108 celdas de 30 m lo promediamos en bloques de 2×2 para reducir el
número de estados y acelerar el aprendizaje.

### 1.3 Formulación como MDP

Formulamos el problema como un Proceso de Decisión de Markov (MDP) con sus
cuatro ingredientes:

- **Estados (S):** cada celda de la cuadrícula 54×54, identificada por
  (fila, columna). Hay 2 686 celdas libres y 230 obstáculos.
- **Acciones (A):** moverse en cruz: arriba, abajo, izquierda, derecha.
- **Transición (P):** determinística. El agente se mueve a la celda vecina
  elegida; si esa celda es borde del mapa u obstáculo, rebota y se queda
  donde estaba.
- **Recompensa (R):**
  - paso normal: paga el costo de la celda que pisa, `−(1 + 8·pendiente)`.
    Una celda plana cuesta 1; una ladera de ~30° cuesta ~5.6;
  - chocar contra borde u obstáculo: castigo de `−50` (y no se mueve);
  - llegar a la meta: premio de `+100` y el episodio termina.

La pendiente de cada celda la calculamos con `np.gradient` sobre la
elevación (metros que sube el terreno por metro avanzado, combinando las dos
direcciones con Pitágoras). Las celdas con pendiente mayor a 0.60 (~31°) las
marcamos como obstáculo: nadie sube una pared.

![El entorno](results/01_entorno.png)

**Figura 1.** Izquierda: el terreno real con relieve sombreado; en granate
los obstáculos (pendiente intransitable), A es el inicio y B la meta.
Derecha: el costo de pisar cada celda, `1 + 8·pendiente`. El agente nunca ve
estos mapas: solo recibe recompensas paso a paso.

Dos detalles del entorno que tuvimos que resolver:

1. **Islas inalcanzables.** Puede haber celdas libres completamente rodeadas
   de obstáculos. Las detectamos con una búsqueda en anchura (BFS) desde la
   meta y las enmascaramos como obstáculo, de modo que desde toda celda
   libre exista siempre un camino a B.
2. **Inicio y meta.** Los colocamos en esquinas opuestas del mapa (o en la
   celda libre más cercana si la esquina cae en obstáculo): A=(3, 15),
   B=(50, 50). Así la ruta tiene que cruzar todo el terreno.

## 2. Solución planteada

### 2.1 Por qué Q-learning

Elegimos Q-learning porque encaja con la restricción central del problema:
el agente no tiene modelo del mundo. Q-learning es **model-free**, aprende
solo con experiencia, y en un MDP tabular como el nuestro (unos 2 700
estados × 4 acciones) la tabla Q entra cómodamente en memoria. Lo
implementamos desde cero en `qlearning.py` (~50 líneas de lógica) para
entender cada pieza en lugar de llamar a una librería.

Lo que el agente aprende es una tabla Q que, para cada celda y cada acción,
estima "si estoy aquí y hago esto, ¿qué tan bien me irá en total?". Las
acciones durante el entrenamiento se eligen con una política **ε-greedy**:
con probabilidad ε el agente prueba una acción al azar (explorar) y el resto
del tiempo toma la mejor según su tabla (explotar).

### 2.2 Hiperparámetros

| Parámetro | Valor | Justificación |
|---|---|---|
| γ (descuento) | 1.0 | La tarea es episódica con meta absorbente: queremos minimizar el costo TOTAL del camino sin descontar. Con γ < 1 y rutas de ~100 pasos el premio de la meta se desvanece (100·0.99¹⁰⁰ ≈ 37) y al agente le conviene vagar por celdas baratas en vez de llegar. Es el mismo planteo que el *cliff walking* de Sutton & Barto. |
| α (tasa de aprendizaje) | 0.5 | El entorno es determinístico: no hay ruido que promediar, así que una tasa grande solo acelera la propagación del valor de la meta. |
| ε | 1.0 → 0.005 | Decae multiplicativamente (×0.999 por episodio): primero explora mucho, al final casi solo explota lo aprendido. |
| Episodios | 15 000 | Suficiente para que la curva de aprendizaje se aplane. |
| Semilla | 42 | Resultados reproducibles. |

### 2.3 Dos decisiones que resultaron clave

Al diseñar el entrenamiento nos dimos cuenta de que el algoritmo solo no
bastaba; hicieron falta dos decisiones adicionales:

1. **Exploring starts** (Sutton & Barto). Los episodios de entrenamiento
   nacen en celdas aleatorias del mapa; la evaluación, en cambio, siempre
   parte de A. Sin esto, el agente tendría que cruzar ~100 celdas por pura
   casualidad antes de descubrir la meta por primera vez, y propagar ese
   valor hasta A tomaría decenas de miles de episodios.
2. **Inicialización optimista.** La tabla Q arranca en cero y cada paso da
   recompensa negativa, así que las celdas nunca visitadas "se ven bien"
   comparadas con las ya probadas. Eso empuja al agente hacia lo
   inexplorado: exploración gratis, sin código extra.

## 3. Resultados obtenidos

El entrenamiento completo (15 000 episodios) toma unos 6 segundos en una
laptop normal. La política final, evaluada desde A sin exploración ni azar,
**llega a la meta en 83 pasos con un costo total de 154.5**.

![Política aprendida y ruta](results/02_politica_qlearning.png)

**Figura 2.** La política aprendida (flechas: la mejor acción en cada celda,
submuestreadas para legibilidad) y la ruta A→B que resulta de seguirla desde
el inicio (naranja). La ruta bordea las zonas granate (obstáculos) y evita
las laderas caras, en lugar de ir en línea recta.

![Curva de aprendizaje](results/03_curva_aprendizaje.png)

**Figura 3.** Recompensa total por episodio (media móvil de 100 episodios).
Los primeros episodios son muy malos (recompensas de ~−7000: puro chocar y
vagar; el eje y está recortado en −1600 para que la mejora no quede
aplastada contra el cero). La curva sube de forma sostenida y se aplana
cerca de cero.

## 4. Análisis de los resultados

### 4.1 La curva de aprendizaje

La forma de la curva (Figura 3) cuenta la historia del entrenamiento en tres
etapas. Al inicio, con ε ≈ 1, el agente se mueve al azar: episodios
larguísimos, llenos de choques contra obstáculos (−50 cada uno), con
recompensas totales cerca de −7000. En la etapa intermedia pasan dos cosas a
la vez: ε decae, así que el agente explota cada vez más lo que sabe, y el
valor de la meta se va propagando celda a celda hacia atrás por los updates
TD. Por eso la mejora es gradual y no un salto: cada episodio que llega a B
"enseña" el camino a las celdas cercanas, y esas a las siguientes. En la
etapa final la curva se aplana cerca de cero justo cuando ε llega a su
mínimo: el agente ya casi no explora y su comportamiento convergió.

### 4.2 La ruta aprendida

La ruta de la Figura 2 no es la línea recta entre A y B, y eso es
exactamente lo que queríamos ver. La recta cruzaría zonas de pendiente alta
(celdas caras) y obstáculos (castigo de −50). El agente aprendió a rodear
las laderas empinadas y a preferir el terreno llano, que es el
comportamiento que la función de recompensa buscaba inducir, y lo descubrió
sin haber visto nunca el mapa de costos. Para nosotros este es el resultado
central del trabajo: la función de recompensa se tradujo en el
comportamiento esperado.

También revisamos las flechas de la política fuera de la ruta: en casi todo
el mapa apuntan "hacia B rodeando obstáculos", no solo sobre el camino
final. Eso es efecto de los exploring starts: como los episodios nacieron
por todos lados, la tabla Q aprendió una política razonable para todo el
mapa, no únicamente para el trayecto A→B.

### 4.3 El efecto de las decisiones de diseño

Las tres decisiones de la sección 2 no fueron adornos; sin ellas el
entrenamiento no funcionaba o tardaba muchísimo más:

- **γ = 1.0.** Hicimos la cuenta antes de elegirlo: con γ = 0.99 y rutas de
  ~100 pasos, el premio de la meta visto desde A vale 100·0.99¹⁰⁰ ≈ 37,
  comparable al costo acumulado del camino. El agente puede preferir
  deambular por celdas baratas a llegar. Con γ = 1 el objetivo que optimiza
  el agente es exactamente el costo total del camino, que es lo que pide el
  problema.
- **Exploring starts.** Con inicio fijo en A, la primera visita a B depende
  de una caminata aleatoria de ~100 celdas de largo, que es un evento
  rarísimo. Naciendo por todos lados, algunos episodios arrancan cerca de B,
  la descubren pronto, y ese conocimiento se propaga hacia atrás.
- **Inicialización optimista.** Como toda recompensa por paso es negativa,
  el cero inicial de la tabla hace que lo no visitado parezca prometedor y
  el agente se reparta por el mapa en vez de dar vueltas sobre lo conocido.

### 4.4 Limitaciones

Somos conscientes de varias limitaciones. La política aprendida vale solo
para este mapa y esta meta: si movemos B, hay que reentrenar desde cero,
porque la tabla Q no generaliza. El entorno es determinístico, así que no
sabemos qué tan robusta es la ruta si el agente pudiera "resbalar". Y no
comparamos contra la solución exacta (por ejemplo, Value Iteration o un
algoritmo de camino mínimo tipo Dijkstra), así que sabemos que la ruta es
buena pero no podemos afirmar que sea la óptima. Estas comparaciones quedan
para el trabajo final, junto con un barrido de hiperparámetros y obstáculos
reales de OpenStreetMap.

## 5. Comprensión del algoritmo

Esta sección resume, con nuestras palabras, cómo funciona Q-learning y por
qué funciona en este problema.

Todo el código lo desarrollamos en Python (NumPy para los cálculos,
Matplotlib para las figuras y rasterio para leer el DEM). Cada función lleva
su docstring explicando qué hace y por qué, y en las partes menos obvias
(el update TD en `qlearning.py`, la construcción del entorno en
`entorno.py`) dejamos comentarios línea a línea. La idea es que cualquiera
pueda leer el código de corrido y seguir el razonamiento sin necesidad de
este informe al lado.

Todo el algoritmo se reduce a una línea, el update de diferencia temporal
(TD):

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

La idea: comparar lo que el agente **creía** que valía ese movimiento,
Q(s,a), contra lo que **acaba de vivir**: la recompensa r más lo mejor que
le espera desde la celda donde cayó, γ·max Q(s',a'). La diferencia entre
ambas cosas es el error TD. Si la realidad fue mejor de lo esperado, el
número sube un poco; si fue peor, baja. Ese "un poco" lo controla α: con
α = 1 cada experiencia borraría todo lo anterior, con α = 0 no se aprendería
nada. Repetido miles de veces, la tabla converge a los valores reales.

Un detalle que nos parece el corazón del algoritmo: el objetivo usa el
**máximo** sobre las acciones siguientes, aunque el agente después haga otra
cosa (por ejemplo, explorar al azar por el ε-greedy). Es decir, el agente
aprende sobre la mejor jugada posible mientras juega distinto. Eso es lo que
hace a Q-learning **off-policy**: la política que aprende (la greedy) no es
la misma que la política con la que se comporta (la ε-greedy). La
alternativa on-policy sería SARSA, que en el update usa la acción que
realmente se tomó en s'; en nuestro caso, con entorno determinístico y ε
decayendo hasta casi cero, esperaríamos resultados parecidos, pero la
comparación formal la dejamos para el trabajo final.

El dilema exploración/explotación lo maneja el ε-greedy: sin explorar, el
agente nunca encontraría rutas mejores que la primera que le funcionó; sin
explotar, nunca aprovecharía lo aprendido. Por eso ε arranca en 1 (de joven
el agente prueba de todo) y decae hasta 0.005 (de viejo ya conoce el terreno
y va a lo seguro).

Hay un caso borde que tuvimos que tratar aparte en el código: cuando el
agente llega a la meta, el episodio termina y no hay futuro que estimar, así
que el objetivo del update es solo r, sin el término γ·max Q. Si no se hace
esto, el valor de la meta se contamina con estimaciones de un "después" que
no existe.

Por último, por qué estamos razonablemente seguros de que convergió: el MDP
es finito y determinístico, la tarea es episódica (la meta es absorbente y
desde toda celda libre existe camino a B, garantizado por el filtro BFS), y
con exploring starts más el ε-greedy todas las parejas estado-acción se
visitan muchas veces, que es la condición clásica de convergencia de
Q-learning. La evidencia empírica acompaña: la curva se aplana y la política
final llega a la meta de forma estable.

## 6. Conclusiones

Modelamos un problema de navegación sobre terreno real como un MDP tabular y
lo resolvimos con Q-learning implementado desde cero, sin librerías de RL.
El agente, sin conocer costos ni obstáculos, aprendió una política que llega
a la meta con una ruta de bajo costo que evita las pendientes. La lección
que más nos llevamos es que las decisiones de diseño importaron tanto como
el algoritmo: γ = 1 (costo total sin descontar), exploring starts y la
inicialización optimista fueron necesarias para que el aprendizaje fuera
viable en un mapa de ~2 700 estados con recompensa escasa.

## 7. Reproducibilidad del codigo

```bash
pip install -r requirements.txt
python main.py     # entrena y genera las 4 figuras en results/ (~10 s)
```

`data/dem.tif` está incluido en el repositorio; `download_dem.py` permite
re-descargar el recorte desde AWS Open Data. La semilla está fija, así que
toda corrida reproduce exactamente las figuras y números de este informe. El
código está comentado paso a paso (en especial `qlearning.py` y
`entorno.py`) para facilitar la revisión.

## Referencias

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2.ª ed.). MIT Press.
- ESA. Copernicus DEM GLO-30. Tile S17/W072, vía AWS Open Data
  (`s3://copernicus-dem-30m/`).

"""
El entorno: un tablero (gridworld) construido sobre elevación REAL del terreno.

La idea es simple: tomamos un mapa de alturas del volcán Misti y lo
convertimos en un tablero donde un agente camina celda por celda.
Caminar por terreno plano es barato; subir laderas cuesta más; y las
zonas muy empinadas directamente no se pueden pisar.

Modelamos el problema como un MDP (Proceso de Decisión de Markov),
que en la práctica se reduce a cuatro ingredientes:

- Estados (S): dónde puede estar el agente. Aquí, cada celda del mapa,
  identificada por (fila, columna).
- Acciones (A): qué puede hacer en cada celda. Aquí, moverse en cruz:
  {0: arriba, 1: abajo, 2: izquierda, 3: derecha}.
- Transición (P): qué pasa cuando hace algo. Aquí es determinística
  (sin azar): el agente se mueve a la celda vecina que eligió; si esa
  celda es borde del mapa u obstáculo, rebota y se queda donde estaba.
- Recompensa (R): la "nota" que recibe por cada movimiento:
    * paso normal      -> paga el costo de la celda que pisa (negativo)
    * chocar obstáculo -> castigo fuerte (y no se mueve)
    * llegar a la meta -> premio grande y se acaba el episodio

El costo de cada celda lo calculamos a partir de la inclinación del terreno:

    costo = 1 + peso_pendiente * pendiente

Una celda plana cuesta 1 (el costo base de dar un paso). Una ladera de
30° (pendiente ~0.58) cuesta ~5.6: casi seis veces más caro que caminar
en plano. Y si la pendiente pasa cierto umbral, la celda se marca como
obstáculo: nadie sube una pared.
"""

import numpy as np
import rasterio

# Cada acción es un desplazamiento (cuánto cambia la fila, cuánto la columna).
ACCIONES = {
    0: (-1, 0),   # arriba
    1: (1, 0),    # abajo
    2: (0, -1),   # izquierda
    3: (0, 1),    # derecha
}
NUM_ACCIONES = len(ACCIONES)
FLECHAS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


class TerrenoEnv:
    """Gridworld de navegación de costo mínimo sobre un DEM real."""

    def __init__(
        self,
        ruta_dem="data/dem.tif",
        tam_celda=30.0,          # cada celda del DEM mide 30 m (Copernicus GLO-30)
        factor_reduccion=2,      # juntamos celdas de 2x2 -> celdas de 60 m,
                                 # cuadrícula ~54x54. Con menos estados la
                                 # tabla Q es más chica y se aprende más rápido.
        peso_pendiente=8.0,      # qué tanto castiga la inclinación al caminar
        umbral_obstaculo=0.60,   # pendiente > 0.60 (~31°): demasiado empinado, no se pasa
        recompensa_meta=100.0,   # premio por llegar a B
        castigo_obstaculo=50.0,  # multa por intentar pisar un obstáculo
        max_pasos=3000,          # si el episodio se alarga demasiado, lo cortamos
        inicio=None,             # (fila, col); si es None se elige automático
        meta=None,
    ):
        # --- 1. Cargar la elevación real desde el GeoTIFF ---
        # El DEM es literalmente una matriz de alturas: cada celda dice
        # a cuántos metros sobre el mar está ese pedazo de terreno.
        with rasterio.open(ruta_dem) as src:
            self.elevacion = src.read(1).astype(float)

        # Bajar la resolución promediando bloques de f x f celdas.
        # Con f=2 el mapa pasa de 108x108 a 54x54 celdas: es el mismo
        # terreno pero "pixeleado" más grueso. La ganancia: la tabla Q
        # queda 4 veces más chica y el entrenamiento va mucho más rápido.
        f = factor_reduccion
        if f > 1:
            filas, cols = (self.elevacion.shape[0] // f) * f, (self.elevacion.shape[1] // f) * f
            self.elevacion = (
                self.elevacion[:filas, :cols]
                .reshape(filas // f, f, cols // f, f)
                .mean(axis=(1, 3))
            )
            tam_celda *= f
        self.filas, self.columnas = self.elevacion.shape
        self.tam_celda = tam_celda  # lado de cada celda en metros (tras la reducción)

        # --- 2. Pendiente del terreno ---
        # Para saber qué tan inclinada está cada celda usamos np.gradient,
        # que compara la altura de cada celda con la de sus vecinas:
        # "cuántos metros sube el terreno por cada metro avanzado" (por
        # eso dividimos entre el tamaño de celda). Como el terreno puede
        # inclinarse en dos direcciones (norte-sur y este-oeste),
        # combinamos ambas con Pitágoras para la inclinación total.
        gy, gx = np.gradient(self.elevacion, tam_celda)
        self.pendiente = np.sqrt(gx ** 2 + gy ** 2)

        # --- 3. Costo por celda y obstáculos ---
        # Traducimos la pendiente a "precio del paso": plano cuesta 1,
        # y cada punto de pendiente lo encarece. Lo muy empinado ni se
        # paga: se marca directamente como obstáculo.
        self.costo = 1.0 + peso_pendiente * self.pendiente
        self.obstaculo = self.pendiente > umbral_obstaculo

        self.recompensa_meta = recompensa_meta
        self.castigo_obstaculo = castigo_obstaculo
        self.max_pasos = max_pasos

        # --- 4. Inicio y meta ---
        # Por defecto los ponemos en esquinas opuestas del mapa (o en la
        # celda libre más cercana a cada esquina, si la esquina justo cae
        # en un obstáculo). Así la ruta tiene que cruzar todo el terreno.
        self.meta = meta or self._celda_libre_cercana(
            (self.filas - 4, self.columnas - 4)
        )

        # Detalle importante: puede haber "islas" de celdas libres
        # completamente rodeadas de obstáculos. Si el agente naciera ahí,
        # jamás podría llegar a la meta. Para que el problema siempre
        # tenga solución, marcamos esas islas también como obstáculo:
        # desde cualquier celda libre que quede, sí existe un camino a B.
        self.obstaculo |= ~self._alcanzables_desde_meta()

        self.inicio = inicio or self._celda_libre_cercana((3, 3))

        self.estado = None
        self.pasos = 0

    def _celda_libre_cercana(self, celda):
        """
        Devuelve la celda libre (sin obstáculo) más cercana a la pedida.
        La usamos porque el punto exacto que elegimos para inicio o meta
        puede caer en una ladera intransitable; en ese caso lo movemos
        a su vecino caminable más próximo.
        """
        objetivo = np.array(celda)
        libres = np.argwhere(~self.obstaculo)
        distancias = np.abs(libres - objetivo).sum(axis=1)
        return tuple(libres[distancias.argmin()])

    def _alcanzables_desde_meta(self):
        """
        Calcula desde qué celdas del mapa se puede llegar a la meta.

        Lo averiguamos "inundando" el mapa desde la meta: partimos de B
        y vamos marcando vecino libre tras vecino libre, como agua que
        se expande (esto es una búsqueda en anchura, BFS). Toda celda
        que el agua toca puede llegar a B; las que quedan secas, no.

        Devuelve una máscara booleana: True = desde aquí sí hay camino.
        """
        alcanzable = np.zeros_like(self.obstaculo)
        alcanzable[self.meta] = True
        frontera = [self.meta]
        while frontera:
            fila, col = frontera.pop()
            for df, dc in ACCIONES.values():
                vecino = (fila + df, col + dc)
                if (
                    0 <= vecino[0] < self.filas
                    and 0 <= vecino[1] < self.columnas
                    and not self.obstaculo[vecino]
                    and not alcanzable[vecino]
                ):
                    alcanzable[vecino] = True
                    frontera.append(vecino)
        return alcanzable

    def reset(self, inicio=None):
        """
        Empieza un episodio nuevo: devuelve al agente a la casilla de
        salida y pone el contador de pasos en cero.

        Por defecto arranca en A, pero se puede pasar otra celda. Eso lo
        aprovecha el entrenamiento con "exploring starts": episodios que
        nacen en celdas aleatorias para conocer todo el mapa más rápido.
        """
        self.estado = inicio or self.inicio
        self.pasos = 0
        return self.estado

    def step(self, accion):
        """
        El agente da UN paso. Devuelve (estado_siguiente, recompensa, terminado).

        Esta función es el corazón de nuestro entorno: aquí decidimos a
        dónde se mueve el agente y qué recompensa le damos. Es la única
        vía por la que el agente "conoce" el mundo: todo lo que aprende
        sale de lo que esta función le va devolviendo.
        """
        df, dc = ACCIONES[accion]
        fila, col = self.estado
        nueva = (fila + df, col + dc)

        fuera = not (0 <= nueva[0] < self.filas and 0 <= nueva[1] < self.columnas)

        if fuera or self.obstaculo[nueva]:
            # Intentó salirse del mapa o pisar un obstáculo: se lleva la
            # multa y se queda donde estaba (rebota).
            recompensa = -self.castigo_obstaculo
            nueva = self.estado
        elif nueva == self.meta:
            # ¡Llegó a B! Premio grande y el episodio termina.
            recompensa = self.recompensa_meta
        else:
            # Paso normal: paga el "peaje" de la celda que pisa.
            # Más pendiente, peaje más caro.
            recompensa = -self.costo[nueva]

        self.estado = nueva
        self.pasos += 1

        terminado = (nueva == self.meta) or (self.pasos >= self.max_pasos)
        return nueva, recompensa, terminado

    # ------------------------------------------------------------------
    # Utilidades usadas por los algoritmos y los gráficos
    # ------------------------------------------------------------------

    def seguir_politica(self, politica, max_pasos=None):
        """
        Usamos esta función como el "examen final" del agente: recorremos
        el mapa desde A obedeciendo la política al pie de la letra (la
        política es una matriz que dice, para cada celda, qué acción
        tomar). Sin exploración, sin azar.

        Devuelve las celdas visitadas, el costo total del camino y si
        llegó o no a la meta. Con esto evaluamos y dibujamos la ruta
        que el agente aprendió.
        """
        max_pasos = max_pasos or self.max_pasos
        celda = self.inicio
        camino = [celda]
        costo_total = 0.0

        for _ in range(max_pasos):
            if celda == self.meta:
                break
            df, dc = ACCIONES[politica[celda]]
            siguiente = (celda[0] + df, celda[1] + dc)
            fuera = not (
                0 <= siguiente[0] < self.filas and 0 <= siguiente[1] < self.columnas
            )
            if fuera or self.obstaculo[siguiente]:
                break  # la política manda contra un muro: hasta aquí llegó la ruta
            costo_total += self.costo[siguiente]
            celda = siguiente
            camino.append(celda)

        llego = celda == self.meta
        return camino, costo_total, llego

"""
El entorno del trabajo final: mismo terreno real del Misti, problema más duro.

Tres cambios respecto al parcial, y los tres apuntan a lo mismo — que una
tabla ya no alcance y la red neuronal sea necesaria:

1. RESOLUCIÓN COMPLETA. Ya no promediamos el DEM a 54x54: usamos las
   108x108 celdas originales de 30 m. Más estados, más detalle del terreno.

2. INICIO Y META ALEATORIOS. Cada episodio sortea un par (A, B) distinto.
   El agente ya no aprende UNA ruta: aprende a navegar entre cualquier par
   de puntos. Una tabla necesitaría una fila por cada pareja
   (posición, meta) — millones — así que este cambio es el que mata a la
   tabla y justifica DQN.

3. OBSERVACIÓN LOCAL. El agente ya no "sabe" su coordenada: VE un parche
   de 9x9 celdas de pendiente alrededor suyo más un vector hacia la meta.
   83 números, sin importar el tamaño del mapa. Terrenos parecidos producen
   observaciones parecidas: eso es lo que la red puede generalizar.

Extra opcional: transición estocástica (resbalón). Con probabilidad
proporcional a la pendiente, el paso sale hacia un lado en vez de a donde
se apuntó. Hace el problema más realista y menos memorizable.
"""

import numpy as np
import rasterio

ACCIONES = {
    0: (-1, 0),   # arriba
    1: (1, 0),    # abajo
    2: (0, -1),   # izquierda
    3: (0, 1),    # derecha
}
NUM_ACCIONES = len(ACCIONES)


class TerrenoDRLEnv:
    """Gridworld sobre DEM real con metas aleatorias y observación local."""

    def __init__(
        self,
        ruta_dem="../data/dem.tif",
        tam_celda=30.0,
        peso_pendiente=8.0,
        umbral_obstaculo=0.60,
        parche=9,                # lado de la ventana que el agente ve (9x9)
        max_pasos=500,
        dist_minima=15,          # separación mínima (en celdas) entre A y B
        prob_resbalon=0.0,       # 0 = determinístico; >0 = resbala según pendiente
        k_shaping=0.2,           # bono por acercarse a la meta (0 = apagado)
        gamma=0.99,              # el γ del agente (el shaping potencial lo usa)
        meta_fija=True,          # True = una sola meta para todo (estable);
                                 # False = meta aleatoria por episodio (más duro)
        semilla=0,
    ):
        # --- Terreno: idéntico al parcial, pero sin reducir resolución ---
        with rasterio.open(ruta_dem) as src:
            self.elevacion = src.read(1).astype(float)
        self.filas, self.columnas = self.elevacion.shape

        gy, gx = np.gradient(self.elevacion, tam_celda)
        self.pendiente = np.sqrt(gx ** 2 + gy ** 2)
        self.costo = 1.0 + peso_pendiente * self.pendiente
        self.obstaculo = self.pendiente > umbral_obstaculo

        # --- Mapa que el agente "ve" en su parche ---
        # Pendiente recortada a [0, 1], con obstáculos forzados a 1.0 (para
        # el agente, un obstáculo se ve igual que una pared vertical). El
        # borde del mapa también se verá como 1.0 (padding al extraer).
        self.vista = np.clip(self.pendiente / umbral_obstaculo, 0.0, 1.0)
        self.vista[self.obstaculo] = 1.0

        assert parche % 2 == 1, "el parche debe tener lado impar (agente al centro)"
        self.parche = parche
        self.radio = parche // 2
        # observación = parche aplanado + vector a la meta normalizado
        self.dim_obs = parche * parche + 2

        self.max_pasos = max_pasos
        self.dist_minima = dist_minima
        self.prob_resbalon = prob_resbalon
        self.k_shaping = k_shaping
        self.gamma = gamma
        self.rng = np.random.default_rng(semilla)

        # --- Celdas donde pueden caer A y B ---
        # Nos quedamos con la componente conexa más grande de celdas libres:
        # así cualquier par (A, B) que sorteemos tiene camino garantizado.
        self.libres = self._componente_mas_grande()
        if len(self.libres) < 2:
            raise RuntimeError("el mapa no tiene suficientes celdas libres conectadas")

        # Meta fija (modo estable): una sola celda destino para todos los
        # episodios. La ponemos en la esquina inferior-derecha del área libre,
        # como en el parcial, para que las rutas crucen el terreno. Con una
        # sola meta, la función de valor es un único campo suave (como la V
        # tabular que sí convergía) y DQN entrena estable. El inicio SÍ es
        # aleatorio: el agente debe aprender a llegar desde cualquier celda,
        # así la red generaliza y la observación local importa.
        self.meta_fija = None
        if meta_fija:
            objetivo = np.array([self.filas - 4, self.columnas - 4])
            d = np.abs(self.libres - objetivo).sum(axis=1)
            self.meta_fija = tuple(self.libres[d.argmin()])

        self.estado = None
        self.meta = None
        self.pasos = 0

    def _componente_mas_grande(self):
        """BFS sobre celdas libres; devuelve las celdas de la componente mayor."""
        visitado = np.zeros_like(self.obstaculo)
        mejor = []
        for celda in map(tuple, np.argwhere(~self.obstaculo)):
            if visitado[celda]:
                continue
            frontera, comp = [celda], [celda]
            visitado[celda] = True
            while frontera:
                fila, col = frontera.pop()
                for df, dc in ACCIONES.values():
                    v = (fila + df, col + dc)
                    if (
                        0 <= v[0] < self.filas and 0 <= v[1] < self.columnas
                        and not self.obstaculo[v] and not visitado[v]
                    ):
                        visitado[v] = True
                        frontera.append(v)
                        comp.append(v)
            if len(comp) > len(mejor):
                mejor = comp
        return np.array(mejor)

    # ------------------------------------------------------------------

    def reset(self, inicio=None, meta=None, dist_maxima=None):
        """
        Episodio nuevo: sortea inicio y meta (o usa los que le pasen, para
        evaluación). Devuelve la primera observación.

        dist_maxima limita qué tan lejos puede caer la meta del inicio. Lo
        usa el CURRICULUM del entrenamiento: al principio metas cercanas
        (fáciles de alcanzar, para que el agente aprenda los estados de
        salida), y se va soltando hasta cubrir todo el mapa. Sin esto, el
        agente casi nunca practica salidas lejanas y su política greedy se
        traba en ping-pong justo en esos estados (lo que medimos).
        """
        # Meta: la que pasen (evaluación) o la fija; en modo meta aleatoria
        # (meta_fija=None) queda None y se sortea abajo.
        if meta is None:
            meta = self.meta_fija

        # Si inicio y meta ya vienen dados, usarlos tal cual (evaluación).
        if inicio is not None and meta is not None:
            self.estado, self.meta, self.pasos = tuple(inicio), tuple(meta), 0
            return self._observar()

        # Si no, sortear lo que falte (inicio, y meta si es aleatoria)
        # respetando la distancia mínima y el tope del curriculum.
        while True:
            ini = tuple(inicio) if inicio is not None \
                else tuple(self.libres[self.rng.integers(len(self.libres))])
            met = tuple(meta) if meta is not None \
                else tuple(self.libres[self.rng.integers(len(self.libres))])
            d = abs(ini[0] - met[0]) + abs(ini[1] - met[1])
            if d >= self.dist_minima and (dist_maxima is None or d <= dist_maxima):
                break
        self.estado, self.meta, self.pasos = ini, met, 0
        return self._observar()

    def _observar(self):
        """
        Construye los 83 números que ve el agente:
        el parche de "vista" (pendiente 0-1, obstáculo/borde = 1) centrado
        en él, más el vector hacia la meta normalizado a [-1, 1].
        """
        r = self.radio
        fila, col = self.estado
        # padding con 1.0: más allá del borde del mapa "todo es pared"
        vista = np.pad(self.vista, r, constant_values=1.0)
        patch = vista[fila:fila + 2 * r + 1, col:col + 2 * r + 1]
        delta = np.array(
            [(self.meta[0] - fila) / self.filas, (self.meta[1] - col) / self.columnas]
        )
        return np.concatenate([patch.ravel(), delta]).astype(np.float32)

    def _dist_meta(self, celda):
        return abs(celda[0] - self.meta[0]) + abs(celda[1] - self.meta[1])

    def step(self, accion):
        """
        Un paso. Devuelve (observación, recompensa, terminado, llego_a_meta).

        Recompensas a escala /10 respecto al parcial (a las redes les
        sientan mejor números chicos): paso ~ -0.1..-0.6, choque -5,
        meta +10.

        REWARD SHAPING (la lección más importante que nos dejó este
        entorno): con meta aleatoria en 108x108, el agente casi nunca
        encuentra B por azar (~1 de 1000 episodios), así que el premio de la
        meta no existe en la práctica y no hay nada que aprender. Le damos un
        bono denso por ACERCARSE: recompensa += k·(dist_antes − dist_después).

        OJO — probamos primero el shaping potencial "de libro" (Ng et al.
        1999): γ·Φ(s') − Φ(s) con Φ = −k·dist. Falló feo: con γ<1 ese bono
        deja un residuo (1−γ)·k·dist POSITIVO por estar lejos, así que hacer
        ping-pong lejos de la meta GANABA recompensa (+0.2 a +0.4 por ciclo a
        distancia ≥100). El agente aprendía a oscilar lejos en vez de llegar;
        su política greedy se trababa en 2-ciclos y empeoraba con el
        entrenamiento. El shaping de progreso k·Δdist telescopea a 0 en
        cualquier ciclo, así que el ping-pong queda penalizado por el costo
        del paso. Cuesta un poco de garantía teórica (no es potencial puro),
        pero es lo que converge — y la historia es oro para el informe.
        """
        # Resbalón: en pendiente, a veces el paso sale torcido. La
        # probabilidad crece con la inclinación de la celda actual.
        if self.prob_resbalon > 0:
            p = self.prob_resbalon * self.vista[self.estado]
            if self.rng.random() < p:
                accion = int(self.rng.integers(NUM_ACCIONES))

        df, dc = ACCIONES[accion]
        fila, col = self.estado
        nueva = (fila + df, col + dc)

        fuera = not (0 <= nueva[0] < self.filas and 0 <= nueva[1] < self.columnas)
        if fuera or self.obstaculo[nueva]:
            recompensa = -5.0
            nueva = self.estado
        elif nueva == self.meta:
            recompensa = 10.0
        else:
            recompensa = -self.costo[nueva] / 10.0

        # shaping de progreso: +k por cada celda que se acerca a la meta,
        # −k por cada una que se aleja. Sin residuo por distancia -> sin la
        # trampa de ping-pong del shaping potencial (ver docstring).
        if self.k_shaping > 0:
            recompensa += self.k_shaping * (
                self._dist_meta(self.estado) - self._dist_meta(nueva)
            )

        self.estado = nueva
        self.pasos += 1
        llego = nueva == self.meta
        terminado = llego or (self.pasos >= self.max_pasos)
        return self._observar(), recompensa, terminado, llego

    # ------------------------------------------------------------------

    def costo_camino(self, camino):
        """Costo total (escala del parcial) de una lista de celdas."""
        return float(sum(self.costo[c] for c in camino[1:]))

r"""
Q-learning implementado desde cero.

Punto de partida: nuestro agente NO conoce el mapa. No sabe dónde están
los obstáculos ni cuánto cuesta cada celda. Lo único que puede hacer es
moverse, recibir recompensas y sacar conclusiones. Por eso Q-learning es
un método MODEL-FREE: aprende sin tener un modelo del mundo, solo con
experiencia (prueba y error).

Lo que aprende es una tabla, la tabla Q. Para cada celda y cada acción
guarda un número que responde: "si estoy aquí y hago esto, ¿qué tan
bien me va a ir en total?". Al principio la tabla está en cero (no sabe
nada); con cada paso los números se van afinando.

Todo el algoritmo se reduce a UNA línea, el update TD (diferencia
temporal):

    Q(s,a) <- Q(s,a) + α · [ r + γ · max_a' Q(s', a') - Q(s,a) ]
                           \___________________________________/
                                       error TD (δ)

En palabras simples: compara lo que el agente creía que valía ese
movimiento (Q(s,a)) contra lo que acaba de vivir (la recompensa r más
lo mejor que le espera desde la celda donde cayó). Si la realidad fue
mejor de lo esperado, el número sube un poco; si fue peor, baja. Ese
"un poco" lo controla α. Repetido miles de veces, la tabla converge
a los valores reales.
"""

import numpy as np

from entorno import NUM_ACCIONES


def politica_epsilon_greedy(Q, estado, epsilon, rng):
    """
    Así decide el agente su próxima acción: tira una moneda cargada.

    - Con probabilidad ε: acción al azar (EXPLORAR — probar cosas
      nuevas, aunque parezcan malas, por si hay algo mejor escondido).
    - El resto del tiempo: la mejor acción según su tabla Q
      (EXPLOTAR — usar lo que ya sabe).

    Sin explorar nunca encontraría rutas mejores; sin explotar nunca
    aprovecharía lo aprendido. ε es la perilla con la que regulamos
    ese balance.
    """
    if rng.random() < epsilon:
        return int(rng.integers(NUM_ACCIONES))
    return int(np.argmax(Q[estado]))


def entrenar(
    env,
    episodios=6000,
    alpha=0.3,                # α: qué tanto corrige cada experiencia nueva
    gamma=0.99,               # γ: cuánto pesa el futuro frente al presente
    epsilon=1.0,              # ε inicial = 1: al arrancar, TODO es exploración
    epsilon_min=0.005,        # ε final casi 0: al terminar, casi pura explotación
    epsilon_decay=0.999,      # cada episodio, ε se multiplica por esto (baja suave)
    exploring_starts=True,    # episodios desde celdas aleatorias (ver abajo)
    semilla=0,
):
    """
    Entrena el agente con Q-learning y devuelve:
      Q                 : tabla (filas, columnas, 4) con los valores aprendidos
      recompensas       : recompensa total de cada episodio (curva de aprendizaje)
      pasos_por_episodio: largo de cada episodio
    """
    rng = np.random.default_rng(semilla)

    # La tabla Q arranca toda en cero: el agente no sabe nada.
    # Aquí aprovechamos un truco: como cada paso da recompensa negativa,
    # un cero "se ve bien" comparado con lo ya visitado. Eso hace que
    # las celdas nunca probadas parezcan prometedoras y el agente vaya
    # solo hacia ellas. Se llama inicialización OPTIMISTA y nos regala
    # exploración gratis.
    Q = np.zeros((env.filas, env.columnas, NUM_ACCIONES))

    celdas_libres = np.argwhere(~env.obstaculo)

    recompensas = []
    pasos_por_episodio = []

    for _ in range(episodios):
        # EXPLORING STARTS (Sutton & Barto): hacemos que cada episodio
        # nazca en una celda aleatoria del mapa, no siempre en A. La
        # razón: si el agente siempre saliera de A, tendría que cruzar
        # ~100 celdas de pura casualidad antes de descubrir la meta por
        # primera vez. Naciendo por todos lados, algunos episodios
        # arrancan cerca de B, la descubren pronto, y ese conocimiento
        # se propaga hacia atrás por todo el mapa. La evaluación final,
        # eso sí, siempre la hacemos desde A.
        inicio = None
        if exploring_starts:
            inicio = tuple(celdas_libres[rng.integers(len(celdas_libres))])
        estado = env.reset(inicio)

        total = 0.0
        terminado = False

        while not terminado:
            accion = politica_epsilon_greedy(Q, estado, epsilon, rng)
            estado_sig, recompensa, terminado = env.step(accion)
            total += recompensa

            if terminado and estado_sig == env.meta:
                # Llegó a la meta: no hay futuro que estimar, así que
                # "lo que vale este movimiento" es solo la recompensa.
                objetivo = recompensa
            else:
                # Lo vivido = recompensa inmediata + lo MEJOR que se
                # puede hacer desde la celda donde cayó. Nota: usa el
                # máximo aunque el agente después haga otra cosa (quizá
                # explore al azar). Aprender sobre la mejor jugada
                # mientras se juega distinto es lo que hace a
                # Q-learning "off-policy".
                objetivo = recompensa + gamma * np.max(Q[estado_sig])

            # El update TD: acercar la creencia vieja una fracción α
            # hacia lo recién vivido. Ni borrar todo lo anterior (α=1)
            # ni ignorar la experiencia nueva (α=0): un punto medio.
            Q[estado][accion] += alpha * (objetivo - Q[estado][accion])

            estado = estado_sig

        # Terminado el episodio, bajamos ε un poquito: de joven el
        # agente prueba de todo; de viejo, ya conoce el terreno y va
        # a lo seguro.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        recompensas.append(total)
        pasos_por_episodio.append(env.pasos)

    return Q, np.array(recompensas), np.array(pasos_por_episodio)


def politica_desde_Q(Q):
    """
    De la tabla Q a un plan de acción: en cada celda, quedarse con la
    acción de mayor valor. El resultado es un "mapa de instrucciones"
    (la política greedy): estés donde estés, te dice hacia dónde ir.
    """
    return np.argmax(Q, axis=2)

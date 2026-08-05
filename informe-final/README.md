# Informe final — formato IEEE

Informe del trabajo final en LaTeX, formato IEEE conference (clase `IEEEtran`).

## Archivos

- `main.tex` — archivo maestro: preámbulo, título, resumen y `\input{}` de las secciones.
- `secciones/` — una sección por archivo:
  - `introduccion.tex`, `marco-teorico.tex`, `metodologia.tex`,
    `experimentos.tex`, `discusion.tex`, `conclusiones.tex`, `figuras.tex`.
- `figuras/` — las 8 figuras (PNG) que usa el informe.

Al compilar en Overleaf sube **toda** la carpeta `informe-final/` (con `secciones/` y
`figuras/`), no solo `main.tex`.

## Cómo compilar en Overleaf

1. Entra a [overleaf.com](https://www.overleaf.com) → **New Project → Upload Project**.
2. Sube `main.tex` y la carpeta `figuras/` (comprime `informe-final/` en un `.zip` y súbelo).
3. Overleaf detecta `IEEEtran` automáticamente (viene incluido). Compilador: **pdfLaTeX**.
4. Compila. Debe salir el PDF en formato IEEE de 2 columnas.

> Alternativa: abre el template IEEE de la profesora
> (`https://www.overleaf.com/read/swnqjnsrsfgt`), haz "Copy Project", y pega el contenido
> de `main.tex` en su archivo principal. Sube la carpeta `figuras/`.

## Cómo regenerar las figuras

Las figuras se generan entrenando los modelos:

```bash
cd ../dqn
python gen_figuras_informe.py     # guarda los PNG en ../informe-final/figuras/
```

También puedes exportarlas de los notebooks en Colab (clic derecho sobre cada gráfico →
guardar imagen) para tener las versiones con más episodios.

## Pendientes antes de entregar

- [ ] Poner los nombres de los 5 integrantes en `main.tex` (bloque `\author`).
- [ ] Rellenar los números marcados con `% TODO` (están en `figuras/numeros.txt`).
- [ ] Confirmar con la profesora: ¿español o inglés? ¿límite de páginas?
- [ ] Revisar que todas las figuras se vean bien en el PDF compilado.

## Fuente de los contenidos

- `../HALLAZGOS_INFORME.md` — todos los resultados y su justificación.
- `../conceptos-preguntas.md` — los conceptos explicados.

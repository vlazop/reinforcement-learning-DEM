"""
Descarga un recorte del DEM Copernicus GLO-30 (elevación global a 30 m)
directamente desde AWS Open Data, sin necesidad de login.

Zona elegida: faldas del volcán Misti, Arequipa, Perú (~3 x 3 km).
Es una zona con pendientes variadas: quebradas, laderas suaves y
zonas empinadas — ideal para que el agente tenga decisiones que tomar.

El resultado se guarda en data/dem.tif para que el resto del proyecto
corra siempre igual, sin depender de la red.

Uso:
    python download_dem.py
"""

import rasterio
from rasterio.windows import from_bounds

# Tile de Copernicus DEM que cubre Arequipa (lat -17..-16, lon -72..-71).
# Es un COG (Cloud Optimized GeoTIFF): permite leer solo la ventana
# que necesitamos sin descargar el tile completo (~50 MB).
URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_S17_00_W072_00_DEM/"
    "Copernicus_DSM_COG_10_S17_00_W072_00_DEM.tif"
)

# Bounding box de la zona de estudio (lon_min, lat_min, lon_max, lat_max).
# ~3 x 3 km en las faldas del Misti, al noreste de la ciudad de Arequipa.
BBOX = (-71.475, -16.375, -71.445, -16.345)

SALIDA = "data/dem.tif"


def main():
    print(f"Leyendo ventana {BBOX} desde AWS Open Data...")
    with rasterio.open(URL) as src:
        ventana = from_bounds(*BBOX, transform=src.transform)
        elevacion = src.read(1, window=ventana)
        transform = src.window_transform(ventana)

        perfil = src.profile.copy()
        perfil.update(
            height=elevacion.shape[0],
            width=elevacion.shape[1],
            transform=transform,
        )

    with rasterio.open(SALIDA, "w", **perfil) as dst:
        dst.write(elevacion, 1)

    print(f"Guardado {SALIDA}: {elevacion.shape[0]} x {elevacion.shape[1]} celdas")
    print(f"Elevación: min {elevacion.min():.0f} m, max {elevacion.max():.0f} m")


if __name__ == "__main__":
    main()

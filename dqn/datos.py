"""
Ubica el DEM (modelo de elevación) local, o lo descarga de AWS Open Data.
Código común: los 3 notebooks llaman a ruta_dem() y no se preocupan por dónde
está el archivo. El recorte es de las faldas del Misti (Copernicus GLO-30, 30 m).
"""

import os

URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
       "Copernicus_DSM_COG_10_S17_00_W072_00_DEM/"
       "Copernicus_DSM_COG_10_S17_00_W072_00_DEM.tif")
BBOX = (-71.475, -16.375, -71.445, -16.345)   # lon/lat: ~3x3 km en el Misti


def ruta_dem(base="."):
    """
    Devuelve la ruta a data/dem.tif si existe (busca en varios lugares
    relativos a 'base'); si no, descarga el recorte desde AWS y lo guarda.
    """
    candidatos = ["data/dem.tif", "../data/dem.tif", "dem.tif"]
    for r in [os.path.join(base, c) for c in candidatos]:
        if os.path.exists(r):
            return r

    import rasterio
    from rasterio.windows import from_bounds
    salida = os.path.join(base, "dem.tif")
    print("Descargando recorte del DEM desde AWS Open Data...")
    with rasterio.open(URL) as src:
        ventana = from_bounds(*BBOX, transform=src.transform)
        elev = src.read(1, window=ventana)
        perfil = src.profile.copy()
        perfil.update(height=elev.shape[0], width=elev.shape[1],
                      transform=src.window_transform(ventana))
    with rasterio.open(salida, "w", **perfil) as dst:
        dst.write(elev, 1)
    print(f"Guardado {salida}")
    return salida

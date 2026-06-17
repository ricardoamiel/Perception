from __future__ import annotations

import os
from typing import Iterable

import pandas as pd

from .geo_utils import transformar_a_mercator, calcular_centroides_por_cluster, mercator_a_lonlat
from .clustering import ejecutar_hdbscan
from .plotting import plot_clusters_map


def _filtrar_distritos(df: pd.DataFrame, distritos: Iterable[str]) -> pd.DataFrame:
    mask = df["DISTRITO"].astype(str).isin(list(distritos))
    return df[mask].copy()


def _subconjunto_anio_distrito(df: pd.DataFrame, anio: int, distrito: str) -> pd.DataFrame:
    m1 = df["YEAR"].astype(int) == int(anio)
    m2 = df["DISTRITO"].astype(str) == str(distrito)
    return df[m1 & m2].copy()


def ejecutar_pipeline(
    df: pd.DataFrame,
    distritos: Iterable[str],
    anios: Iterable[int],
    dir_figuras: str,
    min_cluster_size: int = 15,
    min_samples: int | None = None,
    basemap_source: str | None = "CartoDB.Positron",
    basemap_alpha: float = 0.4,
    overlay_source: str | None = "Stamen.TonerLines",
    overlay_alpha: float = 0.5,
):
    df_f = _filtrar_distritos(df, distritos)

    registros_resumen = []
    detalles_por_grupo: dict[tuple[int, str], pd.DataFrame] = {}

    for anio in anios:
        for distrito in distritos:
            sub = _subconjunto_anio_distrito(df_f, anio, distrito)
            if sub.empty:
                continue

            # Proyectar a EPSG:3857 para distancias euclidianas y mapas de fondo
            sub_xy = transformar_a_mercator(sub[["longitude", "latitude"]])

            # Ejecutar HDBSCAN
            etiquetado = ejecutar_hdbscan(
                sub_xy, min_cluster_size=min_cluster_size, min_samples=min_samples
            )

            # Calcular centroides (x,y)
            centroides = calcular_centroides_por_cluster(etiquetado)
            if not centroides.empty:
                centroides_lonlat = mercator_a_lonlat(
                    centroides.rename(columns={"centroid_x": "x", "centroid_y": "y"})
                )
                centroides["centroid_lon"] = centroides_lonlat["longitude"]
                centroides["centroid_lat"] = centroides_lonlat["latitude"]
            else:
                centroides["centroid_lon"] = []
                centroides["centroid_lat"] = []

            # Resumen por cluster
            if not centroides.empty:
                cent = centroides.copy()
                cent.insert(0, "DISTRITO", distrito)
                cent.insert(0, "YEAR", anio)
                registros_resumen.append(cent)

            # Exportar imagen
            titulo = f"Clusters HDBSCAN - {distrito} - {anio}"
            nombre = f"{anio}_{distrito.replace(' ', '_')}.png"
            ruta_img = os.path.join(dir_figuras, nombre)
            plot_clusters_map(
                etiquetado,
                centroides,
                titulo,
                ruta_img,
                basemap_source=basemap_source,
                basemap_alpha=basemap_alpha,
                overlay_source=overlay_source,
                overlay_alpha=overlay_alpha,
            )

            # Guardar detalle de puntos + etiquetas
            detalle = sub.copy()
            detalle_xy = etiquetado[["x", "y", "cluster", "probability"]]
            detalle = pd.concat([detalle.reset_index(drop=True), detalle_xy.reset_index(drop=True)], axis=1)
            detalles_por_grupo[(anio, distrito)] = detalle

    df_resumen = (
        pd.concat(registros_resumen, ignore_index=True) if registros_resumen else pd.DataFrame()
    )

    return {"resumen": df_resumen, "detalles": detalles_por_grupo}

from __future__ import annotations

import pandas as pd
import numpy as np


def calcular_centroides_por_cluster(df_xy: pd.DataFrame, etiqueta_col: str = "cluster") -> pd.DataFrame:
    # df_xy debe contener columnas: x, y, cluster
    agrupado = (
        df_xy[df_xy[etiqueta_col] >= 0]
        .groupby(etiqueta_col)
        .agg(centroid_x=("x", "mean"), centroid_y=("y", "mean"), n_puntos=("x", "size"))
        .reset_index()
    )
    return agrupado


def transformar_a_mercator(df_lonlat: pd.DataFrame) -> pd.DataFrame:
    # Aproximación de proyección Web Mercator (EPSG:3857) sin geopandas, usando fórmulas
    # Recomendado para integrarse con contextily cuando esté disponible.
    # Fórmulas de proyección Web Mercator
    origen = df_lonlat.copy()
    lon = np.radians(origen["longitude"].astype(float).to_numpy())
    lat = np.radians(origen["latitude"].astype(float).to_numpy())

    R = 6378137.0  # radio WGS84
    x = R * lon
    # Evitar valores extremos en latitud (clamp en ~85.05113°)
    max_lat = np.radians(85.05113)
    lat = np.clip(lat, -max_lat, max_lat)
    y = R * np.log(np.tan(np.pi / 4.0 + lat / 2.0))

    origen["x"] = x
    origen["y"] = y
    return origen


def mercator_a_lonlat(df_xy: pd.DataFrame) -> pd.DataFrame:
    R = 6378137.0
    lon = (df_xy["x"].to_numpy() / R)
    lat = (2.0 * np.arctan(np.exp(df_xy["y"].to_numpy() / R)) - np.pi / 2.0)
    out = df_xy.copy()
    out["longitude"] = np.degrees(lon)
    out["latitude"] = np.degrees(lat)
    return out


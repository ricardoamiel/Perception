from __future__ import annotations

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_provider(cx, provider):
    if provider is None:
        return None
    # Si se pasa un objeto provider directamente
    if not isinstance(provider, str):
        return provider
    # Si se pasa como string tipo "CartoDB.Positron" o "Stamen.TonerLines"
    base = cx.providers
    for part in provider.split('.'):
        base = getattr(base, part)
    return base


def _try_add_basemap(ax, source=None, alpha: float = 0.6, zorder: int = -1):
    """Intenta añadir mapas base con `contextily`.
    - `source`: provider o string del provider (ej. "CartoDB.Positron").
    - `alpha`: transparencia del mapa base.
    - `zorder`: orden de pintado.
    Si no hay red o no existe el provider, continúa sin fondo.
    """
    try:
        import contextily as cx  # type: ignore
        prov = _get_provider(cx, source) if source is not None else cx.providers.CartoDB.Positron
        cx.add_basemap(ax, crs="EPSG:3857", source=prov, alpha=alpha, attribution=False, zorder=zorder)
    except Exception as e:
        warnings.warn(
            f"No se añadió mapa base (contextily/tiles). Continuando sin fondo. Motivo: {e}"
        )


def _palette_from_labels(labels: np.ndarray) -> np.ndarray:
    # Genera colores para etiquetas; -1 en gris
    uniq = np.unique(labels)
    cmap = plt.get_cmap("tab20")
    color_map = {}
    idx = 0
    for u in uniq:
        if u == -1:
            color_map[u] = (0.6, 0.6, 0.6, 0.4)
        else:
            color_map[u] = cmap(idx % cmap.N)
            idx += 1
    return np.array([color_map[l] for l in labels])


def plot_clusters_map(
    df_xy_labels: pd.DataFrame,
    df_centroides: pd.DataFrame,
    titulo: str,
    ruta_salida: str,
    dpi: int = 200,
    basemap_source: str | None = "CartoDB.Positron",
    basemap_alpha: float = 0.5,
    overlay_source: str | None = "Stamen.TonerLines",
    overlay_alpha: float = 0.5,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Puntos
    labels = df_xy_labels["cluster"].to_numpy()
    colors = _palette_from_labels(labels)

    # Pequeños círculos para cada punto
    ax.scatter(
        df_xy_labels["x"],
        df_xy_labels["y"],
        c=colors,
        s=8,
        alpha=0.9,
        linewidths=0.0,
    )

    # Centroides
    if not df_centroides.empty:
        ax.scatter(
            df_centroides["centroid_x"],
            df_centroides["centroid_y"],
            c="black",
            s=40,
            marker="x",
            label="Centroides",
            zorder=5,
        )

    # Fondo de calles
    _try_add_basemap(ax, source=basemap_source, alpha=basemap_alpha, zorder=-2)
    # Capa de líneas para enfatizar calles (si posible)
    if overlay_source:
        _try_add_basemap(ax, source=overlay_source, alpha=overlay_alpha, zorder=-1)

    ax.set_title(titulo)
    ax.set_xlabel("x (EPSG:3857)")
    ax.set_ylabel("y (EPSG:3857)")
    ax.set_aspect("equal")
    if not df_centroides.empty:
        ax.legend(loc="best")

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=dpi)
    plt.close(fig)

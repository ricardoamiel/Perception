from __future__ import annotations

import numpy as np
import pandas as pd


def _importar_hdbscan():
    try:
        import hdbscan  # type: ignore

        return hdbscan
    except Exception as e:
        raise ImportError(
            "No se pudo importar 'hdbscan'. Instale con `pip install hdbscan`."
        ) from e


def ejecutar_hdbscan(
    df_xy: pd.DataFrame,
    min_cluster_size: int = 15,
    min_samples: int | None = None,
    metric: str = "cityblock",
) -> pd.DataFrame:
    """
    Ejecuta HDBSCAN sobre columnas 'x' y 'y'. Devuelve un DataFrame con etiquetas y probabilidades.
    """
    hdbscan = _importar_hdbscan()
    X = df_xy[["x", "y"]].to_numpy(dtype=float)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=1,
    )
    etiquetas = clusterer.fit_predict(X)
    prob = getattr(clusterer, "probabilities_", np.ones_like(etiquetas, dtype=float))

    out = df_xy.copy()
    out["cluster"] = etiquetas
    out["probability"] = prob
    return out


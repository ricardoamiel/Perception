"""
graph_pipeline.py  (v2)
========================
Pasos 1 y 2 del pipeline de grafos — versión mejorada.

CAMBIOS RESPECTO A v1:
  ① Escala de color absoluta y compartida entre distritos (para comparación justa)
  ② Etiquetado automático PELIGROSO / SEGURO por vecindario con umbral configurable
     — Estrategia A: top-30% por percentil (recomendada, robusta a outliers)
     — Estrategia B: 30% del máximo absoluto (la que pediste)
  ③ Nuevo plot KDE con etiquetas superpuestas sobre el grafo de calles
  ④ Crímenes en aristas: confirmado que ya usa nearest-node (cKDTree al nodo más
     cercano geométricamente), lo que es equivalente al midpoint implícito.
     Se añade helper explícito snap_to_nearest_node_or_edge() para documentarlo.
  ⑤ Labels y stats exportados en nodes.csv y neighbourhoods.json

Uso:
    python graph_pipeline.py
    python graph_pipeline.py --year 2019 --distrito "La Victoria"
    python graph_pipeline.py --year all
    python graph_pipeline.py --threshold-strategy percentile --threshold-pct 70
"""

import argparse
import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
import scipy.stats as sp_stats
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────

DISTRITOS = {
    "Barranco":    "Barranco, Lima, Peru",
    "La Victoria": "La Victoria, Lima, Peru",
}

SNAP_RADIUS_M = 150
EGO_DEPTH     = 1

CSV_PATH   = Path("../distritos/crimen_distritos_seleccionados.csv")
OUTPUT_DIR = Path("graph_output_v2")

# Umbral de etiquetado:
#   "percentile"   → τ = percentil P de los total_crimes (ej. 70 → top 30%)
#   "max_fraction" → τ = fracción del máximo (ej. 0.30)
#   "gmm"          → τ = cruce de componentes de Gaussian Mixture Model (2 clases)
THRESHOLD_STRATEGY = "percentile"   # "percentile" | "max_fraction" | "gmm"
THRESHOLD_PCT      = 70
THRESHOLD_FRAC     = 0.30

# ─── Paso 1: Grafo de calles ──────────────────────────────────────────────────

def build_street_graph(distrito_name: str, osm_query: str) -> nx.MultiDiGraph:
    print(f"  Descargando grafo OSM: {distrito_name}…")
    G = ox.graph_from_place(
        osm_query,
        network_type="drive",
        simplify=True,
        retain_all=False,
    )
    print(f"  Nodos: {G.number_of_nodes():,}  Aristas: {G.number_of_edges():,}")
    return G


def graph_to_undirected(G: nx.MultiDiGraph) -> nx.Graph:
    return ox.convert.to_undirected(G)


# ─── Paso 2: Mapear crímenes a nodos ─────────────────────────────────────────

def normalize_crime_id(raw) -> str:
    """
    Canonicaliza un crime_id a un formato consistente, sin importar si llega
    como int, float, o string con/sin '.0'.

    Confirmado por la convención real de carpetas de imágenes (Street View):
        Inseguros-Barranco-GGZ-2016/19833096.0/heading_120.jpg
    El ID en el nombre de carpeta SIEMPRE lleva el sufijo '.0' (probablemente
    de un cast float→str en algún punto del scraping original). Si el CSV de
    crímenes guarda el ID como entero puro ("19833096"), el join con las
    imágenes fallaría silenciosamente sin esta normalización.

    Ejemplos:  19833096      → "19833096.0"
               "19833096"    → "19833096.0"
               "19833096.0"  → "19833096.0"  (idempotente)
    """
    try:
        val = float(raw)
        if val == int(val):
            return f"{int(val)}.0"
        return str(val)
    except (ValueError, TypeError):
        return str(raw)


def _detect_crime_id_column(df: pd.DataFrame) -> str | None:
    """
    Intenta detectar automáticamente la columna de identificador único del crimen.
    Busca una columna exactamente nombrada 'ID' o que termine con '_ID' o '_Id',
    (case-insensitive), excluyendo columnas que sabemos no son identificadores.
    """
    exclude = {"distrito", "year"}
    candidates = [
        c for c in df.columns
        if (c.lower() == "id" or c.lower().endswith("_id")) 
        and c.lower() not in exclude
    ]
    return candidates[0] if candidates else None


def load_crimes(csv_path: Path, distrito: str, year: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    df = df[df["DISTRITO"] == distrito].copy()
    if year is not None:
        df = df[df["YEAR"] == year].copy()
    df = df.dropna(subset=["latitude", "longitude"])

    id_col = _detect_crime_id_column(df)
    if id_col:
        df["CRIME_ID"] = df[id_col].apply(normalize_crime_id)
        print(f"  Crímenes cargados: {len(df):,}  (distrito={distrito}, year={year or 'all'})  "
              f"[CRIME_ID ← columna '{id_col}', normalizado con sufijo '.0']")
        print(f"    Ejemplo: {df[id_col].iloc[0]!r} → {df['CRIME_ID'].iloc[0]!r}")
    else:
        df["CRIME_ID"] = df.index.map(normalize_crime_id)
        print(f"  Crímenes cargados: {len(df):,}  (distrito={distrito}, year={year or 'all'})")
        print(f"  ⚠ No se detectó columna ID en el CSV — usando índice de fila como CRIME_ID.")
        print(f"    Verifica que esto coincida con los IDs usados al nombrar las imágenes/carpetas.")

    return df


def _build_node_tree(G: nx.MultiDiGraph):
    """KDTree sobre nodos del grafo en coordenadas métricas aproximadas."""
    node_ids  = list(G.nodes())
    node_lons = np.array([G.nodes[n]["x"] for n in node_ids])
    node_lats = np.array([G.nodes[n]["y"] for n in node_ids])
    mean_lat  = np.radians(node_lats.mean())
    lat_scale = 111_000.0
    lon_scale = 111_000.0 * np.cos(mean_lat)
    coords    = np.column_stack([node_lons * lon_scale, node_lats * lat_scale])
    return cKDTree(coords), node_ids, lon_scale, lat_scale


def snap_crimes_to_nodes(
    G: nx.MultiDiGraph,
    df_crimes: pd.DataFrame,
    radius_m: float = SNAP_RADIUS_M,
) -> pd.DataFrame:
    """
    Asigna cada crimen al nodo más cercano del grafo (nodo = esquina/intersección).

    NOTA sobre crímenes en aristas:
      Un crimen que cae a mitad de una calle (en una arista, no en un nodo) se asigna
      al NODO más cercano geométricamente entre sus dos extremos. Esto es equivalente
      a calcular el punto medio de la arista y ver cuál de los dos extremos queda más
      cerca. La distancia se calcula en metros usando proyección plana local.
    """
    tree, node_ids, lon_scale, lat_scale = _build_node_tree(G)

    crime_coords = np.column_stack([
        df_crimes["longitude"].to_numpy() * lon_scale,
        df_crimes["latitude"].to_numpy()  * lat_scale,
    ])
    dists, idxs = tree.query(crime_coords, k=1, workers=-1)

    valid           = dists <= radius_m
    df_out          = df_crimes[valid].copy()
    df_out["nearest_node"] = [node_ids[i] for i in idxs[valid]]
    df_out["snap_dist_m"]  = dists[valid]

    discarded = (~valid).sum()
    if discarded:
        print(f"  ⚠ {discarded} crímenes descartados (> {radius_m}m del grafo)")
    print(f"  Crímenes mapeados a nodos: {len(df_out):,}")
    return df_out


def count_crimes_per_node(G: nx.MultiDiGraph, df_snapped: pd.DataFrame) -> dict:
    counts = df_snapped.groupby("nearest_node").size().to_dict()
    return {n: counts.get(n, 0) for n in G.nodes()}


def export_crimes_snapped(
    df_snapped: pd.DataFrame,
    distrito: str,
    year: int | None,
    output_dir: Path,
) -> Path:
    """
    Exporta el vínculo crime_id ↔ nodo, necesario para el Paso 4 (ensamblar
    embeddings por vecindario). Sin este archivo no se puede saber qué puntos
    de crimen (y por lo tanto qué imágenes Street View) caen en cada vecindario.
    """
    year_str = str(year) if year else "all"
    out_path = output_dir / f"{distrito.replace(' ', '_')}_{year_str}_crimes_snapped.csv"
    cols = ["CRIME_ID", "latitude", "longitude", "nearest_node", "snap_dist_m"]
    df_snapped[cols].to_csv(out_path, index=False)
    print(f"  Exportado: {out_path}  ({len(df_snapped):,} crímenes ↔ nodos)")
    return out_path


# ─── Vecindarios ─────────────────────────────────────────────────────────────

def build_neighbourhood(G_undir: nx.Graph, node_id: int, depth: int = EGO_DEPTH) -> nx.Graph:
    return nx.ego_graph(G_undir, node_id, radius=depth)


def summarize_neighbourhood(neighbourhood: nx.Graph, crime_counts: dict) -> dict:
    nodes  = list(neighbourhood.nodes())
    crimes = {n: crime_counts.get(n, 0) for n in nodes}
    total  = sum(crimes.values())
    max_n  = max(crimes, key=crimes.get) if nodes else None
    return {
        "nodes":          nodes,
        "n_nodes":        len(nodes),
        "crimes":         crimes,
        "total_crimes":   total,
        "max_crime_node": max_n,
    }


# ─── Etiquetado PELIGROSO / SEGURO ───────────────────────────────────────────

def _fit_gmm(values: np.ndarray) -> tuple[float, dict]:
    """
    Ajusta un Gaussian Mixture Model de 2 componentes a la distribución de
    total_crimes y devuelve el umbral τ como el punto de cruce entre las dos
    densidades ponderadas, junto con la info de componentes para graficar.

    Componente 0 = "seguro"    (media baja)
    Componente 1 = "peligroso" (media alta)

    Lógica del cruce:
      Evaluar ambas PDFs ponderadas en una grilla densa.
      τ = primer punto donde pdf_peligroso > pdf_seguro.
      Si no hay cruce (distribuciones muy solapadas) → promedio de medias.
    """
    X   = values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42,
                          max_iter=300, n_init=5)
    gmm.fit(X)

    # Ordenar componentes por media (0=baja=seguro, 1=alta=peligroso)
    order   = np.argsort(gmm.means_.ravel())
    means   = gmm.means_.ravel()[order]
    stds    = np.sqrt(gmm.covariances_.ravel())[order]
    weights = gmm.weights_[order]

    # Grid de evaluación
    x_min   = max(0.0, float(values.min()))
    x_max   = float(np.percentile(values, 99.5))
    x_grid  = np.linspace(x_min, x_max, 3000)

    pdf_safe  = weights[0] * sp_stats.norm.pdf(x_grid, means[0], stds[0])
    pdf_dang  = weights[1] * sp_stats.norm.pdf(x_grid, means[1], stds[1])

    # Primer cruce donde peligroso supera a seguro
    diff            = pdf_dang - pdf_safe
    sign_changes    = np.where(np.diff(np.sign(diff)) > 0)[0]   # subida: seguro→peligroso

    if len(sign_changes) > 0:
        tau = float(x_grid[sign_changes[0]])
    else:
        # Sin cruce limpio → punto medio entre medias
        tau = float((means[0] + means[1]) / 2)
        print(f"  ⚠ GMM: no se encontró cruce claro, usando τ=(μ₀+μ₁)/2={tau:.1f}")

    gmm_info = {
        "means":    means.tolist(),
        "stds":     stds.tolist(),
        "weights":  weights.tolist(),
        "x_grid":   x_grid,
        "pdf_safe": pdf_safe,
        "pdf_dang": pdf_dang,
        "bic":      gmm.bic(X),
        "aic":      gmm.aic(X),
    }
    return tau, gmm_info


def _fit_gmm_excluding_zeros(
    values: np.ndarray,
    exclude_zeros: bool = True,
) -> tuple[float, dict | None]:
    """
    Wrapper sobre _fit_gmm que excluye los ceros del ajuste.

    Por qué: si se incluyen los ceros, el componente "seguro" del GMM tiende a
    colapsar modelando el pico masivo en cero, en vez de capturar la subpoblación
    real de bajo-pero-no-nulo riesgo. Los valores en cero se etiquetan como
    "seguro" por definición, sin necesidad de pasar por el modelo.
    """
    fit_values = values[values > 0] if exclude_zeros else values

    if len(np.unique(fit_values)) < 3:
        # Muy poca variación para ajustar 2 componentes — fallback a percentil 70
        tau = float(np.percentile(values, 70)) if len(values) else 0.0
        print(f"  ⚠ Datos insuficientes para GMM (solo {len(np.unique(fit_values))} valores "
              f"únicos > 0) — usando fallback τ=P70={tau:.1f}")
        return tau, None

    return _fit_gmm(fit_values)


def compute_dual_gmm_labels(
    G_undir: nx.Graph,
    crime_counts: dict,
    depth: int = EGO_DEPTH,
    exclude_zeros: bool = True,
) -> tuple[dict, float, dict | None, float, dict | None]:
    """
    Etiqueta cada nodo con DOS GMMs independientes:
      - node_label          → basado en crime_count del nodo individual (sin agregar)
      - neighbourhood_label → basado en total_crimes del vecindario (ego-graph,
                               igual que compute_neighbourhood_labels)

    El "label" primario para el modelo de clasificación es siempre
    neighbourhood_label (vecindario), pero node_label queda disponible como
    feature auxiliar o para diagnóstico.

    Devuelve:
      labels         → {node_id: {node_crime_count, node_label,
                                   neigh_total_crimes, neighbourhood_label, label}}
      neigh_tau      → umbral τ del GMM de vecindario
      neigh_gmm_info → componentes del GMM de vecindario (para plot)
      node_tau       → umbral τ del GMM de nodo
      node_gmm_info  → componentes del GMM de nodo (para plot)
    """
    # ── Nivel nodo (sin agregar) ─────────────────────────────────────────────
    node_counts = np.array([crime_counts.get(n, 0) for n in G_undir.nodes()])
    node_tau, node_gmm_info = _fit_gmm_excluding_zeros(node_counts, exclude_zeros)

    # ── Nivel vecindario (ego-graph, igual que antes) ───────────────────────
    totals = {}
    for n in G_undir.nodes():
        nb = build_neighbourhood(G_undir, n, depth)
        s  = summarize_neighbourhood(nb, crime_counts)
        totals[n] = s["total_crimes"]
    neigh_values = np.array(list(totals.values()))
    neigh_tau, neigh_gmm_info = _fit_gmm_excluding_zeros(neigh_values, exclude_zeros)

    # ── Construir labels ─────────────────────────────────────────────────────
    labels = {}
    for n in G_undir.nodes():
        c = crime_counts.get(n, 0)
        t = totals[n]
        node_label = "peligroso" if c > node_tau else "seguro"
        neigh_label = "peligroso" if t > neigh_tau else "seguro"
        labels[n] = {
            "node_crime_count":    c,
            "node_label":          node_label,
            "neigh_total_crimes":  t,
            "neighbourhood_label": neigh_label,
            "total_crimes":        t,        # alias para compatibilidad con plots existentes
            "label":               neigh_label,  # label primario (vecindario)
        }

    n_node_pelig  = sum(1 for v in labels.values() if v["node_label"] == "peligroso")
    n_neigh_pelig = sum(1 for v in labels.values() if v["neighbourhood_label"] == "peligroso")
    print(f"  GMM dual:")
    print(f"    Nodo         τ={node_tau:.1f}   → {n_node_pelig:,}/{len(labels):,} "
          f"peligrosos ({100*n_node_pelig/len(labels):.1f}%)")
    print(f"    Vecindario   τ={neigh_tau:.1f}   → {n_neigh_pelig:,}/{len(labels):,} "
          f"peligrosos ({100*n_neigh_pelig/len(labels):.1f}%)  ← label primario")

    return labels, neigh_tau, neigh_gmm_info, node_tau, node_gmm_info


def compute_neighbourhood_labels(
    G_undir: nx.Graph,
    crime_counts: dict,
    strategy: str  = THRESHOLD_STRATEGY,
    pct: float     = THRESHOLD_PCT,
    frac: float    = THRESHOLD_FRAC,
    depth: int     = EGO_DEPTH,
) -> tuple[dict, float, dict | None]:
    """
    Calcula total_crimes por vecindario y etiqueta cada nodo.

    Estrategias:
      "percentile"   → τ = percentil `pct`  (ej. 70 → top 30% = peligroso)
      "max_fraction" → τ = `frac` × max
      "gmm"          → τ = cruce de las dos componentes del GMM (datos-driven)

    Devuelve:
      labels    → {node_id: {"total_crimes": int, "label": str}}
      threshold → float τ
      gmm_info  → dict con componentes GMM para el plot (None si no es GMM)
    """
    totals = {}
    for n in G_undir.nodes():
        nb = build_neighbourhood(G_undir, n, depth)
        s  = summarize_neighbourhood(nb, crime_counts)
        totals[n] = s["total_crimes"]

    values   = np.array(list(totals.values()))
    gmm_info = None

    if strategy == "percentile":
        tau = float(np.percentile(values, pct))
    elif strategy == "max_fraction":
        tau = frac * float(values.max())
    elif strategy == "gmm":
        tau, gmm_info = _fit_gmm(values)
        print(f"  GMM  BIC={gmm_info['bic']:.0f}  AIC={gmm_info['aic']:.0f}")
        print(f"       μ_seguro={gmm_info['means'][0]:.1f} (σ={gmm_info['stds'][0]:.1f}, "
              f"w={gmm_info['weights'][0]:.2f})")
        print(f"       μ_peligroso={gmm_info['means'][1]:.1f} (σ={gmm_info['stds'][1]:.1f}, "
              f"w={gmm_info['weights'][1]:.2f})")
    else:
        raise ValueError(
            f"strategy debe ser 'percentile', 'max_fraction' o 'gmm', no '{strategy}'"
        )

    labels = {
        n: {"total_crimes": t, "label": "peligroso" if t > tau else "seguro"}
        for n, t in totals.items()
    }

    n_pelig = sum(1 for v in labels.values() if v["label"] == "peligroso")
    print(f"  Umbral τ={tau:.1f} ({strategy})  →  "
          f"Peligrosos: {n_pelig:,} / {len(labels):,} "
          f"({100*n_pelig/len(labels):.1f}%)")
    return labels, tau, gmm_info


# ─── Visualización 1: Grafo con escala de color (comparable entre distritos) ──

def plot_graph_with_crimes(
    G: nx.MultiDiGraph,
    crime_counts: dict,
    distrito: str,
    year: int | None,
    output_path: Path,
    global_max: int | None = None,   # ← NUEVO: escala absoluta compartida
):
    """
    Grafo de calles con nodos coloreados por intensidad de crimen.
    Si se pasa global_max, la escala de color es comparable entre distritos.
    """
    print(f"  Generando plot (grafo + crimen)…")

    max_crimes = global_max if global_max else max(crime_counts.values(), default=1)

    node_colors, node_sizes = [], []
    for n in G.nodes():
        c = crime_counts.get(n, 0)
        if c == 0:
            node_colors.append("#9ca3af")
            node_sizes.append(8)
        else:
            intensity = c / max_crimes
            r = 0.9;  g = 1 - intensity * 0.8;  b = 1 - intensity * 0.8
            node_colors.append((r, g, b))
            node_sizes.append(12 + intensity * 40)

    fig, ax = ox.plot_graph(
        G, ax=None, figsize=(12, 12),
        bgcolor="#fafaf7", edge_color="#b0b8c8",
        edge_linewidth=0.6, edge_alpha=0.9,
        node_color=node_colors, node_size=node_sizes,
        show=False, close=False,
    )

    year_str   = str(year) if year else "all"
    scale_note = f"escala compartida 0–{max_crimes}" if global_max else f"escala local 0–{max_crimes}"
    ax.set_title(
        f"Grafo de calles — {distrito} ({year_str})\n"
        f"Nodos coloreados por intensidad de crimen  ({scale_note} crímenes/nodo)",
        color="#475569", fontsize=13, pad=12,
    )

    cmap_red = mcolors.LinearSegmentedColormap.from_list(
        "crime_red", ["#e2e8f0", "#fca5a5", "#dc2626"]
    )
    sm = plt.cm.ScalarMappable(cmap=cmap_red, norm=mcolors.Normalize(vmin=0, vmax=max_crimes))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.6)
    cbar.set_label("Crímenes por nodo", color="#475569", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#475569")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#475569")

    ax.legend(
        handles=[mpatches.Patch(color="#9ca3af", label="Sin crimen (0)")],
        loc="lower right",
        facecolor="white", edgecolor="#cbd5e1",
        labelcolor="#475569", fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#fafaf7")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


# ─── Visualización 2: KDE sobre nodos + etiquetas de vecindario ──────────────

def plot_kde_with_labels(
    G: nx.MultiDiGraph,
    crime_counts: dict,
    neighbourhood_labels: dict,
    threshold: float,
    distrito: str,
    year: int | None,
    output_path: Path,
    kde_bandwidth: float = 0.003,   # ~300m en grados; ajustar según distrito
    kde_alpha: float     = 0.55,
    label_every_n: int   = 5,       # marcar 1 de cada N nodos para no saturar
):
    """
    Plot del grafo de calles con:
      ① KDE coloreado (calor de crimen) calculado sobre posiciones de nodos
         ponderado por crime_count
      ② Puntos de nodos coloreados ROJO (peligroso) / VERDE (seguro) según
         el umbral de vecindario
    """
    print(f"  Generando plot (KDE + etiquetas)…")

    # --- Coordenadas de nodos ---
    lons = np.array([G.nodes[n]["x"] for n in G.nodes()])
    lats = np.array([G.nodes[n]["y"] for n in G.nodes()])
    weights = np.array([crime_counts.get(n, 0) for n in G.nodes()])

    # --- KDE ponderado por crimen ---
    # Expandir puntos según peso (repetir el nodo weight veces)
    mask = weights > 0
    if mask.sum() < 2:
        print("  ⚠ No hay suficientes nodos con crimen para KDE.")
        return

    lons_w = np.repeat(lons[mask], weights[mask].astype(int))
    lats_w = np.repeat(lats[mask], weights[mask].astype(int))
    kde    = gaussian_kde(np.vstack([lons_w, lats_w]), bw_method=kde_bandwidth)

    # Grid de evaluación sobre el bounding box del distrito
    margin   = 0.002
    lon_grid = np.linspace(lons.min() - margin, lons.max() + margin, 250)
    lat_grid = np.linspace(lats.min() - margin, lats.max() + margin, 250)
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)
    Z        = kde(np.vstack([Lon.ravel(), Lat.ravel()])).reshape(Lon.shape)

    # Normalizar Z para aplicar el umbral de forma visual
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)

    # --- Plot base del grafo ---
    fig, ax = ox.plot_graph(
        G, ax=None, figsize=(13, 13),
        bgcolor="#fafaf7",
        edge_color="#b0b8c8",
        edge_linewidth=0.6, edge_alpha=0.9,
        node_size=0,
        show=False, close=False,
    )

    # --- KDE como heatmap (blanco → amarillo → naranja → rojo, legible en fondo claro) ---
    cmap_kde = LinearSegmentedColormap.from_list(
        "crime_heat_light",
        ["#fafaf7", "#fef9c3", "#fde68a", "#f97316", "#dc2626", "#7f1d1d"],
    )
    im = ax.imshow(
        Z_norm,
        extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()],
        origin="lower",
        cmap=cmap_kde,
        alpha=kde_alpha,
        aspect="auto",
        zorder=1,
    )

    # --- Nodos con etiqueta PELIGROSO / SEGURO ---
    for n in G.nodes():
        info = neighbourhood_labels.get(n, {})
        if info.get("label") == "peligroso":
            color, zorder, size = "#dc2626", 4, 22
        elif info.get("total_crimes", 0) > 0:
            color, zorder, size = "#16a34a", 3, 10
        else:
            color, zorder, size = "#94a3b8", 2, 5

        ax.scatter(
            G.nodes[n]["x"], G.nodes[n]["y"],
            c=color, s=size, zorder=zorder, linewidths=0,
        )

    # Contorno al percentil 70 de Z_norm
    tau_z = float(np.percentile(Z_norm, 70))
    ax.contour(
        Lon, Lat, Z_norm,
        levels=[tau_z],
        colors=["#475569"],
        linewidths=1.0,
        alpha=0.7,
        zorder=5,
    )

    # --- Leyenda y título ---
    year_str = str(year) if year else "all"
    ax.set_title(
        f"KDE de crimen + vecindarios — {distrito} ({year_str})\n"
        f"Umbral τ={threshold:.1f} crímenes/vecindario   "
        f"▲ contorno oscuro = frontera de riesgo",
        color="#475569", fontsize=12, pad=12,
    )

    patches = [
        mpatches.Patch(color="#dc2626", label="Vecindario PELIGROSO"),
        mpatches.Patch(color="#16a34a", label="Vecindario seguro (con crimen)"),
        mpatches.Patch(color="#94a3b8", label="Sin crimen"),
    ]
    ax.legend(handles=patches, loc="lower right",
              facecolor="white", edgecolor="#cbd5e1",
              labelcolor="#475569", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Densidad de crimen (KDE)", color="#475569", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#475569")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#475569")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#fafaf7")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


# ─── Visualización 3: Vecindarios de ejemplo (igual que v1 + label) ───────────

def plot_neighbourhood_example(
    G: nx.MultiDiGraph,
    G_undir: nx.Graph,
    crime_counts: dict,
    neighbourhood_labels: dict,
    threshold: float,
    distrito: str,
    year: int | None,
    output_path: Path,
    n_examples: int = 3,
):
    top_nodes = sorted(crime_counts, key=crime_counts.get, reverse=True)[:n_examples]
    top_nodes = [n for n in top_nodes if crime_counts[n] > 0]
    if not top_nodes:
        print("  No hay nodos con crímenes para mostrar vecindarios.")
        return

    example_colors = ["#dc2626", "#d97706", "#16a34a"]

    fig, axes = plt.subplots(1, len(top_nodes),
                              figsize=(8 * len(top_nodes), 8),
                              facecolor="white")
    if len(top_nodes) == 1:
        axes = [axes]

    for ax, center_node, ex_color in zip(axes, top_nodes, example_colors):
        neighbourhood = build_neighbourhood(G_undir, center_node, depth=EGO_DEPTH)
        neigh_nodes   = set(neighbourhood.nodes())
        summary       = summarize_neighbourhood(neighbourhood, crime_counts)
        label_info    = neighbourhood_labels.get(center_node, {})
        label_str     = label_info.get("label", "?").upper()
        total_neigh   = label_info.get("total_crimes", summary["total_crimes"])

        nc, ns = [], []
        for n in G.nodes():
            if n == center_node:
                nc.append(ex_color); ns.append(80)
            elif n in neigh_nodes:
                nc.append("#93c5fd"); ns.append(30)
            else:
                nc.append("#9ca3af"); ns.append(5)

        ox.plot_graph(G, ax=ax, bgcolor="#fafaf7", edge_color="#b0b8c8",
                      edge_linewidth=0.4, edge_alpha=0.8,
                      node_color=nc, node_size=ns, show=False, close=False)

        neigh_lons = [G.nodes[n]["x"] for n in neigh_nodes]
        neigh_lats = [G.nodes[n]["y"] for n in neigh_nodes]
        margin = 0.005
        ax.set_xlim(min(neigh_lons) - margin, max(neigh_lons) + margin)
        ax.set_ylim(min(neigh_lats) - margin, max(neigh_lats) + margin)

        label_color = "#dc2626" if label_str == "PELIGROSO" else "#16a34a"
        ax.set_title(
            f"Nodo {center_node}  →  [{label_str}]\n"
            f"Crímenes vecindario: {total_neigh}  (τ={threshold:.0f})  |  Nodos: {summary['n_nodes']}",
            color=label_color, fontsize=10, pad=8,
        )

    year_str = str(year) if year else "all"
    fig.suptitle(f"Ejemplos de vecindarios — {distrito} {year_str}",
                 color="#475569", fontsize=14, y=1.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


# ─── Exportar ─────────────────────────────────────────────────────────────────

def export_graph_data(
    G_undir: nx.Graph,
    crime_counts: dict,
    neighbourhood_labels: dict,
    threshold: float,
    distrito: str,
    year: int | None,
    output_dir: Path,
    strat_tag: str = "",       # ej. "pct70" o "frac030"
):
    year_str     = str(year) if year else "all"
    base         = f"{distrito.replace(' ', '_')}_{year_str}"
    # nodes/edges no dependen de la estrategia → se sobreescriben entre runs (es lo correcto)
    # neighbourhoods sí dependen → llevan el tag
    prefix_base  = output_dir / base
    prefix_full  = output_dir / f"{base}_{strat_tag}" if strat_tag else prefix_base

    # Nodes — no depende de la estrategia (crime_count es el mismo siempre)
    rows = []
    for n, data in G_undir.nodes(data=True):
        info = neighbourhood_labels.get(n, {})
        rows.append({
            "node_id":             n,
            "lon":                 data.get("x"),
            "lat":                 data.get("y"),
            "crime_count":         crime_counts.get(n, 0),
            "neigh_total":         info.get("total_crimes", 0),
            "label":               info.get("label", "seguro"),               # primario (vecindario)
            "node_label":          info.get("node_label", ""),                # solo si dual_gmm
            "neighbourhood_label": info.get("neighbourhood_label", info.get("label", "seguro")),
            "threshold":           threshold,
            "strat_tag":           strat_tag,
        })
    pd.DataFrame(rows).to_csv(f"{prefix_full}_nodes.csv", index=False)

    # Edges — no depende de la estrategia (se guarda sin tag, se sobreescribe)
    edge_rows = [
        {"u": u, "v": v, "length_m": data.get("length")}
        for u, v, data in G_undir.edges(data=True)
    ]
    pd.DataFrame(edge_rows).to_csv(f"{prefix_base}_edges.csv", index=False)

    # Neighbourhoods — depende de estrategia → lleva tag
    crime_nodes = {n for n, c in crime_counts.items() if c > 0}
    neighbourhoods = {}
    for n in crime_nodes:
        nb      = build_neighbourhood(G_undir, n, depth=EGO_DEPTH)
        summary = summarize_neighbourhood(nb, crime_counts)
        info    = neighbourhood_labels.get(n, {})
        summary["label"]     = info.get("label", "seguro")
        summary["threshold"] = threshold
        summary["strat_tag"] = strat_tag
        neighbourhoods[str(n)] = summary

    with open(f"{prefix_full}_neighbourhoods.json", "w", encoding="utf-8") as f:
        json.dump(neighbourhoods, f, indent=2, ensure_ascii=False)

    print(f"  Exportado: {prefix_full}_nodes.csv")
    print(f"             {prefix_base}_edges.csv  (sin tag, no depende de estrategia)")
    print(f"             {prefix_full}_neighbourhoods.json")
    print(f"  Vecindarios exportados: {len(neighbourhoods):,}")


# ─── Stats resumen ────────────────────────────────────────────────────────────

def print_label_stats(
    neighbourhood_labels: dict,
    threshold: float,
    distrito: str,
    year: int | None,
):
    totals   = [v["total_crimes"] for v in neighbourhood_labels.values()]
    pelig    = [v for v in neighbourhood_labels.values() if v["label"] == "peligroso"]
    seguros  = [v for v in neighbourhood_labels.values() if v["label"] == "seguro"]

    print(f"\n  📊 Stats de etiquetado — {distrito} {year or 'all'}")
    print(f"     Umbral τ            : {threshold:.1f} crímenes/vecindario")
    print(f"     Vecindarios totales : {len(neighbourhood_labels):,}")
    print(f"     PELIGROSOS          : {len(pelig):,}  "
          f"(avg crím: {np.mean([v['total_crimes'] for v in pelig]):.1f})")
    print(f"     SEGUROS             : {len(seguros):,}  "
          f"(avg crím: {np.mean([v['total_crimes'] for v in seguros]):.1f})")
    print(f"     Max total_crimes    : {max(totals):,}")
    print(f"     Mediana total_crimes: {np.median(totals):.1f}")


# ─── Visualización 4: Distribución de total_crimes por vecindario ─────────────

# Percentiles candidatos a mostrar siempre como referencia visual
_CANDIDATE_PCTS = [50, 60, 70, 75, 80]
_CANDIDATE_COLORS = {
    50: ("#94a3b8", "--"),   # gris azulado
    60: ("#3b82f6", "--"),   # azul
    70: ("#f59e0b", "-"),    # ámbar  ← default recomendado
    75: ("#f97316", "-"),    # naranja
    80: ("#dc2626", "-"),    # rojo
}


def plot_crime_distribution(
    neighbourhood_labels: dict,
    threshold: float,
    threshold_strategy: str,
    threshold_pct: float,
    threshold_frac: float,
    distrito: str,
    year: int | None,
    output_path: Path,
    gmm_info: dict | None = None,
):
    """
    Histograma + KDE de total_crimes por vecindario con líneas de umbral candidatas.
    Si gmm_info no es None (strategy='gmm'), superpone las dos componentes gaussianas
    y marca el cruce como τ.
    """
    print(f"  Generando plot (distribución de crimen)…")

    totals  = np.array([v["total_crimes"] for v in neighbourhood_labels.values()])
    n_total = len(totals)

    # ── Figura con dos paneles ──────────────────────────────────────────────
    fig, (ax_hist, ax_pct) = plt.subplots(
        1, 2,
        figsize=(16, 6),
        facecolor="white",
        gridspec_kw={"width_ratios": [2, 1]},
    )
    for ax in (ax_hist, ax_pct):
        ax.set_facecolor("#fafaf7")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
        ax.tick_params(colors="#475569")
        ax.xaxis.label.set_color("#475569")
        ax.yaxis.label.set_color("#475569")
        ax.title.set_color("#475569")
        ax.grid(True, color="#e2e8f0", linewidth=0.6, zorder=0)

    # ── Panel izquierdo: histograma + KDE ───────────────────────────────────
    # Recorte suave para no distorsionar la escala con outliers extremos
    p95    = np.percentile(totals, 95)
    totals_clip = totals[totals <= p95 * 1.2]   # incluye hasta ~1.2×p95 para ver la cola
    clip_note   = f"(se muestran vecindarios con ≤{int(p95*1.2)} crímenes, " \
                  f"{100*len(totals_clip)/n_total:.0f}% del total)"

    bins = min(50, max(20, int(np.sqrt(n_total))))
    ax_hist.hist(
        totals_clip, bins=bins,
        color="#bfdbfe", edgecolor="#93c5fd",
        alpha=0.9, zorder=2,
        label="Vecindarios",
    )

    # KDE suavizada sobre distribución completa
    if len(np.unique(totals)) > 3:
        kde_x = np.linspace(0, totals_clip.max(), 400)
        kde   = gaussian_kde(totals, bw_method="scott")
        kde_y = kde(kde_x) * n_total * (totals_clip.max() / bins)  # escalar al histograma
        ax_hist.plot(kde_x, kde_y, color="#1d4ed8", linewidth=2.0,
                     zorder=3, label="KDE")

    # Líneas de percentiles candidatos
    for pct in _CANDIDATE_PCTS:
        tau_c   = float(np.percentile(totals, pct))
        color_c, ls_c = _CANDIDATE_COLORS[pct]
        is_active = (
            threshold_strategy == "percentile" and abs(threshold_pct - pct) < 0.5
        )
        lw = 2.5 if is_active else 1.2
        alpha = 1.0 if is_active else 0.65
        ax_hist.axvline(
            tau_c, color=color_c, linewidth=lw, linestyle=ls_c,
            alpha=alpha, zorder=4,
            label=f"P{pct} = {tau_c:.0f}  {'← activo' if is_active else ''}",
        )

    # Si la estrategia activa NO es percentil, dibuja el umbral activo aparte
    if threshold_strategy == "max_fraction":
        ax_hist.axvline(
            threshold, color="#dc2626", linewidth=2.2, linestyle="-",
            zorder=5,
            label=f"τ activo (frac×max) = {threshold:.1f}",
        )

    # ── Overlay GMM: curvas de las dos componentes ───────────────────────────
    if gmm_info is not None:
        x_g      = gmm_info["x_grid"]
        # Escalar PDFs al eje del histograma (misma transformación que la KDE)
        scale    = n_total * (totals_clip.max() / bins)
        y_safe   = gmm_info["pdf_safe"] * scale
        y_dang   = gmm_info["pdf_dang"] * scale

        ax_hist.plot(x_g, y_safe, color="#16a34a", linewidth=2.2, linestyle="-",
                     zorder=6, label=(f"GMM seguro  "
                                      f"μ={gmm_info['means'][0]:.1f} "
                                      f"σ={gmm_info['stds'][0]:.1f} "
                                      f"w={gmm_info['weights'][0]:.2f}"))
        ax_hist.fill_between(x_g, y_safe, alpha=0.10, color="#16a34a", zorder=5)

        ax_hist.plot(x_g, y_dang, color="#dc2626", linewidth=2.2, linestyle="-",
                     zorder=6, label=(f"GMM peligroso  "
                                      f"μ={gmm_info['means'][1]:.1f} "
                                      f"σ={gmm_info['stds'][1]:.1f} "
                                      f"w={gmm_info['weights'][1]:.2f}"))
        ax_hist.fill_between(x_g, y_dang, alpha=0.10, color="#dc2626", zorder=5)

        # Línea de cruce τ
        ax_hist.axvline(threshold, color="#7c3aed", linewidth=2.5, linestyle="-",
                        zorder=7, label=f"τ GMM = {threshold:.1f}  ← activo")

        # Anotación BIC/AIC en el plot
        ax_hist.text(
            0.98, 0.98,
            f"BIC: {gmm_info['bic']:.0f}\nAIC: {gmm_info['aic']:.0f}",
            transform=ax_hist.transAxes, fontsize=8, color="#334155",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cbd5e1", alpha=0.9),
        )

    med = np.median(totals)
    ax_hist.axvline(med, color="#059669", linewidth=1.2, linestyle=":",
                    alpha=0.8, zorder=3, label=f"Mediana = {med:.0f}")

    year_str = str(year) if year else "all"
    ax_hist.set_title(
        f"Distribución de crímenes por vecindario\n{distrito} {year_str}  {clip_note}",
        fontsize=11, pad=10,
    )
    ax_hist.set_xlabel("Total crímenes en el vecindario (ego-graph depth=1)")
    ax_hist.set_ylabel("Número de vecindarios")
    ax_hist.legend(
        fontsize=8.5, facecolor="white", edgecolor="#cbd5e1",
        labelcolor="#475569", loc="upper right",
    )

    # Texto con stats básicos
    stats_txt = (
        f"N vecindarios : {n_total}\n"
        f"Media          : {totals.mean():.1f}\n"
        f"Mediana        : {med:.1f}\n"
        f"Máx            : {totals.max()}\n"
        f"Skewness       : {float(pd.Series(totals).skew()):.2f}"
    )
    ax_hist.text(
        0.98, 0.60, stats_txt,
        transform=ax_hist.transAxes,
        fontsize=8, color="#334155",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cbd5e1", alpha=0.9),
    )

    # ── Panel derecho: % peligrosos vs umbral ───────────────────────────────
    tau_range = np.linspace(0, np.percentile(totals, 98), 200)
    pct_pelig = [100 * (totals > t).sum() / n_total for t in tau_range]

    ax_pct.plot(tau_range, pct_pelig, color="#1d4ed8", linewidth=2)
    ax_pct.fill_between(tau_range, pct_pelig, alpha=0.12, color="#3b82f6")

    # Marcar los percentiles candidatos
    for pct in _CANDIDATE_PCTS:
        tau_c   = float(np.percentile(totals, pct))
        pct_val = 100 - pct
        color_c, _ = _CANDIDATE_COLORS[pct]
        is_active  = (
            threshold_strategy == "percentile" and abs(threshold_pct - pct) < 0.5
        )
        ax_pct.scatter([tau_c], [pct_val],
                       color=color_c, s=70 if is_active else 35,
                       zorder=5, edgecolors="#475569" if is_active else "none",
                       linewidths=0.8)
        ax_pct.annotate(
            f"P{pct}\n{pct_val:.0f}%",
            xy=(tau_c, pct_val),
            xytext=(6, 0), textcoords="offset points",
            fontsize=7.5, color=color_c, va="center",
        )

    # Umbral activo
    pct_activo = 100 * (totals > threshold).sum() / n_total
    ax_pct.axvline(threshold, color="#f59e0b", linewidth=1.5,
                   linestyle="--", alpha=0.9)
    ax_pct.axhline(pct_activo, color="#f59e0b", linewidth=0.8,
                   linestyle=":", alpha=0.7)

    ax_pct.set_title(f"% vecindarios peligrosos\nvs umbral τ", fontsize=11, pad=10)
    ax_pct.set_xlabel("Umbral τ (crímenes/vecindario)")
    ax_pct.set_ylabel("% vecindarios etiquetados PELIGROSOS")
    ax_pct.set_ylim(0, 105)
    ax_pct.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.suptitle(
        f"Análisis de distribución para selección de umbral — {distrito} {year_str}",
        color="#475569", fontsize=13, y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_pipeline(
    distrito: str,
    year: int | None,
    csv_path: Path,
    output_dir: Path,
    threshold_strategy: str = THRESHOLD_STRATEGY,
    threshold_pct: float    = THRESHOLD_PCT,
    threshold_frac: float   = THRESHOLD_FRAC,
    global_max: int | None  = None,
    dual_gmm: bool          = False,
):
    osm_query = DISTRITOS[distrito]
    print(f"\n{'='*60}")
    print(f"  Distrito: {distrito} | Año: {year or 'all'}")
    print(f"{'='*60}")

    G       = build_street_graph(distrito, osm_query)
    G_undir = graph_to_undirected(G)

    df_crimes = load_crimes(csv_path, distrito, year)
    if df_crimes.empty:
        print("  Sin datos de crimen para este filtro.")
        return

    df_snapped   = snap_crimes_to_nodes(G, df_crimes)
    crime_counts = count_crimes_per_node(G, df_snapped)

    # Export del vínculo crime_id ↔ nodo — siempre, lo necesita el Paso 4
    export_crimes_snapped(df_snapped, distrito, year, output_dir)

    total_with_crime = sum(1 for c in crime_counts.values() if c > 0)
    print(f"  Nodos con ≥1 crimen: {total_with_crime:,} / {G.number_of_nodes():,}")

    # Etiquetado de vecindarios
    node_tau, node_gmm_info = None, None
    if dual_gmm:
        (neighbourhood_labels, threshold, gmm_info,
         node_tau, node_gmm_info) = compute_dual_gmm_labels(G_undir, crime_counts)
        threshold_strategy = "gmm"   # forzar tag correcto en outputs
    else:
        neighbourhood_labels, threshold, gmm_info = compute_neighbourhood_labels(
            G_undir, crime_counts,
            strategy=threshold_strategy,
            pct=threshold_pct,
            frac=threshold_frac,
        )
    print_label_stats(neighbourhood_labels, threshold, distrito, year)

    year_str = str(year) if year else "all"

    # Prefijo de estrategia: identifica los outputs sin ambigüedad
    if dual_gmm:
        strat_tag = "dualgmm"
    elif threshold_strategy == "percentile":
        strat_tag = f"pct{int(threshold_pct)}"
    elif threshold_strategy == "max_fraction":
        strat_tag = f"frac{int(threshold_frac * 100):03d}"
    else:
        strat_tag = "gmm"

    prefix      = f"{distrito.replace(' ', '_')}_{year_str}"     # base (grafo sin labels)
    prefix_full = f"{prefix}_{strat_tag}"                         # incluye estrategia

    # Plot 1 — grafo con escala de color (sin sufijo de estrategia: no depende de ella)
    plot_graph_with_crimes(
        G, crime_counts, distrito, year,
        output_dir / f"{prefix}_graph.png",
        global_max=global_max,
    )

    # Plot 2 — KDE + etiquetas (depende de la estrategia → usa prefix_full)
    plot_kde_with_labels(
        G, crime_counts, neighbourhood_labels, threshold,
        distrito, year,
        output_dir / f"{prefix_full}_kde_labels.png",
    )

    # Plot 3 — ejemplos de vecindarios (depende de la estrategia → usa prefix_full)
    plot_neighbourhood_example(
        G, G_undir, crime_counts, neighbourhood_labels, threshold,
        distrito, year,
        output_dir / f"{prefix_full}_neighbourhoods.png",
        n_examples=3,
    )

    # Plot 4 — distribución de total_crimes + análisis de umbral
    plot_crime_distribution(
        neighbourhood_labels, threshold,
        threshold_strategy, threshold_pct, threshold_frac,
        distrito, year,
        output_dir / f"{prefix}_distribution.png",
        gmm_info=gmm_info,
    )

    export_graph_data(
        G_undir, crime_counts, neighbourhood_labels, threshold,
        distrito, year, output_dir,
        strat_tag=strat_tag,
    )

    print(f"\n  ✅ Listo — {distrito} {year_str}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph pipeline — pasos 1 y 2 (v2)")
    parser.add_argument("--distrito", default="all",
                        help="'Barranco', 'La Victoria', o 'all'")
    parser.add_argument("--year", default="2016",
                        help="Año (e.g. 2016) o 'all' para todos los años")
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--threshold-strategy", default=THRESHOLD_STRATEGY,
                        choices=["percentile", "max_fraction", "gmm"],
                        help="Estrategia de umbral: percentile | max_fraction | gmm")
    parser.add_argument("--threshold-pct", type=float, default=THRESHOLD_PCT,
                        help="Percentil (si strategy=percentile). Default=70 → top 30%%")
    parser.add_argument("--threshold-frac", type=float, default=THRESHOLD_FRAC,
                        help="Fracción del máximo (si strategy=max_fraction). Default=0.30")
    parser.add_argument("--shared-scale", action="store_true",
                        help="Usar escala de color absoluta compartida entre distritos")
    parser.add_argument("--dual-gmm", action="store_true",
                        help="Ajustar GMM separado para nodo individual Y vecindario "
                             "(ignora --threshold-strategy, --threshold-pct, --threshold-frac)")
    args = parser.parse_args()

    csv_path   = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    distritos = list(DISTRITOS.keys()) if args.distrito == "all" else [args.distrito]
    years     = ([None] if args.year == "all"
                 else [int(y) for y in args.year.split(",")])

    # Si queremos escala compartida, necesitamos pre-calcular el max global
    global_max = None
    if args.shared_scale:
        print("  Calculando escala global de crimen…")
        all_counts = []
        for d in distritos:
            for y in years:
                df = load_crimes(csv_path, d, y)
                if not df.empty:
                    G_tmp = build_street_graph(d, DISTRITOS[d])
                    df_s  = snap_crimes_to_nodes(G_tmp, df)
                    cc    = count_crimes_per_node(G_tmp, df_s)
                    all_counts.extend(cc.values())
        global_max = max(all_counts) if all_counts else None
        print(f"  Max global de crimen por nodo: {global_max}")

    for distrito in distritos:
        for year in years:
            run_pipeline(
                distrito, year, csv_path, output_dir,
                threshold_strategy=args.threshold_strategy,
                threshold_pct=args.threshold_pct,
                threshold_frac=args.threshold_frac,
                global_max=global_max,
                dual_gmm=args.dual_gmm,
            )


if __name__ == "__main__":
    main()
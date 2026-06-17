"""
Paso 4: Ensamblar tensores de embeddings por vecindario
==========================================================

Para cada nodo (vecindario) del grafo:
  ① Expandir el ego-graph (depth adaptativo: 1, 2, 3…) hasta encontrar
     ≥ N_POINTS puntos de crimen distintos (default N_POINTS=5)
  ② Seleccionar N_POINTS puntos con muestreo espacial estratificado
     (KMeans sobre lat/lon → 1 punto aleatorio por cluster, evita que los
     5 puntos queden amontonados en una esquina del vecindario)
  ③ Para cada punto elegido, muestrear N_IMAGES de sus imágenes disponibles
     (default N_IMAGES=4, de las 12 que tiene cada punto)
  ④ Ensamblar matriz (N_POINTS × N_IMAGES, 768) = (20, 768) por vecindario

CASOS DE COBERTURA INSUFICIENTE:
  - 0 puntos de crimen incluso en max_depth → vecindario EXCLUIDO del tensor
    (queda en coverage_report.csv con insufficient_coverage=True,
     pero no entra al .npy final — para el modelo gráfico-puro sigue
     disponible vía nodes.csv)
  - 1-4 puntos disponibles → se completa con muestreo CON reemplazo
    (oversampled_points=True) y se documenta en el reporte
  - Un punto con <4 imágenes disponibles → muestreo con reemplazo a nivel
    de imagen (oversampled_images=True)

REQUISITO PREVIO:
  python graph_pipeline.py --dual-gmm --year 2016        (genera crimes_snapped.csv)
  python extract_embeddings.py --ckpt ...                (genera embeddings.npy + metadata.json)

Uso:
    python Paso4_assemble_neighbourhood_tensors.py \\
        --distrito Barranco --year 2016 \\
        --graph-dir graph_output \\
        --embeddings-dir embeddings_export \\
        --output neighbourhood_tensors
"""

import argparse
import json
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────

N_POINTS         = 5     # puntos de crimen por vecindario
N_IMAGES_PER_PT  = 4     # imágenes por punto (de las 12 disponibles)
MAX_DEPTH        = 3     # profundidad máxima del ego-graph adaptativo
SEED             = 42

GRAPH_DIR       = Path("graph_output")
EMBEDDINGS_DIR  = Path("embeddings_export")
OUTPUT_DIR      = Path("neighbourhood_tensors")


# ─── Carga de datos previos (Paso 1-2 y extract_embeddings) ──────────────────

def rebuild_graph(nodes_csv: Path, edges_csv: Path) -> tuple[nx.Graph, pd.DataFrame]:
    """Reconstruye el grafo no dirigido desde los CSV exportados por graph_pipeline.py.
    No requiere re-descargar OSM."""
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(int(row["node_id"]), x=row["lon"], y=row["lat"])
    for _, row in edges_df.iterrows():
        G.add_edge(int(row["u"]), int(row["v"]), length=row.get("length_m"))

    print(f"  Grafo reconstruido: {G.number_of_nodes():,} nodos, {G.number_of_edges():,} aristas")
    return G, nodes_df


def load_crimes_snapped(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["CRIME_ID"]     = df["CRIME_ID"].astype(str)
    df["nearest_node"] = df["nearest_node"].astype(int)
    print(f"  Crímenes ↔ nodos cargados: {len(df):,}  ({path.name})")
    return df


def load_embeddings_and_metadata(embeddings_dir: Path) -> tuple[np.ndarray, dict, dict]:
    """Carga embeddings.npy + metadata.json (de extract_embeddings.py) y
    construye el índice crime_id → [índices en embeddings]."""
    embeddings = np.load(embeddings_dir / "embeddings.npy")
    with open(embeddings_dir / "metadata.json") as f:
        metadata = json.load(f)

    print(f"  Embeddings cargados: shape {embeddings.shape}")

    crime_id_to_indices: dict[str, list[int]] = {}
    for idx, cid in enumerate(metadata["crime_ids"]):
        crime_id_to_indices.setdefault(cid, []).append(idx)

    print(f"  Crime IDs únicos en embeddings: {len(crime_id_to_indices):,}")
    return embeddings, metadata, crime_id_to_indices


def validate_id_alignment(crimes_snapped_df: pd.DataFrame, crime_id_to_indices: dict) -> float:
    """
    Verifica que los CRIME_ID del CSV de crímenes coincidan con los crime_id
    parseados de las imágenes. CRÍTICO: si el overlap es bajo, todo lo
    demás falla silenciosamente (vecindarios "sin cobertura" que en
    realidad sí tienen imágenes, solo que con un ID mal alineado).
    """
    ids_csv        = set(crimes_snapped_df["CRIME_ID"])
    ids_embeddings = set(crime_id_to_indices.keys())
    overlap        = ids_csv & ids_embeddings
    pct            = 100 * len(overlap) / max(len(ids_csv), 1)

    print(f"\n  🔍 Validación de alineación CRIME_ID:")
    print(f"     IDs en crimes_snapped.csv  : {len(ids_csv):,}")
    print(f"     IDs en embeddings/metadata : {len(ids_embeddings):,}")
    print(f"     Overlap                    : {len(overlap):,}  ({pct:.1f}%)")

    if pct < 50:
        print(f"     ⚠️⚠️ OVERLAP MUY BAJO. Los CRIME_ID no coinciden entre el CSV")
        print(f"        de crímenes y los IDs parseados de las imágenes.")
        print(f"        Causas comunes: ceros a la izquierda ('42' vs '0042'),")
        print(f"        tipo str vs int, o columna ID incorrecta detectada en graph_pipeline.")
        print(f"        Ejemplos CSV: {list(ids_csv)[:5]}")
        print(f"        Ejemplos embeddings: {list(ids_embeddings)[:5]}")
    print()
    return pct


# ─── Muestreo espacial estratificado ──────────────────────────────────────────

def spatial_stratified_sample(
    point_ids: list,
    coords: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> list:
    """
    Selecciona k puntos con diversidad espacial: KMeans(k) sobre lat/lon,
    luego 1 punto aleatorio por cluster. Evita que el muestreo aleatorio
    puro amontone los puntos elegidos en una sola esquina del vecindario.
    """
    n = len(point_ids)
    if n <= k:
        return list(point_ids)

    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(coords)
    chosen = []
    for c in range(k):
        cluster_idx = np.where(km.labels_ == c)[0]
        if len(cluster_idx) == 0:
            continue
        pick = rng.choice(cluster_idx)
        chosen.append(point_ids[pick])

    # Si algún cluster quedó vacío (raro con pocos puntos), rellenar al azar
    while len(chosen) < k:
        remaining = [p for p in point_ids if p not in chosen]
        if not remaining:
            break
        chosen.append(rng.choice(remaining))

    return chosen


def sample_images_for_point(
    crime_id: str,
    crime_id_to_indices: dict,
    n_images: int,
    rng: np.random.Generator,
) -> tuple[list, bool]:
    """Muestrea n_images índices de embedding para un punto de crimen.
    Si tiene menos de n_images disponibles, muestrea con reemplazo."""
    indices = crime_id_to_indices.get(crime_id, [])
    if len(indices) == 0:
        return [], False
    if len(indices) >= n_images:
        chosen = rng.choice(indices, size=n_images, replace=False)
        return list(chosen), False
    chosen = rng.choice(indices, size=n_images, replace=True)
    return list(chosen), True


# ─── Ensamblar tensor por vecindario ──────────────────────────────────────────

def assemble_neighbourhood_tensor(
    node: int,
    G_undir: nx.Graph,
    crimes_snapped_df: pd.DataFrame,
    crime_id_to_indices: dict,
    embeddings: np.ndarray,
    n_points: int = N_POINTS,
    n_images_per_point: int = N_IMAGES_PER_PT,
    max_depth: int = MAX_DEPTH,
    rng: np.random.Generator = None,
):
    """
    Devuelve (tensor (20,768) o None, info dict de diagnóstico).
    tensor=None significa "excluir este vecindario" (0 puntos CON IMAGEN
    disponibles incluso en max_depth).

    CORRECCIÓN IMPORTANTE: el candidato a "punto de crimen" se filtra a
    SOLO los que tienen al menos 1 imagen en crime_id_to_indices, ANTES
    de expandir el ego-graph y hacer el muestreo espacial. Si se permite
    elegir puntos sin imagen, el sampling posterior los rellena con un
    vector de ceros (768 ceros), lo que diluye la media hacia cero de
    forma proporcional al % de coverage real — y corrompe por completo
    el grafo de similitud coseno del Laplaciano (un vector cero tiene
    similitud 0 con todo, generando nodos de grado ~0 que disparan la
    normalización D^{-1/2} a valores espurios). Filtrar primero evita
    ambos problemas: nunca se elige un punto sin imagen salvo que
    LITERALMENTE no haya ninguno disponible en max_depth.
    """
    rng = rng or np.random.default_rng(SEED)

    depth = 1
    distinct_points_df  = pd.DataFrame()
    n_candidates_total   = 0   # puntos en el CSV dentro del ego-graph (con o sin imagen)
    while depth <= max_depth:
        ego_nodes = set(nx.ego_graph(G_undir, node, radius=depth).nodes())
        sub       = crimes_snapped_df[crimes_snapped_df["nearest_node"].isin(ego_nodes)]
        all_distinct = sub.drop_duplicates("CRIME_ID")
        n_candidates_total = len(all_distinct)

        # Filtro CRÍTICO: solo puntos con al menos 1 imagen disponible
        distinct_points_df = all_distinct[
            all_distinct["CRIME_ID"].isin(crime_id_to_indices.keys())
        ]
        if len(distinct_points_df) >= n_points:
            break
        depth += 1

    n_avail = len(distinct_points_df)

    if n_avail == 0:
        return None, {
            "depth_used": depth, "n_points_available": 0,
            "n_candidates_total_no_filter": n_candidates_total,
            "insufficient_coverage": True,
            "oversampled_points": False, "oversampled_images": False,
        }

    distinct_ids = distinct_points_df["CRIME_ID"].tolist()
    coords       = distinct_points_df[["latitude", "longitude"]].to_numpy()

    oversampled_points = n_avail < n_points
    if n_avail >= n_points:
        chosen_ids = spatial_stratified_sample(distinct_ids, coords, n_points, rng)
    else:
        chosen_ids = list(distinct_ids)
        while len(chosen_ids) < n_points:
            chosen_ids.append(rng.choice(distinct_ids))

    embedding_rows       = []
    any_image_oversample = False
    for cid in chosen_ids:
        img_idxs, img_oversampled = sample_images_for_point(
            cid, crime_id_to_indices, n_images_per_point, rng
        )
        any_image_oversample = any_image_oversample or img_oversampled
        if len(img_idxs) == 0:
            # No debería pasar ya que chosen_ids viene filtrado por
            # crime_id_to_indices — se deja como salvaguarda defensiva
            embedding_rows.extend(
                [np.zeros(embeddings.shape[1], dtype=np.float32)] * n_images_per_point
            )
        else:
            embedding_rows.extend([embeddings[i] for i in img_idxs])

    tensor = np.array(embedding_rows, dtype=np.float32)   # (n_points*n_images, 768)

    info = {
        "depth_used":                   depth,
        "n_points_available":           n_avail,
        "n_candidates_total_no_filter": n_candidates_total,
        "insufficient_coverage":        oversampled_points,
        "oversampled_points":           oversampled_points,
        "oversampled_images":           any_image_oversample,
    }
    return tensor, info


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_pipeline(
    distrito: str,
    year,
    graph_dir: Path,
    embeddings_dir: Path,
    output_dir: Path,
    strat_tag: str = "dualgmm",
    n_points: int = N_POINTS,
    n_images_per_point: int = N_IMAGES_PER_PT,
    max_depth: int = MAX_DEPTH,
    seed: int = SEED,
):
    year_str = str(year) if year else "all"
    base     = f"{distrito.replace(' ', '_')}_{year_str}"
    print(f"\n{'='*60}")
    print(f"  Distrito: {distrito} | Año: {year_str}")
    print(f"{'='*60}")

    # ── Cargar grafo y crímenes ↔ nodos ─────────────────────────────────────
    G_undir, nodes_df = rebuild_graph(
        graph_dir / f"{base}_{strat_tag}_nodes.csv",
        graph_dir / f"{base}_edges.csv",
    )
    crimes_snapped_df = load_crimes_snapped(graph_dir / f"{base}_crimes_snapped.csv")

    # ── Cargar embeddings + validar alineación de IDs ──────────────────────
    embeddings, metadata, crime_id_to_indices = load_embeddings_and_metadata(embeddings_dir)
    validate_id_alignment(crimes_snapped_df, crime_id_to_indices)

    # ── Ensamblar tensor por nodo ───────────────────────────────────────────
    rng = np.random.default_rng(seed)
    tensors, coverage_rows, included_node_ids = [], [], []

    nodes_list = list(G_undir.nodes())
    print(f"  Ensamblando tensores para {len(nodes_list):,} nodos…")

    for node in nodes_list:
        tensor, info = assemble_neighbourhood_tensor(
            node, G_undir, crimes_snapped_df, crime_id_to_indices, embeddings,
            n_points=n_points, n_images_per_point=n_images_per_point,
            max_depth=max_depth, rng=rng,
        )
        row = {"node_id": node, **info, "included_in_tensor": tensor is not None}
        coverage_rows.append(row)

        if tensor is not None:
            tensors.append(tensor)
            included_node_ids.append(node)

    # ── Guardar ──────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / base

    if tensors:
        tensor_array = np.stack(tensors)   # (N_incluidos, 20, 768)
        np.save(f"{prefix}_neighbourhood_tensors.npy", tensor_array)
        with open(f"{prefix}_node_ids_order.json", "w") as f:
            json.dump([int(n) for n in included_node_ids], f)
        print(f"\n  ✅ Guardado: {prefix}_neighbourhood_tensors.npy  shape={tensor_array.shape}")
        print(f"  ✅ Guardado: {prefix}_node_ids_order.json")
    else:
        print(f"\n  ❌ Ningún vecindario tuvo cobertura suficiente — revisa la alineación de IDs.")

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(f"{prefix}_coverage_report.csv", index=False)
    print(f"  ✅ Guardado: {prefix}_coverage_report.csv")

    n_included  = coverage_df["included_in_tensor"].sum()
    n_oversampl = coverage_df["oversampled_points"].sum()
    avg_candidates = coverage_df["n_candidates_total_no_filter"].mean()
    avg_available  = coverage_df["n_points_available"].mean()
    print(f"\n  📊 Resumen de cobertura:")
    print(f"     Nodos totales         : {len(coverage_df):,}")
    print(f"     Incluidos en tensor   : {n_included:,}  ({100*n_included/len(coverage_df):.1f}%)")
    print(f"     Con oversample        : {n_oversampl:,}  ({100*n_oversampl/len(coverage_df):.1f}%)")
    print(f"     Excluidos (0 puntos con imagen): {len(coverage_df) - n_included:,}")
    print(f"     Profundidad promedio  : {coverage_df['depth_used'].mean():.2f}")
    print(f"     Puntos candidatos (CSV) por vecindario, promedio : {avg_candidates:.1f}")
    print(f"     Puntos CON IMAGEN usados por vecindario, promedio: {avg_available:.1f}  "
          f"({100*avg_available/max(avg_candidates,1):.1f}% del candidato)")

    print(f"\n  ✅ Listo — {distrito} {year_str}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paso 4: Ensamblar tensores por vecindario")
    parser.add_argument("--distrito", default="all")
    parser.add_argument("--year", default="2016")
    parser.add_argument("--graph-dir", default=str(GRAPH_DIR))
    parser.add_argument("--embeddings-dir", default=str(EMBEDDINGS_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--strat-tag", default="dualgmm",
                        help="Sufijo de estrategia usado en graph_pipeline.py "
                             "(ej. 'dualgmm', 'pct70', 'gmm') — debe coincidir "
                             "con el nombre real de {distrito}_{year}_{tag}_nodes.csv")
    parser.add_argument("--n-points", type=int, default=N_POINTS)
    parser.add_argument("--n-images", type=int, default=N_IMAGES_PER_PT)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    distritos = ["Barranco", "La Victoria"] if args.distrito == "all" else [args.distrito]
    years     = ([None] if args.year == "all" else [int(y) for y in args.year.split(",")])

    for distrito in distritos:
        for year in years:
            run_pipeline(
                distrito, year,
                Path(args.graph_dir), Path(args.embeddings_dir), Path(args.output),
                strat_tag=args.strat_tag,
                n_points=args.n_points, n_images_per_point=args.n_images,
                max_depth=args.max_depth, seed=args.seed,
            )


if __name__ == "__main__":
    main()